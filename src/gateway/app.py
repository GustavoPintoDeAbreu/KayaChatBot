"""The Raspberry Pi gateway: Kaya's always-on front door.

WAHA posts every WhatsApp event here. The gateway journals it, downloads its
media at once (WAHA deletes media 180 s after receipt), sends the offline reply
when the PC is really off, and wakes the forwarder that delivers events to the
PC's ``/whatsapp/relay`` in order. It also serves the public landing page and
``/status``. See deploy/pi/README.md.

Two listeners, on purpose: the public one (landing page and a reduced status) is
the only thing the Cloudflare tunnel reaches; the internal one (webhook, media,
going-down) is bound to the LAN and firewalled to the PC.

This module must never import torch or the model stack; the Pi cannot run it and
tests/gateway/test_torch_free.py fails if it tries.

    python -m src.gateway.app
"""
from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Set, Tuple
from urllib.parse import urlsplit

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from src.chat.stt import rewrite_media_url
from src.chat.whatsapp_adapter import (
    _normalize_jid,
    _phone_from_alt,
    is_addressed,
    parse_waha_message,
)
from src.gateway.autoreply import OfflineResponder
from src.gateway.forwarder import Forwarder
from src.gateway.journal import Journal
from src.gateway.monitor import PcMonitor, PcState
from src.gateway.schedule import PowerSchedule

logger = logging.getLogger(__name__)

_EVENT_TYPES = ("message", "message.reaction", "reaction")
# A message older than this on arrival is WAHA replaying its own backlog after a
# reconnect, not something somebody just sent: never auto-replied or deferred.
_BACKLOG_SECONDS = 300


@dataclass
class GatewaySettings:
    """Everything the gateway reads from its environment."""

    data_dir: str
    config_path: str
    landing_html: str
    whitelist_path: str
    pc_url: str
    relay_token: str
    waha_url: str
    waha_api_key: str
    waha_session: str
    webhook_token: str
    media_base_url: str
    public_port: int
    internal_port: int
    max_media_bytes: int

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "GatewaySettings":
        """Read the settings, with the defaults the Pi deployment uses."""
        env = os.environ if environ is None else environ
        return cls(
            data_dir=env.get("GATEWAY_DATA_DIR", "/data"),
            config_path=env.get("GATEWAY_CONFIG", "config.yaml"),
            landing_html=env.get("GATEWAY_LANDING_HTML", "src/chat/static/landing.html"),
            whitelist_path=env.get("GATEWAY_WHITELIST", "/config/whatsapp_whitelist.json"),
            pc_url=env.get("KAYA_PC_URL", "http://192.168.1.149:7860").rstrip("/"),
            relay_token=env.get("KAYA_RELAY_TOKEN", ""),
            waha_url=env.get("KAYA_WAHA_URL", "http://waha:3000").rstrip("/"),
            waha_api_key=env.get("KAYA_WAHA_API_KEY", ""),
            waha_session=env.get("KAYA_WAHA_SESSION", "default"),
            webhook_token=env.get("KAYA_WHATSAPP_WEBHOOK_TOKEN", ""),
            media_base_url=env.get("GATEWAY_MEDIA_BASE_URL", "http://192.168.1.238:8088").rstrip("/"),
            public_port=int(env.get("GATEWAY_PUBLIC_PORT", "8080")),
            internal_port=int(env.get("GATEWAY_INTERNAL_PORT", "8088")),
            max_media_bytes=int(float(env.get("GATEWAY_MAX_MEDIA_MB", "50")) * 1024 * 1024),
        )


class Gateway:
    """Journal, monitor, forwarder and offline responder behind two small apps."""

    def __init__(self, settings: GatewaySettings, *, journal: Optional[Journal] = None,
                 monitor: Optional[PcMonitor] = None,
                 send_text: Optional[Callable[[str, str, Optional[str]], Any]] = None,
                 fetch_media: Optional[Callable[[str], Awaitable[Tuple[bytes, str]]]] = None,
                 now: Callable[[], float] = time.time) -> None:
        self.settings = settings
        self._now = now
        with open(settings.config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        wcfg = config.get("whatsapp") or {}
        group = wcfg.get("group") or {}
        whitelist = wcfg.get("whitelist") or {}
        self._respond_on_mention = bool(group.get("respond_on_mention", True))
        self._respond_on_reply = bool(group.get("respond_on_reply", True))
        self._whitelist_enabled = bool(whitelist.get("enabled", False))
        self._whitelist_numbers = self._load_whitelist(whitelist.get("allowed") or [])
        self.schedule = PowerSchedule.from_config(config.get("power") or {})

        self.journal = journal or Journal(f"{settings.data_dir}/journal.sqlite3",
                                          f"{settings.data_dir}/media")
        self.monitor = monitor or PcMonitor(settings.pc_url)
        self._send_text = send_text or self._waha_send_text()
        self._fetch_media = fetch_media or self._download_media
        self.forwarder = Forwarder(self.journal, self.monitor, pc_url=settings.pc_url,
                                   relay_token=settings.relay_token,
                                   media_base_url=settings.media_base_url, now=now)
        self.responder = OfflineResponder(self.journal, self._send_text, self.schedule,
                                          self.monitor, now=now)
        self._bot_jids: Set[str] = set(json.loads(self.journal.get_meta("bot_jids", "[]")))
        self._stop = asyncio.Event()
        self._tasks: list = []
        if not settings.relay_token:
            logger.error("KAYA_RELAY_TOKEN is unset: events are journaled but the PC will refuse them")

        self.public_app = self._build_public_app()
        self.internal_app = self._build_internal_app()

    def _load_whitelist(self, configured: list) -> Set[str]:
        """The DM whitelist: config.yaml plus the gitignored runtime file."""
        numbers = list(configured)
        whitelist_file = Path(self.settings.whitelist_path)
        if whitelist_file.exists():
            numbers.extend(json.loads(whitelist_file.read_text(encoding="utf-8")).get("allowed", []))
        return {_phone_from_alt(str(number)) for number in numbers if number}

    def _waha_send_text(self) -> Callable[[str, str, Optional[str]], Any]:
        """WAHA's sendText, imported lazily so tests never need httpx.Client."""
        from src.chat.waha_client import WahaClient

        client = WahaClient(self.settings.waha_url, self.settings.waha_session,
                            self.settings.waha_api_key)
        return lambda chat_id, text, reply_to=None: client.send_text(chat_id, text, reply_to=reply_to)

    async def _download_media(self, url: str) -> Tuple[bytes, str]:
        """Fetch a media file from WAHA now, before WAHA deletes it."""
        headers = {"X-Api-Key": self.settings.waha_api_key} if self.settings.waha_api_key else {}
        async with httpx.AsyncClient(timeout=60, follow_redirects=True, headers=headers) as client:
            response = await client.get(rewrite_media_url(url, self.settings.waha_url))
            response.raise_for_status()
            if len(response.content) > self.settings.max_media_bytes:
                raise ValueError(f"media larger than {self.settings.max_media_bytes} bytes")
            return response.content, response.headers.get("content-type", "")

    def _learn_bot_ids(self, event: Dict[str, Any]) -> None:
        """Remember the bot's own ids from the envelope, as the adapter does."""
        me = event.get("me") or {}
        found = {_normalize_jid(me.get(key)) for key in ("id", "lid")} - {""}
        if not found <= self._bot_jids:
            self._bot_jids |= found
            self.journal.set_meta("bot_jids", json.dumps(sorted(self._bot_jids)))

    async def handle_webhook(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Journal one WAHA event, fetch its media, maybe reply offline, wake the forwarder."""
        event_type = event.get("event")
        if event_type not in _EVENT_TYPES:
            return {"ignored": event_type}
        self._learn_bot_ids(event)
        payload = event.get("payload") or {}
        message_id = str(payload.get("id") or "")
        dedup_key = (f"{event_type}:{message_id}" if message_id else f"{event_type}:"
                     + hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest())
        now = self._now()

        msg = None
        if event_type == "message":
            msg = parse_waha_message(event)
            if msg is None:
                return {"ignored": "unparseable"}
            chat_id, sender_id, wa_ts = msg.chat_id, msg.sender_id, msg.timestamp
            addressed = is_addressed(msg, self._bot_jids, self._respond_on_mention,
                                     self._respond_on_reply, self._whitelist_enabled,
                                     self._whitelist_numbers)
        else:
            chat_id = str(payload.get("from") or "")
            sender_id = str(payload.get("participant") or payload.get("from") or "")
            timestamp = payload.get("timestamp")
            wa_ts = timestamp if isinstance(timestamp, int) else None
            addressed = False
        backlog = wa_ts is not None and now - wa_ts > _BACKLOG_SECONDS

        seq = self.journal.append(event, dedup_key=dedup_key, event_type=event_type,
                                  chat_id=chat_id, sender_id=sender_id, message_id=message_id,
                                  wa_ts=wa_ts, received_at=now, addressed=addressed,
                                  backlog=backlog)
        if seq is None:
            return {"duplicate": True}

        if msg is not None and msg.media_url:
            try:
                data, mime = await self._fetch_media(msg.media_url)
                name = msg.media_filename or Path(urlsplit(msg.media_url).path).name or "media"
                self.journal.attach_media(seq, data, name, mime or msg.media_mimetype)
            except Exception as exc:  # noqa: BLE001 — a lost file must not lose the message
                logger.warning("media for seq=%d not stored: %s", seq, exc)
                self.journal.mark_media_failed(seq, str(exc))

        entry = self.journal.get(seq)
        if entry is not None:
            await asyncio.to_thread(self.responder.maybe_reply, entry)
        self.forwarder.wake()
        return {"seq": seq, "addressed": addressed}

    def status(self) -> Dict[str, Any]:
        """PC state, the schedule, and the journal's size."""
        now = datetime.datetime.fromtimestamp(self._now(), self.schedule.timezone)
        next_wake = self.schedule.next_wake(now)
        return {
            "pc": self.monitor.state.value,
            "pc_online": self.monitor.state is PcState.ONLINE,
            "next_wake": next_wake.isoformat(),
            "next_wake_hhmm": next_wake.strftime("%H:%M"),
            "scheduled_on": self.schedule.is_scheduled_on(now),
            "outage": self.responder.outage(),
            "journal": self.journal.stats(),
        }

    def _build_public_app(self) -> FastAPI:
        """What the internet sees: the landing page and a reduced status."""
        app = FastAPI(title="Kaya", docs_url=None, redoc_url=None, openapi_url=None)

        @app.get("/", response_class=HTMLResponse)
        def landing() -> HTMLResponse:
            page = Path(self.settings.landing_html)
            if not page.exists():
                raise HTTPException(status_code=404)
            return HTMLResponse(page.read_text(encoding="utf-8"))

        @app.get("/status")
        def public_status() -> JSONResponse:
            body = {key: value for key, value in self.status().items() if key != "journal"}
            return JSONResponse(body, headers={"Cache-Control": "no-store"})

        return app

    def _build_internal_app(self) -> FastAPI:
        """What the LAN sees: WAHA's webhook, media for the PC, the shutdown notice."""
        app = FastAPI(title="Kaya gateway", docs_url=None, redoc_url=None, openapi_url=None)
        settings = self.settings

        @app.post("/waha/webhook")
        async def webhook(request: Request, x_webhook_token: str = Header(default="")):
            if settings.webhook_token and x_webhook_token != settings.webhook_token:
                raise HTTPException(status_code=401, detail="invalid webhook token")
            return await self.handle_webhook(await request.json())

        @app.get("/media/{seq}/{name}")
        def media(seq: int, name: str, x_api_key: str = Header(default="")):
            if settings.waha_api_key and x_api_key != settings.waha_api_key:
                raise HTTPException(status_code=401, detail="invalid api key")
            entry = self.journal.get(seq)
            if (entry is None or entry.media_status != "stored"
                    or Path(entry.media_path).name != name):
                raise HTTPException(status_code=404)
            return FileResponse(entry.media_path, media_type=entry.media_mime or None)

        @app.post("/pc/going-down")
        def going_down(x_relay_token: str = Header(default="")):
            if not settings.relay_token or x_relay_token != settings.relay_token:
                raise HTTPException(status_code=401, detail="invalid relay token")
            self.monitor.announce_going_down()
            return {"ok": True}

        @app.post("/pc/degraded")
        def degraded(x_relay_token: str = Header(default="")):
            if not settings.relay_token or x_relay_token != settings.relay_token:
                raise HTTPException(status_code=401, detail="invalid relay token")
            self.monitor.mark_degraded()
            return {"ok": True}

        @app.get("/status")
        def internal_status():
            return self.status()

        @app.get("/health")
        def health():
            return {"ok": True}

        return app

    async def start_background(self) -> None:
        """Start the PC monitor and the forwarder."""
        self._stop.clear()
        self._tasks = [asyncio.create_task(self.monitor.run(self._stop)),
                       asyncio.create_task(self.forwarder.run(self._stop))]

    async def stop_background(self) -> None:
        """Stop them and wait."""
        self._stop.set()
        self.forwarder.wake()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def serve(self) -> None:
        """Run both listeners until cancelled."""
        import uvicorn

        await self.start_background()
        servers = [
            uvicorn.Server(uvicorn.Config(self.public_app, host="0.0.0.0",
                                          port=self.settings.public_port, log_level="info")),
            uvicorn.Server(uvicorn.Config(self.internal_app, host="0.0.0.0",
                                          port=self.settings.internal_port, log_level="info")),
        ]
        try:
            await asyncio.gather(*(server.serve() for server in servers))
        finally:
            await self.stop_background()


def main() -> None:
    """Entry point for the container."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(Gateway(GatewaySettings.from_env()).serve())


if __name__ == "__main__":
    main()
