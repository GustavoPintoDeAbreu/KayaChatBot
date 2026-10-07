"""The owner's ``/homelaboff`` and ``/homelabon`` DMs go to the PC's power
listener, not to Kaya.

The GPU PC has no automatic shutdown; Gustavo turns it off by hand, and this
makes that possible from the phone, from anywhere. The gateway is the only
client of the PC's authenticated power listener (``kaya-power-listener`` on
:8099), and this tap is where the two are kept apart: a message it claims is
never journaled, never forwarded to the PC's ``/whatsapp/relay``, and never
answered by Kaya.

It claims, in DMs only:

- the owner's ``/homelaboff`` (ask what is running, then confirm) and
  ``/homelabon`` (ask the Pi to send Wake-on-LAN);
- the owner's ``yes``/``no`` replies to a pending shutdown question, and his
  ``cancel`` while a shutdown is waiting;
- the tool's own ``🖥️[homelab]`` echoes coming back as ``from_me``, which are
  dropped rather than written into Kaya's DM history.

A confirmation lasts five minutes and belongs to the chat that asked; a
``cancel`` is answered by the PC's listener, which settles the race with the
shutdown script under a lock.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Set, Tuple
from urllib.parse import urlsplit

import httpx

from src.chat.whatsapp_adapter import InboundMessage, _phone_from_alt, dm_allowed
from src.gateway.journal import Journal
from src.gateway.monitor import PcMonitor, PcState

logger = logging.getLogger(__name__)

HOMELAB_TAG = "🖥️[homelab]"
POWER_PORT = 8099
CONFIRM_TTL_SECONDS = 300.0
WAKE_TIMEOUT_SECONDS = 900.0
STALE_SECONDS = 300.0
ACTIVE_META_KEY = "homelab_off_chat"
WAKE_REQUEST_NAME = "pc-wake.request"
CONFIRM_WORDS = frozenset({"yes", "y", "sim", "s", "confirm", "confirmo"})
CANCEL_WORDS = frozenset({"no", "n", "não", "nao", "cancel", "cancela", "cancelar", "stop", "para"})

ASK_BUSY = ('Running on the PC:\n{items}\n\nKaya goes last: her replies in progress are finished first.\n'
            'Shut down once these finish? Reply "yes" within 5 min, or "no".')
ASK_IDLE = 'Nothing is running besides Kaya.\nShut down now? Reply "yes" within 5 min, or "no".'
NOTES = "\n\nNotes:\n{items}"
ALREADY_WAITING = 'A shutdown is already waiting for jobs. Reply "cancel" to stop it.'
ALREADY_STOPPING = "The PC is already shutting down."
ALREADY_OFF = "The PC is already off. /homelabon wakes it."
UNREACHABLE = "Couldn't reach the PC's power listener ({error}). Nothing was done."
STARTED = "OK. The PC shuts down once it's free; I'll keep you posted here. Reply \"cancel\" to stop."
DECLINED = "OK, the PC stays on."
CANCELLING = "Cancelling…"
TOO_LATE = "Too late to cancel: Kaya is being stopped and the PC is going down."
NOTHING_TO_CANCEL = "Nothing to cancel: no shutdown is running."
CANCEL_FAILED = "Couldn't reach the PC to cancel ({error})."
WAKE_SENT = "Wake-on-LAN sent. I'll tell you when the PC is up."
ALREADY_ON = "The PC is already on."
WAKE_FAILED = "Couldn't ask the Pi to send Wake-on-LAN ({error})."
PC_ONLINE = "The PC is on and Kaya is answering."
WAKE_TIMEOUT = "The PC isn't up after 15 min. Check the power and the BIOS Wake-on-LAN setting."

Call = Callable[[str, str, Dict[str, str]], Awaitable[Tuple[int, Dict[str, Any]]]]


async def _http_call(method: str, url: str, headers: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
    """The real request. 30 s: the PC's report runs a few checks, one of them sleeps 5 s."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.request(method, url, headers=headers)
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    return response.status_code, body


def _normalize_word(text: str) -> str:
    """Whole message, lower-cased, stripped of spaces and trailing '.!'."""
    return (text or "").strip().lower().rstrip(".!").strip()


@dataclass
class HomelabPower:
    """Claims the owner's power DMs and drives the PC's power listener."""

    owner_numbers: Set[str]
    pc_power_url: str
    token: str
    call: Optional[Call] = None
    _seen: Set[str] = field(default_factory=set)
    _pending: Dict[str, float] = field(default_factory=dict)      # chat_id -> asked at
    _waking: Dict[str, float] = field(default_factory=dict)       # chat_id -> requested at
    _journal: Optional[Journal] = None
    _monitor: Optional[PcMonitor] = None
    _send: Optional[Callable[[str, str, Optional[str]], Any]] = None
    _wake_request: Optional[Path] = None
    _now: Callable[[], float] = time.time

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Optional["HomelabPower"]:
        """The tap, or None when HOMELAB_OWNER_NUMBERS is unset (the feature is off)."""
        owners = {_phone_from_alt(number.strip().lstrip("+")) for number in
                  environ.get("HOMELAB_OWNER_NUMBERS", "").split(",") if number.strip()}
        if not owners:
            return None
        url = environ.get("HOMELAB_PC_POWER_URL", "").rstrip("/")
        if not url:
            pc_host = urlsplit(environ.get("KAYA_PC_URL", "http://192.168.1.149:7860")).hostname
            url = f"http://{pc_host}:{POWER_PORT}"
        token = environ.get("KAYA_RELAY_TOKEN", "")
        if not token:
            logger.error("HOMELAB_OWNER_NUMBERS is set but KAYA_RELAY_TOKEN is empty: "
                         "the PC's power listener will refuse every call")
        return cls(owners, url, token)

    def attach(self, *, journal: Journal, monitor: PcMonitor,
               send_text: Callable[[str, str, Optional[str]], Any],
               data_dir: str, now: Callable[[], float]) -> None:
        self._journal = journal
        self._monitor = monitor
        self._send = send_text
        self._wake_request = Path(data_dir) / WAKE_REQUEST_NAME
        self._now = now

    def _is_owner(self, msg: InboundMessage) -> bool:
        return dm_allowed(msg, True, self.owner_numbers)

    def active_chat(self) -> str:
        return self._journal.get_meta(ACTIVE_META_KEY, "") if self._journal else ""

    def _set_active(self, chat_id: str) -> None:
        if self._journal:
            self._journal.set_meta(ACTIVE_META_KEY, chat_id)

    def default_chat(self) -> str:
        return f"{sorted(self.owner_numbers)[0]}@c.us"

    def classify(self, msg: InboundMessage) -> Optional[str]:
        """``"off"``/``"on"``/``"confirm"``/``"decline"``/``"cancel"``/``"echo"``,
        or None to leave the message to Kaya."""
        if msg.is_group:
            return None
        text = (msg.text or "").strip()
        if msg.from_me:
            return "echo" if text.startswith(HOMELAB_TAG) else None
        if not self._is_owner(msg):
            return None
        first = text.split(maxsplit=1)[0].lower() if text else ""
        if first == "/homelaboff":
            return "off"
        if first == "/homelabon":
            return "on"
        word = _normalize_word(msg.text)
        asked = self._pending.get(msg.chat_id)
        if asked is not None and self._now() - asked <= CONFIRM_TTL_SECONDS:
            if word in CONFIRM_WORDS:
                return "confirm"
            if word in CANCEL_WORDS:
                return "decline"
        if word in CANCEL_WORDS and self.active_chat():
            return "cancel"
        return None

    async def handle(self, msg: InboundMessage, kind: str) -> Dict[str, Any]:
        if kind == "echo":
            return {"homelab": "echo"}
        if msg.message_id in self._seen:
            return {"homelab": "duplicate"}
        if len(self._seen) > 1000:
            self._seen.clear()
        self._seen.add(msg.message_id)
        if msg.timestamp is not None and self._now() - msg.timestamp > STALE_SECONDS:
            return {"homelab": "stale"}
        if kind == "off":
            return await self._off(msg)
        if kind == "confirm":
            return await self._confirm(msg)
        if kind == "decline":
            return await self._decline(msg)
        if kind == "cancel":
            return await self._cancel(msg)
        if kind == "on":
            return await self._on(msg)
        return {"homelab": "ignored"}

    def _say(self, chat_id: str, text: str, reply_to: Optional[str] = None) -> bool:
        try:
            self._send(chat_id, f"{HOMELAB_TAG} {text}", reply_to)
            return True
        except Exception:  # noqa: BLE001 — a failed send must not lose the flow
            logger.exception("homelab reply to %s failed", chat_id)
            return False

    def _headers(self) -> Dict[str, str]:
        return {"X-Relay-Token": self.token}

    async def _request(self, method: str, path: str) -> Tuple[int, Dict[str, Any]]:
        return await (self.call or _http_call)(method, f"{self.pc_power_url}{path}", self._headers())

    async def _off(self, msg: InboundMessage) -> Dict[str, Any]:
        try:
            status, body = await self._request("GET", "/power/status")
        except (httpx.HTTPError, OSError) as exc:
            if self._monitor.state in (PcState.OFFLINE, PcState.GOING_DOWN):
                await asyncio.to_thread(self._say, msg.chat_id, ALREADY_OFF, msg.message_id)
                return {"homelab": "off-already"}
            await asyncio.to_thread(self._say, msg.chat_id, UNREACHABLE.format(error=type(exc).__name__),
                                    msg.message_id)
            return {"homelab": "off-unreachable"}
        if status != 200:
            if self._monitor.state in (PcState.OFFLINE, PcState.GOING_DOWN):
                await asyncio.to_thread(self._say, msg.chat_id, ALREADY_OFF, msg.message_id)
                return {"homelab": "off-already"}
            await asyncio.to_thread(self._say, msg.chat_id, UNREACHABLE.format(error=f"HTTP {status}"),
                                    msg.message_id)
            return {"homelab": "off-unreachable"}
        manual = body.get("manual")
        if manual == "waiting":
            self._set_active(msg.chat_id)
            await asyncio.to_thread(self._say, msg.chat_id, ALREADY_WAITING, msg.message_id)
            return {"homelab": "off-waiting"}
        if manual == "stopping":
            await asyncio.to_thread(self._say, msg.chat_id, ALREADY_STOPPING, msg.message_id)
            return {"homelab": "off-stopping"}
        busy = body.get("busy") or []
        notes = body.get("notes") or []
        if busy:
            text = ASK_BUSY.format(items="\n".join(f"- {reason}" for reason in busy))
        else:
            text = ASK_IDLE
        if notes:
            text += NOTES.format(items="\n".join(f"- {note}" for note in notes))
        self._pending[msg.chat_id] = self._now()
        await asyncio.to_thread(self._say, msg.chat_id, text, msg.message_id)
        return {"homelab": "off-asked"}

    async def _confirm(self, msg: InboundMessage) -> Dict[str, Any]:
        self._pending.pop(msg.chat_id, None)
        try:
            status, _ = await self._request("POST", "/power/off")
        except (httpx.HTTPError, OSError) as exc:
            await asyncio.to_thread(self._say, msg.chat_id, UNREACHABLE.format(error=type(exc).__name__),
                                    msg.message_id)
            return {"homelab": "off-unreachable"}
        if status == 202:
            self._set_active(msg.chat_id)
            await asyncio.to_thread(self._say, msg.chat_id, STARTED, msg.message_id)
            return {"homelab": "off-started"}
        if status == 409:
            await asyncio.to_thread(self._say, msg.chat_id, ALREADY_WAITING, msg.message_id)
            return {"homelab": "off-waiting"}
        await asyncio.to_thread(self._say, msg.chat_id, UNREACHABLE.format(error=f"HTTP {status}"),
                                msg.message_id)
        return {"homelab": "off-unreachable"}

    async def _decline(self, msg: InboundMessage) -> Dict[str, Any]:
        self._pending.pop(msg.chat_id, None)
        await asyncio.to_thread(self._say, msg.chat_id, DECLINED, msg.message_id)
        return {"homelab": "off-declined"}

    async def _cancel(self, msg: InboundMessage) -> Dict[str, Any]:
        try:
            status, body = await self._request("POST", "/power/cancel")
        except (httpx.HTTPError, OSError) as exc:
            await asyncio.to_thread(self._say, msg.chat_id, CANCEL_FAILED.format(error=type(exc).__name__),
                                    msg.message_id)
            return {"homelab": "cancel-failed"}
        if status == 200:
            await asyncio.to_thread(self._say, msg.chat_id, CANCELLING, msg.message_id)
            return {"homelab": "cancel-requested"}
        if status == 409:
            if body.get("reason") == "too late":
                await asyncio.to_thread(self._say, msg.chat_id, TOO_LATE, msg.message_id)
                return {"homelab": "cancel-too-late"}
            self._set_active("")
            await asyncio.to_thread(self._say, msg.chat_id, NOTHING_TO_CANCEL, msg.message_id)
            return {"homelab": "cancel-none"}
        await asyncio.to_thread(self._say, msg.chat_id, CANCEL_FAILED.format(error=f"HTTP {status}"),
                                msg.message_id)
        return {"homelab": "cancel-failed"}

    async def _on(self, msg: InboundMessage) -> Dict[str, Any]:
        if self._monitor.state is PcState.ONLINE:
            await asyncio.to_thread(self._say, msg.chat_id, ALREADY_ON, msg.message_id)
            return {"homelab": "on-already"}
        try:
            self._wake_request.write_text(str(self._now()))
        except OSError as exc:
            await asyncio.to_thread(self._say, msg.chat_id, WAKE_FAILED.format(error=type(exc).__name__),
                                    msg.message_id)
            return {"homelab": "on-failed"}
        self._waking[msg.chat_id] = self._now()
        await asyncio.to_thread(self._say, msg.chat_id, WAKE_SENT, msg.message_id)
        return {"homelab": "on-requested"}

    def relay_update(self, text: str, final: bool) -> bool:
        """Deliver one update from the PC's shutdown script to the owner's chat."""
        chat = self.active_chat() or self.default_chat()
        sent = self._say(chat, text)
        if final:
            self._set_active("")
        return sent

    def tick(self) -> None:
        for chat, started in list(self._waking.items()):
            if self._monitor.state is PcState.ONLINE:
                self._say(chat, PC_ONLINE)
                del self._waking[chat]
            elif self._now() - started > WAKE_TIMEOUT_SECONDS:
                self._say(chat, WAKE_TIMEOUT)
                del self._waking[chat]
        stale = [chat for chat, asked in self._pending.items()
                 if self._now() - asked > CONFIRM_TTL_SECONDS]
        for chat in stale:
            del self._pending[chat]

    async def watch(self, stop: asyncio.Event, interval: float = 10.0) -> None:
        """Report wake results and drop expired confirmations until *stop* is set."""
        while not stop.is_set():
            await asyncio.to_thread(self.tick)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
