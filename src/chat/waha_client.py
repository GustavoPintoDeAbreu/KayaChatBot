"""Outbound side of the WhatsApp bridge.

WAHA (WhatsApp HTTP API, https://waha.devlike.pro) is a self-hosted Docker
container that wraps the WhatsApp Web protocol behind a REST API + webhooks. The
bridge receives inbound messages as webhooks (handled in ``whatsapp_adapter.py``)
and sends replies back through WAHA's ``/api/sendText`` endpoint.

``WahaClient`` talks to a real WAHA instance. ``MockWahaClient`` records what
*would* be sent instead — this is how the whole flow is developed and tested
before a real phone number / WAHA session exists. Both expose the same
``send_text`` / ``send_seen`` / ``start_typing`` surface so the adapter is
agnostic to which one it holds.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WahaClient:
    """Thin HTTP client for a running WAHA instance (lazy ``httpx`` import)."""

    def __init__(self, base_url: str, session: str = "default", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session = session
        self._headers = {"X-Api-Key": api_key} if api_key else {}
        import httpx  # imported lazily so tests/mock mode don't need it installed

        self._client = httpx.Client(base_url=self.base_url, headers=self._headers, timeout=30.0)

    def send_text(self, chat_id: str, text: str, reply_to: Optional[str] = None) -> Dict[str, Any]:
        """Send a text message. ``reply_to`` quotes a prior message id (groups)."""
        body: Dict[str, Any] = {"session": self.session, "chatId": chat_id, "text": text}
        if reply_to:
            body["reply_to"] = reply_to
        resp = self._client.post("/api/sendText", json=body)
        resp.raise_for_status()
        return resp.json()

    def send_seen(self, chat_id: str) -> None:
        try:
            self._client.post("/api/sendSeen", json={"session": self.session, "chatId": chat_id})
        except Exception as exc:  # noqa: BLE001 — presence is best-effort
            logger.debug("sendSeen failed: %s", exc)

    def start_typing(self, chat_id: str) -> None:
        try:
            self._client.post("/api/startTyping", json={"session": self.session, "chatId": chat_id})
        except Exception as exc:  # noqa: BLE001 — presence is best-effort
            logger.debug("startTyping failed: %s", exc)

    def stop_typing(self, chat_id: str) -> None:
        try:
            self._client.post("/api/stopTyping", json={"session": self.session, "chatId": chat_id})
        except Exception as exc:  # noqa: BLE001 — presence is best-effort
            logger.debug("stopTyping failed: %s", exc)


class MockWahaClient:
    """Drop-in replacement that captures sends instead of calling WhatsApp.

    Used for local development and tests: there is no real number, so every
    "reply" is appended to ``self.sent`` (and optionally echoed to stdout) so it
    can be inspected or asserted on.
    """

    def __init__(self, echo: bool = True):
        self.sent: List[Dict[str, Any]] = []
        self.seen: List[str] = []
        self.echo = echo

    def send_text(self, chat_id: str, text: str, reply_to: Optional[str] = None) -> Dict[str, Any]:
        record = {"chat_id": chat_id, "text": text, "reply_to": reply_to}
        self.sent.append(record)
        if self.echo:
            print(f"\n[→ WhatsApp {chat_id}] {text}\n")
        return {"mocked": True, **record}

    def send_seen(self, chat_id: str) -> None:
        self.seen.append(chat_id)

    def start_typing(self, chat_id: str) -> None:
        pass

    def stop_typing(self, chat_id: str) -> None:
        pass
