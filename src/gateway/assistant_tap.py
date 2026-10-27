"""The owner's DMs go to the lab assistant, which decides what each one is for.

The lab assistant (~/Desktop/lab-assistant) is Gustavo's natural-language front door
to the lab: he talks (a voice note) or types in his DM with Kaya's number, and a local
model on the PC works out whether it is a question for the lab, a change to one of the
apps, an idea for the idea pipeline, or something for Kaya. It runs over Kaya's WhatsApp
session, because that session is the only one the lab has and must stay the only one.

It runs after the idea, homelab and help taps, so ``/idea``, ``/homelab*``, ``/help``, sudo answers and
quoted 💡 replies are unchanged. It claims, in the owner's DMs only:

- voice notes (any audio), downloaded here because WAHA deletes media;
- plain text that is not a ``/command`` and does not reply to a message of Kaya's
  (replying to Kaya keeps talking to Kaya);
- the assistant's own ``🤖`` echoes coming back as ``from_me``, which are dropped
  rather than written into Kaya's DM history.

What Kaya should answer comes back through ``POST /assistant/handback`` as text, and is
journaled like any other owner DM. When the inbox cannot be reached, the message is
left to Kaya, as before the assistant existed.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Set

import httpx

from src.chat.whatsapp_adapter import InboundMessage, _phone_from_alt

logger = logging.getLogger(__name__)

TAG = "🤖"
ASLEEP_TEXT = "🤖 The PC is asleep; I'll answer as soon as it wakes."

Post = Callable[[str, Dict[str, Any], Dict[str, str]], Awaitable[int]]


@dataclass
class AssistantTap:
    """Decides which messages belong to the lab assistant and hands them over."""

    inbox_url: str
    token: str
    owner_numbers: Set[str]
    post: Optional[Post] = None
    _warned: Set[str] = field(default_factory=set)

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Optional["AssistantTap"]:
        """The tap, or None when ASSISTANT_INBOX_URL is unset (the tap is off).

        The owner is the idea pipeline's (``IDEA_OWNER_NUMBERS``) unless
        ``ASSISTANT_OWNER_NUMBERS`` says otherwise.
        """
        url = environ.get("ASSISTANT_INBOX_URL", "").rstrip("/")
        if not url:
            return None
        raw = environ.get("ASSISTANT_OWNER_NUMBERS") or environ.get("IDEA_OWNER_NUMBERS", "")
        owners = {_phone_from_alt(number.strip().lstrip("+")) for number in raw.split(",") if number.strip()}
        if not owners:
            logger.error("ASSISTANT_INBOX_URL is set but no owner number: the assistant tap claims nothing")
        return cls(url, environ.get("ASSISTANT_INBOX_TOKEN", ""), owners)

    def _is_owner(self, msg: InboundMessage) -> bool:
        candidates = {_phone_from_alt(msg.sender_phone), _phone_from_alt(msg.sender_id)}
        return bool((candidates - {""}) & self.owner_numbers)

    @staticmethod
    def is_voice(msg: InboundMessage) -> bool:
        return bool(msg.media_url) and msg.media_mimetype.lower().startswith("audio/")

    def classify(self, msg: InboundMessage) -> Optional[str]:
        """``"assistant"`` to forward, ``"echo"`` to drop, None to leave for Kaya."""
        if msg.is_group:
            return None
        text = (msg.text or "").strip()
        if msg.from_me:
            return "echo" if text.startswith(TAG) else None
        if not self._is_owner(msg):
            return None
        if self.is_voice(msg):
            return "assistant"
        if not text or msg.media_url or text.startswith("/"):
            return None
        quoted = (msg.quoted_text or "").strip()
        if quoted and not quoted.startswith(TAG):
            return None
        return "assistant"

    def payload(self, msg: InboundMessage, event: Dict[str, Any], audio: bytes = b"",
                mime: str = "") -> Dict[str, Any]:
        """What the inbox stores: the message, the audio, and the event (for a handback)."""
        return {"message_id": msg.message_id, "chat_id": msg.chat_id,
                "sender": _phone_from_alt(msg.sender_phone) or _phone_from_alt(msg.sender_id),
                "text": msg.text, "quoted_text": msg.quoted_text, "reply_to_id": msg.reply_to_id,
                "timestamp": msg.timestamp, "audio_b64": base64.b64encode(audio).decode("ascii") if audio else "",
                "audio_mime": mime or (msg.media_mimetype if audio else ""), "event": event}

    async def forward(self, msg: InboundMessage, event: Dict[str, Any], audio: bytes = b"",
                      mime: str = "") -> bool:
        """POST the message to the inbox. False when the inbox did not take it."""
        headers = {"X-Assistant-Token": self.token} if self.token else {}
        try:
            status = await (self.post or _http_post)(f"{self.inbox_url}/ingest",
                                                     self.payload(msg, event, audio, mime), headers)
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("assistant inbox unreachable for %s: %s", msg.message_id, exc)
            return False
        if status >= 400:
            logger.warning("assistant inbox refused %s: HTTP %d", msg.message_id, status)
            return False
        return True

    def should_warn(self, msg: InboundMessage) -> bool:
        """Say "PC asleep" once per message, whatever WAHA redelivers."""
        if msg.message_id in self._warned:
            return False
        if len(self._warned) > 1000:
            self._warned.clear()
        self._warned.add(msg.message_id)
        return True


def handback_event(event: Dict[str, Any], text: str, now: float) -> Dict[str, Any]:
    """The owner's original event, as a text DM saying ``text``: no media (the words are
    already transcribed), and stamped now so a message that waited for the PC is not
    treated as backlog."""
    out = dict(event)
    payload = dict(out.get("payload") or {})
    payload["body"] = text
    payload["hasMedia"] = False
    payload.pop("media", None)
    payload["timestamp"] = int(now)
    out["payload"] = payload
    return out


async def _http_post(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> int:
    """The real POST, with a short timeout: the webhook must not stall on it."""
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(url, json=body, headers=headers)
        return response.status_code
