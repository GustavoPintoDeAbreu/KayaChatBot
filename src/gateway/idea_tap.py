"""The owner's ``/idea`` messages go to the idea pipeline, not to Kaya.

The idea pipeline (~/Desktop/idea-pipeline) turns a feature idea sent from the
phone into a planned, implemented draft PR. It talks to its owner over Kaya's
WhatsApp session, because that session is the only one the lab has and must
stay the only one. This tap is where the two are kept apart: a message it
claims is never journaled, never forwarded to the PC, never answered offline
and so never becomes something "the group said" in Kaya's memory.

It claims, in DMs only:

- the owner's ``/idea`` and ``/ideas`` messages (new idea, status, answers);
- the owner's replies that quote one of the pipeline's ``💡[idea-N]`` messages,
  which is how questions get answered;
- the pipeline's own ``💡[idea-`` messages coming back as ``from_me`` echoes,
  which are dropped rather than written into Kaya's DM history.

The quoted body is not always in WAHA's payload, which is why the ``/idea N
<answer>`` form exists as well.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Set

import httpx

from src.chat.whatsapp_adapter import InboundMessage, _phone_from_alt

logger = logging.getLogger(__name__)

IDEA_TAG = "💡[idea-"
OFFLINE_TEXT = ("💡 A caixa de ideias está desligada agora; a ideia não foi guardada. "
                "Manda outra vez mais tarde.")
_COMMANDS = ("/idea", "/ideas")

Post = Callable[[str, Dict[str, Any], Dict[str, str]], Awaitable[int]]


@dataclass
class IdeaTap:
    """Decides which messages belong to the idea pipeline and hands them over."""

    inbox_url: str
    token: str
    owner_numbers: Set[str]
    post: Optional[Post] = None
    _warned: Set[str] = field(default_factory=set)

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Optional["IdeaTap"]:
        """The tap, or None when IDEA_INBOX_URL is unset (the tap is off)."""
        url = environ.get("IDEA_INBOX_URL", "").rstrip("/")
        if not url:
            return None
        owners = {_phone_from_alt(number.strip().lstrip("+")) for number in
                  environ.get("IDEA_OWNER_NUMBERS", "").split(",") if number.strip()}
        if not owners:
            logger.error("IDEA_INBOX_URL is set but IDEA_OWNER_NUMBERS is empty: the tap claims nothing")
        return cls(url, environ.get("IDEA_INBOX_TOKEN", ""), owners)

    def _is_owner(self, msg: InboundMessage) -> bool:
        """Whether the sender is the owner, however WhatsApp addressed them."""
        candidates = {_phone_from_alt(msg.sender_phone), _phone_from_alt(msg.sender_id)}
        return bool((candidates - {""}) & self.owner_numbers)

    def classify(self, msg: InboundMessage) -> Optional[str]:
        """``"idea"`` to forward, ``"echo"`` to drop, None to leave for Kaya."""
        if msg.is_group:
            return None
        text = (msg.text or "").strip()
        if msg.from_me:
            return "echo" if text.startswith(IDEA_TAG) else None
        if not self._is_owner(msg):
            return None
        first = text.split(maxsplit=1)[0].lower() if text else ""
        if first in _COMMANDS or (msg.quoted_text or "").lstrip().startswith(IDEA_TAG):
            return "idea"
        return None

    def payload(self, msg: InboundMessage) -> Dict[str, Any]:
        """What the inbox stores: enough to answer, nothing else."""
        return {"message_id": msg.message_id, "chat_id": msg.chat_id,
                "sender": _phone_from_alt(msg.sender_phone) or _phone_from_alt(msg.sender_id),
                "text": msg.text, "quoted_text": msg.quoted_text,
                "reply_to_id": msg.reply_to_id, "timestamp": msg.timestamp}

    async def forward(self, msg: InboundMessage) -> bool:
        """POST the message to the inbox. False when the inbox did not take it."""
        headers = {"X-Idea-Token": self.token} if self.token else {}
        try:
            status = await (self.post or _http_post)(f"{self.inbox_url}/ingest",
                                                     self.payload(msg), headers)
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("idea inbox unreachable for %s: %s", msg.message_id, exc)
            return False
        if status >= 400:
            logger.warning("idea inbox refused %s: HTTP %d", msg.message_id, status)
            return False
        return True

    def should_warn(self, msg: InboundMessage) -> bool:
        """Say "inbox offline" once per message, whatever WAHA redelivers."""
        if msg.message_id in self._warned:
            return False
        if len(self._warned) > 1000:
            self._warned.clear()
        self._warned.add(msg.message_id)
        return True


async def _http_post(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> int:
    """The real POST, with a short timeout: the webhook must not stall on it."""
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(url, json=body, headers=headers)
        return response.status_code
