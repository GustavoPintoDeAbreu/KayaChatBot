"""Inbound side of the WhatsApp bridge: WAHA webhook → model → reply.

This module is intentionally model-agnostic and IO-light so the routing rules can
be unit-tested without a GPU or a real WhatsApp number. It depends on:

  * a ``responder`` callable ``(message, speaker, recent_lines) -> str`` — in
    production this wraps ``KayaEngine.generate_reply``; in tests it's a stub;
  * a ``waha_client`` with ``send_text`` — real (``WahaClient``) or
    ``MockWahaClient``;
  * a ``KeyedSessionMemory`` for per-chat history.

Routing rules (matching the chosen behaviour):
  * **DM** (chat id without ``@g.us``): always respond.
  * **Group**: respond only when the bot is **@-mentioned** or when the message
    **replies to one of the bot's own messages**. Never reply to itself.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.chat.memory import KeyedSessionMemory

logger = logging.getLogger(__name__)

Responder = Callable[[str, str, List[str]], str]


@dataclass
class InboundMessage:
    """Normalized view of a WAHA ``message`` event, engine-shape agnostic."""

    chat_id: str
    sender_id: str
    sender_name: str
    text: str
    is_group: bool
    from_me: bool
    message_id: str
    timestamp: Optional[int] = None
    sender_phone: str = ""
    mentioned_ids: List[str] = field(default_factory=list)
    reply_to_participant: Optional[str] = None


def _normalize_jid(value: Optional[str]) -> str:
    """Lower-case and strip a JID so identity comparisons are robust."""
    return (value or "").strip().lower()


def _phone_from_alt(alt: Optional[str]) -> str:
    """Pull the bare phone number from a ``...@s.whatsapp.net``/``@c.us`` JID."""
    if not alt:
        return ""
    return _normalize_jid(alt).split("@", 1)[0]


def _context_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """Find the Baileys ``contextInfo`` (mentions/quotes) inside ``_data.message``.

    NOWEB nests it under the message-type key (``extendedTextMessage`` etc.), so we
    scan the message dict rather than assume one shape.
    """
    message = data.get("message")
    if not isinstance(message, dict):
        return {}
    for value in message.values():
        if isinstance(value, dict) and isinstance(value.get("contextInfo"), dict):
            return value["contextInfo"]
    return {}


def parse_waha_message(event: Dict[str, Any]) -> Optional[InboundMessage]:
    """Parse a WAHA webhook event into an ``InboundMessage``.

    Returns ``None`` for non-``message`` events (status, presence, etc.). Field
    access is defensive because payload keys vary across WAHA engines: NOWEB
    (Baileys, used here) addresses chats by ``@lid``, carries the sender name at
    ``_data.pushName``, mentions at ``_data.message.*.contextInfo.mentionedJid``
    and the real phone at ``_data.key.{remoteJidAlt,participantAlt}``; WEBJS uses
    top-level ``mentionedIds``/``notifyName``. All of that is centralized here.
    """
    if event.get("event") != "message":
        return None
    payload = event.get("payload") or {}

    chat_id = str(payload.get("from") or "")
    if not chat_id:
        return None
    is_group = chat_id.endswith("@g.us")

    data = payload.get("_data") or {}
    key = data.get("key") or {}
    context = _context_info(data)

    # In groups the author is ``participant``/``author``; in DMs it's the chat itself.
    sender_id = str(payload.get("participant") or payload.get("author") or chat_id)
    # The real phone (NOWEB exposes it as the ``...Alt`` JID alongside the @lid id).
    sender_phone = _phone_from_alt(key.get("participantAlt") or key.get("remoteJidAlt"))

    reply_to = payload.get("replyTo") or payload.get("quotedMsg") or {}
    reply_to_participant = None
    if isinstance(reply_to, dict):
        reply_to_participant = reply_to.get("participant") or reply_to.get("author")
    reply_to_participant = reply_to_participant or context.get("participant")

    mentioned = (
        payload.get("mentionedIds")
        or payload.get("mentions")
        or context.get("mentionedJid")
        or data.get("mentionedJidList")
        or []
    )
    if not isinstance(mentioned, list):
        mentioned = []

    sender_name = (
        payload.get("notifyName")
        or payload.get("pushName")
        or data.get("pushName")
        or ""
    )

    try:
        timestamp = int(payload.get("timestamp")) if payload.get("timestamp") is not None else None
    except (TypeError, ValueError):
        timestamp = None

    return InboundMessage(
        chat_id=chat_id,
        sender_id=sender_id,
        sender_name=str(sender_name),
        text=str(payload.get("body") or ""),
        is_group=is_group,
        from_me=bool(payload.get("fromMe", False)),
        message_id=str(payload.get("id") or ""),
        timestamp=timestamp,
        sender_phone=sender_phone,
        mentioned_ids=[_normalize_jid(m) for m in mentioned],
        reply_to_participant=_normalize_jid(reply_to_participant) if reply_to_participant else None,
    )


class WhatsAppAdapter:
    """Decides whether/how to reply and wires history + engine + WAHA together."""

    def __init__(
        self,
        responder: Responder,
        waha_client: Any,
        config: Dict[str, Any],
        session_store: Optional[KeyedSessionMemory] = None,
    ):
        wcfg = config.get("whatsapp", {}) or {}
        self.responder = responder
        self.waha_client = waha_client
        self.bot_jid = _normalize_jid(wcfg.get("bot_jid", ""))
        # The bot is reachable under several identities (its @c.us number AND its
        # NOWEB @lid). Group mentions/replies may reference any of them.
        self.bot_jids = {self.bot_jid} if self.bot_jid else set()
        group_cfg = wcfg.get("group", {}) or {}
        self.respond_on_mention = bool(group_cfg.get("respond_on_mention", True))
        self.respond_on_reply = bool(group_cfg.get("respond_on_reply", True))
        # phone/JID -> display name, so the model knows who is speaking
        self.contacts = {_normalize_jid(k): v for k, v in (wcfg.get("contacts", {}) or {}).items()}
        self.history_turns = int(wcfg.get("history_turns", 10))
        self.send_seen = bool(wcfg.get("send_seen", True))
        # Messages older than this (unix seconds) are ignored — set on startup so a
        # reconnecting WAHA replaying backlog doesn't make the bot answer stale msgs.
        self.ignore_before_ts = 0
        self.session_store = session_store or KeyedSessionMemory(
            base_dir=wcfg.get("sessions_dir", "data/whatsapp_sessions"),
            max_lines=max(2 * self.history_turns, 20),
        )

    # ── decision ────────────────────────────────────────────────────────────
    def should_respond(self, msg: InboundMessage) -> bool:
        if msg.from_me:
            return False
        if not msg.text.strip():
            return False
        if self.ignore_before_ts and msg.timestamp and msg.timestamp < self.ignore_before_ts:
            return False
        if not msg.is_group:
            return True  # DM: always
        mentioned = (
            self.respond_on_mention
            and bool(self.bot_jids.intersection(msg.mentioned_ids))
        )
        replied = (
            self.respond_on_reply
            and msg.reply_to_participant is not None
            and msg.reply_to_participant in self.bot_jids
        )
        return bool(mentioned or replied)

    # ── speaker identity ──────────────────────────────────────────────────────
    def resolve_speaker(self, msg: InboundMessage) -> str:
        """Map the sender to a known Kaya member name when possible.

        Falls back to the WhatsApp push name, then a generic label, so the model
        always gets a usable ``"<who>: <text>"`` and RAG person-filtering can fire.
        """
        id_local = msg.sender_id.split("@", 1)[0]
        for candidate in (
            _normalize_jid(msg.sender_id),
            _normalize_jid(f"{msg.sender_phone}@c.us") if msg.sender_phone else "",
            msg.sender_phone,
            _normalize_jid(f"{id_local}@c.us"),
            id_local,
        ):
            if candidate and candidate in self.contacts:
                return self.contacts[candidate]
        return msg.sender_name or "Alguém"

    def _strip_bot_mention(self, text: str) -> str:
        """Remove an ``@<botnumber/lid>`` token so it doesn't pollute the prompt."""
        cleaned = text
        for jid in self.bot_jids:
            number = jid.split("@", 1)[0]
            if number:
                cleaned = re.sub(rf"@{re.escape(number)}\b", "", cleaned)
        return cleaned.strip()

    # ── main entry ─────────────────────────────────────────────────────────────
    def handle_event(self, event: Dict[str, Any], system_prompt: str = "") -> Optional[Dict[str, Any]]:
        """Process one webhook event end-to-end. Returns a result dict or ``None``.

        ``None`` means "ignored" (not a message, from self, empty, or a group
        message that didn't address the bot). The caller (server/simulator)
        supplies the ``system_prompt`` to use.
        """
        # Learn the bot's own identities from the webhook envelope so group
        # @-mention/reply detection works. NOWEB mentions the bot by its @lid, so
        # we must track both ``me.id`` (its @c.us number) and ``me.lid``.
        me = event.get("me") or {}
        for ident in (me.get("id"), me.get("lid")):
            normalized = _normalize_jid(ident)
            if normalized:
                self.bot_jids.add(normalized)
                if not self.bot_jid:
                    self.bot_jid = normalized

        msg = parse_waha_message(event)
        if msg is None or not self.should_respond(msg):
            return None

        speaker = self.resolve_speaker(msg)
        text = self._strip_bot_mention(msg.text)
        if not text:
            return None

        if self.send_seen:
            self.waha_client.send_seen(msg.chat_id)
            self.waha_client.start_typing(msg.chat_id)

        recent = self.session_store.recent(msg.chat_id, self.history_turns)
        try:
            reply = self.responder(text, speaker, recent)
        finally:
            if self.send_seen:
                self.waha_client.stop_typing(msg.chat_id)

        if not reply or not reply.strip():
            return None

        # Persist both sides so the next turn in this chat has context.
        self.session_store.append(msg.chat_id, f"{speaker}: {text}")
        self.session_store.append(msg.chat_id, f"Kaya Bot: {reply}")

        # Quote the asker's message in groups so it's clear who the bot answers.
        reply_to = msg.message_id if msg.is_group else None
        self.waha_client.send_text(msg.chat_id, reply, reply_to=reply_to)

        return {"chat_id": msg.chat_id, "speaker": speaker, "reply": reply}
