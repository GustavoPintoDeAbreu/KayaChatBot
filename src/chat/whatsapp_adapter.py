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

import datetime
import json
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.chat.memory import ChatPreferences, KeyedSessionMemory
from src.chat.scope import scope_for_chat
from src.data.message_log import MessageLog
from src.chat.waha_client import extract_sent_id

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
    # Voice notes arrive with empty text; the audio has to be fetched and
    # transcribed before the message means anything.
    media_url: str = ""
    media_mimetype: str = ""
    # Filled in once a photo has been read by the vision model.
    image_description: str = ""


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
        # A voice note carries no body text — the words live in this file, which
        # has to be fetched and transcribed before the message means anything.
        media_url=str((payload.get("media") or {}).get("url") or ""),
        media_mimetype=str((payload.get("media") or {}).get("mimetype") or ""),
    )


# Words that make an image request point at a picture already in the chat rather
# than describe a new one from scratch. "põe-LHE uma coroa" and "edita ESTA foto"
# refer; "faz uma imagem de um gato astronauta" does not. Without this test every
# request became an edit of whatever photo was last posted.
# Only *referring* words count. A bare "imagem" or "foto" does not refer to
# anything — "faz uma imagem de um gato astronauta" describes a new picture — so
# matching those nouns is what turned every request into an edit.
_REFERS_TO_IMAGE = re.compile(
    r"\b(lhe|lhes|nele|nela|dele|dela|isto|isso|est[ae]s?|aquel[ae]s?)\b"
    r"|\b(nesta|nessa|naquela|desta|dessa)\s+(foto|imagem|fotografia)\b"
    r"|\b(this|that|him|her)\b"
    r"|\bthe\s+(photo|picture|image)\b",
    re.IGNORECASE,
)


def refers_to_existing_image(text: str) -> bool:
    """Whether an image request points at a picture already in the conversation.

    Deliberately conservative, because the two mistakes are not symmetric:
    wrongly generating gives someone a picture they did not ask for, while
    wrongly editing puts a real person's face into an invented scene nobody
    requested. So an ambiguous request generates, and anyone who genuinely wants
    an edit can attach the photo — which skips this test entirely, being
    unambiguous.
    """
    return bool(_REFERS_TO_IMAGE.search(text or ""))


# Thumbs reactions we treat as quality signal. Skin-tone modifiers and variation
# selectors trail the base codepoint, so a substring test catches every variant.
_THUMBS_UP = "\U0001F44D"  # 👍
_THUMBS_DOWN = "\U0001F44E"  # 👎


def reaction_rating(emoji: str) -> Optional[str]:
    """Map a reaction emoji to ``up``/``down``; ``None`` for anything else.

    An empty string means the reaction was *removed* (also ``None``).
    """
    if not emoji:
        return None
    if _THUMBS_UP in emoji:
        return "up"
    if _THUMBS_DOWN in emoji:
        return "down"
    return None


def parse_waha_reaction(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a WAHA ``message.reaction`` event into a normalized dict.

    Returns ``None`` for non-reaction events. Like ``parse_waha_message`` this is
    defensive because the payload shape varies across WAHA engines: the reaction body
    may sit at ``payload.reaction`` (``{text, messageId}``) or be flattened onto the
    payload, and the target id may be ``messageId``/``msgId``/``id``.
    """
    if event.get("event") not in ("message.reaction", "reaction"):
        return None
    payload = event.get("payload") or {}

    chat_id = str(payload.get("from") or "")
    reactor = _normalize_jid(payload.get("participant") or payload.get("author") or chat_id)

    reaction = payload.get("reaction")
    if not isinstance(reaction, dict):
        reaction = payload
    emoji = str(reaction.get("text") or reaction.get("emoji") or "")
    target_msg_id = str(
        reaction.get("messageId")
        or reaction.get("msgId")
        or reaction.get("id")
        or payload.get("messageId")
        or ""
    )
    if not target_msg_id:
        return None
    return {
        "chat_id": chat_id,
        "reactor": reactor,
        "target_msg_id": target_msg_id,
        "emoji": emoji,
        "from_me": bool(payload.get("fromMe", False)),
    }


class WhatsAppAdapter:
    """Decides whether/how to reply and wires history + engine + WAHA together."""

    def __init__(
        self,
        responder: Responder,
        waha_client: Any,
        config: Dict[str, Any],
        session_store: Optional[KeyedSessionMemory] = None,
        prefs: Optional[ChatPreferences] = None,
        message_log: Optional[MessageLog] = None,
        tts_synthesize: Optional[Callable[[str], Optional[bytes]]] = None,
        transcribe: Optional[Callable[[str, str], Optional[str]]] = None,
        image_generate: Optional[Callable[..., Optional[bytes]]] = None,
        fetch_media: Optional[Callable[[str, str], Optional[str]]] = None,
        describe_image: Optional[Callable[[str, str], Optional[str]]] = None,
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
        # Canonical member name lookup, keyed by name AND alias, used to learn the
        # phone->member mapping from real traffic (see resolve_speaker). Without it
        # every unmapped sender is labelled by their WhatsApp pushName, which may
        # not match the name RAG person-filtering expects.
        self.member_aliases = {k.lower(): v for k, v in (wcfg.get("member_aliases", {}) or {}).items()}
        self.learn_contacts = bool(wcfg.get("learn_contacts", True))
        self._contacts_path = wcfg.get("contacts_file")
        # DM anti-spam whitelist. When enabled, direct messages are only answered
        # for sender numbers in ``allowed`` (groups stay governed by @mention). The
        # numbers are loaded from the gitignored data/whatsapp_whitelist.json and
        # merged into ``whatsapp.whitelist.allowed`` by whatsapp_server at startup.
        whitelist_cfg = wcfg.get("whitelist", {}) or {}
        self.whitelist_enabled = bool(whitelist_cfg.get("enabled", False))
        self.whitelist_dm_only = bool(whitelist_cfg.get("dm_only", True))
        self.whitelist_numbers = {
            _phone_from_alt(n) for n in (whitelist_cfg.get("allowed", []) or []) if n
        }
        self.clear_commands = {"/clear", "/limpar"}
        # Making a picture takes minutes, so it never happens on the webhook
        # thread: the bot acknowledges, works in the background and sends the
        # result when it exists. None disables the feature entirely.
        self.image_generate = image_generate
        self.fetch_media = fetch_media
        self.describe_image = describe_image
        self.config = config
        # An edit needs a photo. Usually it is attached to the request, but "põe-lhe
        # uma coroa" right after someone posts a picture is just as natural, so the
        # last photo seen in each chat is remembered as the implicit subject.
        self._last_photo: Dict[str, Dict[str, str]] = {}
        # Chat ids whose content is group-wide memory (the Kaya group). Everything
        # else is private to its own chat — see src/chat/scope.py.
        self.shared_chats = set(wcfg.get("shared_chats", []) or [])
        # Durable log of every message SEEN (not just replied to) — the group's
        # ordinary chatter is the most valuable thing to remember, and the bot
        # only replies when addressed. Consumed by src/data/ingest.py.
        self.message_log = message_log or MessageLog(
            wcfg.get("message_log_dir", "data/live_messages")
        )
        self.log_messages = bool(wcfg.get("log_messages", True))
        self.history_turns = int(wcfg.get("history_turns", 10))
        self.send_seen = bool(wcfg.get("send_seen", True))
        # Messages older than this (unix seconds) are ignored — set on startup so a
        # reconnecting WAHA replaying backlog doesn't make the bot answer stale msgs.
        self.ignore_before_ts = 0
        # Message ids already answered, so a replay is not answered twice.
        self._answered_ids: "OrderedDict[str, bool]" = OrderedDict()
        self._answered_max = 2000
        self.session_store = session_store or KeyedSessionMemory(
            base_dir=wcfg.get("sessions_dir", "data/whatsapp_sessions"),
            max_lines=max(2 * self.history_turns, 20),
        )
        # Sticky per-chat settings (currently the reply modality). "Responde só em
        # áudio" holds until changed, so it is state on disk, not a per-message flag.
        self.prefs = prefs or ChatPreferences(
            base_dir=wcfg.get("prefs_dir", "data/whatsapp_prefs")
        )
        # Confirmations for router-detected commands, kept here so the wording is
        # not generated (and so it cannot drift or refuse).
        self.command_replies = {
            "audio": "Feito — passo a responder-te por áudio.",
            "text": "Feito — volto a responder por texto.",
            "clear": "Contexto limpo — esqueci as mensagens recentes desta conversa.",
            # Until TTS exists, saying yes and then answering in text forever is
            # worse than saying no. See chat.audio.reply_enabled.
            "audio_unavailable": "Ainda não sei responder por áudio — o Gustavo está a tratar disso. Por agora continuo a escrever.",
            # An edit takes minutes, so the acknowledgement has to set the
            # expectation — silence for five minutes reads as a broken bot.
            "image_editing": "Dá-me uns minutos que isto demora — já mando.",
            "image_generating": "Vou fazer isso, dá-me um bocado.",
            "image_queued": "Fica em fila, tenho {position} pela frente — mando assim que estiver.",
            "image_queue_full": "Tenho imagens a mais em fila. Pede daqui a uns minutos.",
            "image_failed": "Não consegui fazer a imagem. Tenta outra vez ou muda o pedido.",
            "image_not_allowed": "Só faço imagens no grupo, não por aqui.",
        }
        # Capability gate: the routed command is still recognised, but without TTS
        # the preference is NOT stored, because it would silently do nothing.
        # Injected so the adapter stays free of model/TTS imports (and tests can
        # exercise voice delivery without Piper installed).
        self.tts_synthesize = tts_synthesize
        self.transcribe = transcribe
        self.audio_reply_enabled = bool(
            ((config.get("chat", {}) or {}).get("audio", {}) or {}).get("reply_enabled", False)
        )
        # Whether to attribute 👍/👎 emoji reactions on the bot's own replies as
        # feedback. The actual logging happens in the server; the adapter only tracks
        # which sent message ids are the bot's and resolves a reaction back to them.
        self.feedback_enabled = bool((wcfg.get("feedback", {}) or {}).get("enabled", True))
        # bot_message_id -> {chat_id, user_text, reply, is_group, speaker}. Bounded so a
        # long-running process doesn't grow unbounded; oldest entries drop first.
        self.sent_messages: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.sent_messages_max = 500
        # chat_id -> recent message unix times, so we know where the verbatim
        # session window begins (used to stop retrieval duplicating it).
        self._session_times: Dict[str, List[int]] = {}

    # ── decision ────────────────────────────────────────────────────────────
    def should_respond(self, msg: InboundMessage) -> bool:
        if msg.from_me:
            return False
        if not msg.text.strip():
            return False
        if self.ignore_before_ts and msg.timestamp and msg.timestamp < self.ignore_before_ts:
            return False
        # WAHA replays its backlog after a reconnect — which the DNS drop that
        # killed the session in production would cause — and the same message id
        # then arrives twice. `ignore_before_ts` only covers a restart, so without
        # this the bot answers the same question again. Bounded, because the
        # process is long-lived.
        if msg.message_id:
            if msg.message_id in self._answered_ids:
                logger.info("ignoring replayed message %s", msg.message_id)
                return False
            self._answered_ids[msg.message_id] = True
            while len(self._answered_ids) > self._answered_max:
                self._answered_ids.popitem(last=False)
        if not msg.is_group:
            return self._dm_allowed(msg)  # DM: gated by whitelist when enabled
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

    def _deliver(self, chat_id: str, text: str, reply_to: Optional[str] = None,
                 force_voice: bool = False):
        """Send the reply as text, or as a voice note if this chat asked for it.

        Falls back to text whenever synthesis fails — a silent non-reply is much
        worse than the wrong medium.
        """
        wants_voice = force_voice or self.prefs.output_mode(chat_id) == ChatPreferences.OUTPUT_AUDIO
        if not wants_voice:
            return self.waha_client.send_text(chat_id, text, reply_to=reply_to)

        ogg = None
        if self.tts_synthesize is not None:
            ogg = self.tts_synthesize(text)
        if not ogg:
            print("⚠️  voice synthesis unavailable — replying with text instead")
            return self.waha_client.send_text(chat_id, text, reply_to=reply_to)

        try:
            return self.waha_client.send_voice(chat_id, ogg, reply_to=reply_to)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  sending the voice note failed ({exc}); replying with text")
            return self.waha_client.send_text(chat_id, text, reply_to=reply_to)

    def _note_message_time(self, chat_id: str, ts: Optional[int]) -> None:
        """Track message times so we know where this chat's verbatim window starts.

        Kept in memory only. After a restart the window start is unknown until the
        chat is active again, which costs nothing: the ingester runs periodically,
        so the newest messages are not in the vector store yet either.
        """
        if not ts:
            return
        window = self._session_times.setdefault(chat_id, [])
        window.append(int(ts))
        # The session store keeps `history_turns` turns; keep matching timestamps.
        del window[: max(0, len(window) - self.history_turns)]

    def _session_window_start(self, chat_id: str) -> Optional[str]:
        """ISO timestamp of the oldest message still held verbatim in this chat.

        Retrieval drops chunks at or after this, so the session store owns recent
        history and the vector DB owns everything older — the two can never inject
        the same text. None when this chat has not been active since startup.
        """
        window = self._session_times.get(chat_id)
        if not window:
            return None
        return datetime.datetime.fromtimestamp(
            window[0], tz=datetime.timezone.utc
        ).replace(tzinfo=None).isoformat()

    def _note_photo(self, msg: "InboundMessage") -> None:
        """Remember the last photo seen in a chat, as the implicit edit subject."""
        if msg.media_url and (msg.media_mimetype or "").startswith("image/"):
            self._last_photo[msg.chat_id] = {
                "url": msg.media_url, "mimetype": msg.media_mimetype,
                "sender": msg.sender_name or msg.sender_phone,
            }

    def _handle_image_request(self, msg: "InboundMessage", speaker: str,
                              text: str) -> Dict[str, Any]:
        """Acknowledge now, make the picture on a background thread, send it later.

        Generation takes minutes. Holding the webhook open for that would stall
        every other message in the group, so the only thing that happens inline is
        the acknowledgement.
        """
        import threading

        from src.chat import imagegen

        scope = scope_for_chat(msg.chat_id, self.shared_chats)
        # Declining is better than promising. A chat that may not ask, or a box
        # with the feature switched off, gets told so rather than left waiting for
        # a picture that is never coming.
        if self.image_generate is None or not imagegen.allowed_here(self.config, scope):
            reply = self.command_replies.get("image_not_allowed", "")
            if reply:
                self._deliver(msg.chat_id, reply)
            return {"chat_id": msg.chat_id, "speaker": speaker, "reply": reply,
                    "command": "image", "image": "not_allowed"}


        # An attached photo is unambiguous — that is the subject. Without one, the
        # last photo seen counts ONLY if the request actually points at something
        # ("põe-LHE uma coroa", "edita ESTA"). Treating every request as an edit
        # of the last photo meant "faz uma imagem de um gato astronauta" edited
        # somebody's holiday snap instead of drawing a cat, for as long as a photo
        # sat in the chat's history.
        source = None
        if msg.media_url and (msg.media_mimetype or "").startswith("image/"):
            source = {"url": msg.media_url, "mimetype": msg.media_mimetype}
        elif msg.chat_id in self._last_photo and refers_to_existing_image(text):
            source = self._last_photo[msg.chat_id]
        if source and self.fetch_media is None:
            source = None
        mode = "edit" if source else "generate"

        def work() -> None:
            path = None
            try:
                if source:
                    path = self.fetch_media(source["url"], source.get("mimetype", ""))
                    if not path:
                        self._deliver(msg.chat_id,
                                      self.command_replies.get("image_failed", ""))
                        return
                image = self.image_generate(mode=mode, prompt=text, image_path=path)
                if not image:
                    self._deliver(msg.chat_id,
                                  self.command_replies.get("image_failed", ""))
                    return
                self.waha_client.send_image(
                    msg.chat_id, image,
                    reply_to=msg.message_id if msg.is_group else None)
                # Close the loop in the history, so the bot stops saying a
                # picture is still coming once it has arrived.
                self.session_store.append(msg.chat_id, "Kaya Bot: (imagem enviada)")
            except Exception as exc:  # noqa: BLE001 — a worker crash must not kill the thread pool
                logger.warning("image job failed: %s", exc)
                self._deliver(msg.chat_id, self.command_replies.get("image_failed", ""))
            finally:
                if path:
                    Path(path).unlink(missing_ok=True)

        # Queued rather than refused. Telling someone "estou ocupado, tenta
        # depois" loses the request — a long simulator run had two edits asked
        # for, refused, and never made. The queue is separate from the text path:
        # generation runs on the other GPU, so the bot keeps answering questions
        # at full speed while a picture renders.
        position = imagegen.get_queue().submit(work)
        if position is None:
            reply = self.command_replies.get("image_queue_full", "")
            if reply:
                self._deliver(msg.chat_id, reply)
            return {"chat_id": msg.chat_id, "speaker": speaker, "reply": reply,
                    "command": "image", "image": "queue_full"}

        if position > 1:
            reply = self.command_replies.get("image_queued", "").format(position=position)
        else:
            reply = self.command_replies.get(
                "image_editing" if mode == "edit" else "image_generating", "")
        if reply:
            self._deliver(msg.chat_id, reply)

        # So the conversation knows a picture is on the way. Without this the bot
        # is asked "então e a foto?" two turns later and has no idea what it
        # agreed to make; the pending request has to be in the history it reads.
        self.session_store.append(
            msg.chat_id,
            f"Kaya Bot: (a preparar uma imagem, ainda não enviada — pedido: {text[:120]})")

        return {"chat_id": msg.chat_id, "speaker": speaker, "reply": reply,
                "command": "image", "image": mode, "queue_position": position}

    def _apply_command(self, chat_id: str, command: str) -> str:
        """Execute a routed command and return the confirmation to send back."""
        if command == ChatPreferences.OUTPUT_AUDIO and not self.audio_reply_enabled:
            # Don't confirm a mode we cannot honour, and don't persist it either —
            # a stored preference that changes nothing is how "it said yes and
            # then kept typing" happens.
            return self.command_replies["audio_unavailable"]
        if command in (ChatPreferences.OUTPUT_AUDIO, ChatPreferences.OUTPUT_TEXT):
            self.prefs.set_output_mode(chat_id, command)
        elif command == "clear":
            self.session_store._store(chat_id).clear()
        else:
            return ""
        return self.command_replies.get(command, "")

    def output_mode(self, chat_id: str) -> str:
        """The sticky reply modality for this chat (``"text"`` or ``"audio"``)."""
        return self.prefs.output_mode(chat_id)

    def _dm_allowed(self, msg: InboundMessage) -> bool:
        """Anti-spam gate for direct messages.

        When the whitelist is disabled the bot answers every DM (original
        behaviour). When enabled, only DMs from whitelisted numbers are answered;
        everyone else is silently ignored so a leaked number can't be spammed.
        """
        if not self.whitelist_enabled:
            return True
        candidates = {
            _phone_from_alt(msg.sender_phone),
            _phone_from_alt(msg.sender_id),
            msg.sender_id.split("@", 1)[0].strip().lower(),
        }
        return bool(candidates & self.whitelist_numbers)

    # ── speaker identity ──────────────────────────────────────────────────────
    def resolve_speaker(self, msg: InboundMessage) -> str:
        """Map the sender to a known Kaya member name when possible.

        Falls back to the WhatsApp push name, then a generic label, so the model
        always gets a usable ``"<who>: <text>"`` and RAG person-filtering can fire.
        """
        id_local = msg.sender_id.split("@", 1)[0]
        candidates = [
            _normalize_jid(msg.sender_id),
            _normalize_jid(f"{msg.sender_phone}@c.us") if msg.sender_phone else "",
            msg.sender_phone,
            _normalize_jid(f"{id_local}@c.us"),
            id_local,
        ]
        for candidate in candidates:
            if candidate and candidate in self.contacts:
                return self.contacts[candidate]

        # Not mapped yet. If WhatsApp's pushName matches a known member name or
        # alias, adopt the CANONICAL member name (so RAG person-filtering matches)
        # and remember the number, so the mapping self-populates from real traffic
        # instead of needing a hand-maintained file.
        pushname = (msg.sender_name or "").strip()
        canonical = self.member_aliases.get(pushname.lower()) if pushname else None
        if not canonical and pushname:
            first = pushname.split()[0].lower()
            canonical = self.member_aliases.get(first)
        if canonical:
            if self.learn_contacts:
                self._learn_contact(candidates, canonical)
            return canonical

        return pushname or "Alguém"

    def _learn_contact(self, candidates, name: str) -> None:
        """Remember phone -> member name, persisting so it survives a restart."""
        key = next((c for c in candidates if c), None)
        if not key or key in self.contacts:
            return
        self.contacts[key] = name
        if not self._contacts_path:
            return
        try:
            path = Path(self._contacts_path)
            existing = {}
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8")) or {}
            if existing.get(key) == name:
                return
            existing[key] = name
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
            print(f"✓ Learned WhatsApp contact {key} -> {name}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"⚠️  Could not persist learned contact {key}: {exc}")

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
        if msg is None:
            return None

        # A voice note arrives with empty text. Transcribe it first so it becomes
        # an ordinary message — logged as memory, routed, and answered like any
        # other. Without this it is silently dropped by the empty-text gate.
        if not msg.text.strip() and msg.media_url and self.transcribe is not None \
                and not (msg.media_mimetype or "").startswith("image/"):
            transcript = self.transcribe(msg.media_url, msg.media_mimetype)
            if transcript:
                msg.text = transcript
                print(f"🎤 transcribed voice note ({len(transcript)} chars)")

        # A photo is the same problem as a voice note: the message means nothing
        # until its content is turned into text. Described here, it becomes
        # memory, is ingested, and is answerable a week later — and a caption is
        # kept, since "quem é este?" is the question and the description is the
        # evidence.
        if msg.media_url and (msg.media_mimetype or "").startswith("image/") \
                and self.describe_image is not None:
            description = self.describe_image(msg.media_url, msg.media_mimetype)
            if description:
                msg.image_description = description
                caption = msg.text.strip()
                msg.text = (f"{caption}\n[Imagem: {description}]" if caption
                            else f"[Imagem: {description}]")
                print(f"🖼️  described image ({len(description)} chars)")

        # Log every message the bot SEES, before deciding whether to reply. Group
        # conversation the bot was not addressed in is exactly the memory worth
        # keeping, and it would otherwise be dropped by the gate below.
        if self.log_messages and not msg.from_me and msg.text.strip():
            self.message_log.append(
                chat_id=msg.chat_id,
                message_id=msg.message_id,
                sender=self.resolve_speaker(msg),
                text=msg.text,
                timestamp=msg.timestamp,
                scope=scope_for_chat(msg.chat_id, self.shared_chats),
            )

        self._note_photo(msg)

        if not self.should_respond(msg):
            return None

        speaker = self.resolve_speaker(msg)
        text = self._strip_bot_mention(msg.text)
        if not text:
            return None

        # ``/clear`` (or ``/limpar``): wipe this chat's recent context so the bot
        # stops fixating on prior turns. Handled before generation; not stored.
        if text.strip().lower() in self.clear_commands:
            self.session_store._store(msg.chat_id).clear()
            reply = "Contexto limpo — esqueci as mensagens recentes desta conversa."
            self.waha_client.send_text(msg.chat_id, reply)
            return {"chat_id": msg.chat_id, "speaker": speaker, "reply": reply, "command": "clear"}

        if self.send_seen:
            self.waha_client.send_seen(msg.chat_id)
            self.waha_client.start_typing(msg.chat_id)

        recent = self.session_store.recent(msg.chat_id, self.history_turns)
        # What long-term memory may this chat see, and from when. `recent` already
        # holds the last turns verbatim, so retrieval is told to skip anything
        # covering the same window rather than inject it twice.
        self._note_message_time(msg.chat_id, msg.timestamp)
        scope = scope_for_chat(msg.chat_id, self.shared_chats)
        exclude_from = self._session_window_start(msg.chat_id)
        try:
            result = self.responder(
                text, speaker, recent, scope=scope, exclude_from=exclude_from
            )
        finally:
            if self.send_seen:
                self.waha_client.stop_typing(msg.chat_id)

        # The responder returns either a plain string (tests, simulators) or a
        # Reply carrying the routing decision (production). A routed *command* —
        # "responde só em áudio", "esquece o que falámos" — is executed here rather
        # than generated, so the confirmation cannot drift or be refused.
        reply = result if isinstance(result, str) else getattr(result, "text", "")
        route = None if isinstance(result, str) else getattr(result, "route", None)
        command = getattr(route, "command", None) if route else None

        # A one-off voice request changes only how THIS reply is delivered, so it
        # falls through to normal generation rather than being executed as a
        # state change.
        deliver_as_voice_once = command == "audio_once"
        if deliver_as_voice_once:
            command = None

        if command == "image":
            return self._handle_image_request(msg, speaker, text)

        if command:
            reply = self._apply_command(msg.chat_id, command)
            if reply:
                # Deliver via the (possibly just-changed) preference, so "passo a
                # responder-te por áudio" arrives as the first voice note.
                self._deliver(msg.chat_id, reply)
            return {
                "chat_id": msg.chat_id, "speaker": speaker,
                "reply": reply, "command": command,
            }

        if not reply or not reply.strip():
            return None

        # Persist both sides so the next turn in this chat has context.
        self.session_store.append(msg.chat_id, f"{speaker}: {text}")
        self.session_store.append(msg.chat_id, f"Kaya Bot: {reply}")

        # Quote the asker's message in groups so it's clear who the bot answers.
        reply_to = msg.message_id if msg.is_group else None
        sent = self._deliver(msg.chat_id, reply, reply_to=reply_to,
                             force_voice=deliver_as_voice_once)

        # Remember this sent message so a later 👍/👎 reaction on it can be attributed.
        if self.feedback_enabled:
            self._remember_sent(
                extract_sent_id(sent),
                {
                    "chat_id": msg.chat_id,
                    "user_text": text,
                    "reply": reply,
                    "is_group": msg.is_group,
                    "speaker": speaker,
                },
            )

        return {
            "chat_id": msg.chat_id,
            "speaker": speaker,
            "reply": reply,
            "user_text": text,
            "is_group": msg.is_group,
        }

    def _remember_sent(self, message_id: str, info: Dict[str, Any]) -> None:
        """Track a bot-sent message id (bounded LRU) for reaction attribution."""
        if not message_id:
            return
        self.sent_messages[message_id] = info
        self.sent_messages.move_to_end(message_id)
        while len(self.sent_messages) > self.sent_messages_max:
            self.sent_messages.popitem(last=False)

    def handle_reaction(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Resolve a 👍/👎 reaction on one of the bot's replies into a feedback dict.

        Returns ``None`` when feedback is disabled, the event isn't a thumbs reaction,
        the reaction was removed, the reactor is the bot itself, or the target isn't a
        tracked bot message. The caller (server) does the actual feedback logging.
        """
        if not self.feedback_enabled:
            return None
        parsed = parse_waha_reaction(event)
        if parsed is None:
            return None
        rating = reaction_rating(parsed["emoji"])
        if rating is None:
            return None
        if parsed["from_me"] or (parsed["reactor"] and parsed["reactor"] in self.bot_jids):
            return None
        info = self.sent_messages.get(parsed["target_msg_id"])
        if not info:
            return None
        return {
            "chat_id": info.get("chat_id") or parsed["chat_id"],
            "rating": rating,
            "user_text": info.get("user_text", ""),
            "reply": info.get("reply", ""),
            "is_group": bool(info.get("is_group")),
            "reactor": parsed["reactor"],
        }
