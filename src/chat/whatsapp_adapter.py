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
import inspect
import json
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.chat import feedback
from src.chat.memory import ChatPreferences, KeyedSessionMemory
from src.chat.response_utils import truncate_history_line
from src.chat.scope import scope_for_chat
from src.data.message_log import MessageLog
from src.chat.waha_client import extract_sent_id

logger = logging.getLogger(__name__)


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """Whether ``fn`` will accept ``name=`` — asked once, not guessed per call."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind == p.VAR_KEYWORD for p in params.values())

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
    # What this message is a reply to. Only the participant used to be read —
    # enough to decide whether to answer, and the quoted text was thrown away.
    # So "A culpa é do", sent as a reply, reached the model with nothing to
    # attach it to and came back as "A culpa é do quem? Completa lá a frase."
    reply_to_id: str = ""
    quoted_text: str = ""
    quoted_sender: str = ""
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


def _baileys_text(message: Any) -> str:
    """Pull the text out of a Baileys message object, whatever type it is.

    NOWEB keys the body by message type — ``conversation`` for a plain message,
    ``extendedTextMessage.text`` for one with context, ``caption`` for media —
    and a quoted message is just another message object, so the same walk works
    for both. A quoted photo contributes its caption, or nothing.
    """
    if isinstance(message, str):
        return message.strip()
    if not isinstance(message, dict):
        return ""
    direct = message.get("conversation") or message.get("text") or message.get("caption")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for value in message.values():
        if isinstance(value, dict):
            found = _baileys_text(value)
            if found:
                return found
    return ""


def _quoted(payload: Dict[str, Any], context: Dict[str, Any]) -> tuple:
    """``(reply_to_id, quoted_text, quoted_sender)`` for a reply, or blanks.

    Both engine shapes, because the bridge has to work with either: NOWEB hangs
    the original off ``contextInfo.quotedMessage`` with its id in ``stanzaId``,
    while WEBJS sends a flatter ``replyTo``/``quotedMsg`` carrying ``body``.
    WAHA does not always include the quoted body — a reply to something it never
    saw comes through with the participant only — so every field is optional and
    the caller must cope with an id and no text.
    """
    reply_to = payload.get("replyTo") or payload.get("quotedMsg") or {}
    if not isinstance(reply_to, dict):
        reply_to = {}

    text = ""
    for candidate in (reply_to.get("body"), reply_to.get("text"),
                      context.get("quotedMessage")):
        text = _baileys_text(candidate)
        if text:
            break

    quoted_id = str(
        context.get("stanzaId") or reply_to.get("id") or reply_to.get("_serialized") or ""
    )
    sender = _normalize_jid(
        reply_to.get("participant") or reply_to.get("author") or context.get("participant") or ""
    )
    return quoted_id, text, sender


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
    quoted_id, quoted_text, quoted_sender = _quoted(payload, context)

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
        reply_to_id=quoted_id,
        quoted_text=quoted_text,
        quoted_sender=quoted_sender,
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


# A WhatsApp mention in the message body. Baileys writes the @lid or the phone
# number, never the display name — what the group sees as "@Rafa" arrives here as
# "@257487651496102".
_MENTION = re.compile(r"@(\d{5,})\b")


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
        speech_text: Optional[Callable[[str], str]] = None,
        transcribe: Optional[Callable[[str, str], Optional[str]]] = None,
        image_generate: Optional[Callable[..., Optional[bytes]]] = None,
        fetch_media: Optional[Callable[[str, str], Optional[str]]] = None,
        describe_image: Optional[Callable[[str, str], Optional[str]]] = None,
        summary_writer: Any = None,
        sender_resolver: Any = None,
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
        # Display-name resolution, shared with the extraction pipeline so the
        # live bot and the corpus agree on who somebody is. Injected rather than
        # built here: it needs group_members.json, which this module otherwise
        # never touches. None keeps the older, weaker fallback in resolve_speaker.
        self.sender_resolver = sender_resolver
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
        # Words kept from the message a reply quotes. It is context above the
        # real message, not the message itself.
        self.quoted_max_words = int(wcfg.get("quoted_max_words", 30))
        self.clear_commands = {"/clear", "/limpar"}
        # Slash commands that carry an argument: "/bug o áudio não funcionou".
        # Kept literal rather than routed — the router mistaking an ordinary
        # message for a silent bug report is worse than typing seven characters.
        self.bug_commands = {"/bug", "/erro"}
        self.feedback_commands = {"/feedback", "/sugestao", "/sugestão"}
        # Every command token, used to keep these messages OUT of the memory log
        # (see _parse_command — the log is written before the reply gate).
        self.all_commands = (
            self.clear_commands | self.bug_commands | self.feedback_commands
        )
        self._command_families = {
            **{token: "clear" for token in self.clear_commands},
            **{token: "bug" for token in self.bug_commands},
            **{token: "feedback" for token in self.feedback_commands},
        }
        # Where a new report is announced. A JID (…@c.us); empty disables it.
        # Real numbers stay out of git, so this comes from KAYA_REPORT_JID.
        self.report_jid = _normalize_jid(str(wcfg.get("report_to") or ""))
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
        # How many INBOUND messages those lines represent. An answered turn writes
        # two lines (the asker's and the bot's), so it is half — and this is what
        # the retrieval-exclusion window must be measured in. See
        # _note_message_time for what goes wrong when the two disagree.
        self._inbound_window = max(1, self.history_turns // 2)
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
        # No dashes anywhere in here: the group asked for commas at most, and a
        # canned string is the one place a prompt rule can never reach.
        self.command_replies = {
            "audio": "Feito, passo a responder-te por áudio.",
            "text": "Feito, volto a responder por texto.",
            "clear": "Contexto limpo, esqueci as mensagens recentes desta conversa.",
            # Until TTS exists, saying yes and then answering in text forever is
            # worse than saying no. See chat.audio.reply_enabled.
            "audio_unavailable": "Ainda não sei responder por áudio, o Gustavo está a tratar disso. Por agora continuo a escrever.",
            # An edit takes minutes, so the acknowledgement has to set the
            # expectation — silence for five minutes reads as a broken bot.
            "image_editing": "Dá-me uns minutos que isto demora, já mando.",
            "image_generating": "Vou fazer isso, dá-me um bocado.",
            "image_queued": "Fica em fila, tenho {ahead} à frente, mando assim que estiver.",
            "image_queue_full": "Tenho imagens a mais em fila. Pede daqui a uns minutos.",
            # GPU0 is lent to a maintenance job. The request keeps its place; the
            # wait is stated rather than discovered.
            "image_maintenance": ("Estou em manutenção, a placa das imagens está "
                                  "ocupada. Fica em fila, mando daqui a ~{minutes} min."),
            "image_failed": "Não consegui fazer a imagem. Tenta outra vez ou muda o pedido.",
            # The editor handing the photo back unchanged used to be delivered as
            # a success, and the group had to work out for itself that nothing
            # had happened ("Was the generation rejected? The image looks exactly
            # the same"). Asked about it, the bot invented content filters. It
            # says this instead, and the same line goes into the history so the
            # follow-up turn answers from fact.
            "image_unchanged": ("Essa edição não pegou, a imagem saiu na mesma. "
                                "Tenta pedir de outra maneira ou com mais detalhe."),
            "image_not_allowed": "Só faço imagens no grupo, não por aqui.",
            # Reports are collected for a week before anything is acted on, so the
            # confirmation has to say the message landed somewhere a person reads.
            "bug_logged": "Registado. Obrigado, o Gustavo vai ver isto.",
            "bug_usage": ("Escreve /bug e a seguir o que aconteceu. "
                          "Exemplo: /bug não respondeu ao meu áudio."),
            "feedback_logged": "Registado. Obrigado pelo feedback.",
            "feedback_usage": ("Escreve /feedback e a seguir a tua sugestão. "
                               "Exemplo: /feedback devias ser mais curto nas respostas."),
        }
        # Capability gate: the routed command is still recognised, but without TTS
        # the preference is NOT stored, because it would silently do nothing.
        # Injected so the adapter stays free of model/TTS imports (and tests can
        # exercise voice delivery without Piper installed).
        # Keeps each chat's rolling summary of what has scrolled out of the
        # verbatim window. Injected like tts_synthesize so the adapter needs no
        # model import and tests can run without one.
        self.summary_writer = summary_writer
        self._responder_takes_summary = _accepts_kwarg(responder, "summary")
        self.tts_synthesize = tts_synthesize
        # What a reply sounds like is not what it looks like: emoji, markdown and
        # URLs are read out loud by Piper. Injected like tts_synthesize so the
        # adapter keeps no tts import.
        self.speech_text = speech_text
        self.transcribe = transcribe
        audio_cfg = ((config.get("chat", {}) or {}).get("audio", {}) or {})
        self.audio_reply_enabled = bool(audio_cfg.get("reply_enabled", False))
        self.send_citation_as_text = bool(audio_cfg.get("send_citation_as_text", True))
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
                 force_voice: bool = False, citation: str = ""):
        """Send the reply as text, or as a voice note if this chat asked for it.

        Falls back to text whenever synthesis fails — a silent non-reply is much
        worse than the wrong medium.

        ``citation`` is the sources line of a web-grounded answer. In text it is
        appended exactly as before. Spoken, it is NOT: Piper reading "🌐 Fontes:
        x.com, play.google.com" at the end of a voice note is what the group heard
        and did not want. It goes out as a short written follow-up instead.

        Returns ``(sent, delivered_as, spoken_text)`` so the caller can record what
        was actually said, which the metrics log had no way of knowing before.
        """
        wants_voice = force_voice or self.prefs.output_mode(chat_id) == ChatPreferences.OUTPUT_AUDIO
        written = f"{text}\n\n{citation}" if citation else text
        if not wants_voice:
            return self.waha_client.send_text(chat_id, written, reply_to=reply_to), "text", ""

        spoken = self.speech_text(text) if self.speech_text else text
        ogg = None
        if self.tts_synthesize is not None and spoken.strip():
            ogg = self.tts_synthesize(spoken)
        if not ogg:
            print("⚠️  voice synthesis unavailable — replying with text instead")
            return self.waha_client.send_text(chat_id, written, reply_to=reply_to), "text", ""

        try:
            sent = self.waha_client.send_voice(chat_id, ogg, reply_to=reply_to, text=spoken)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  sending the voice note failed ({exc}); replying with text")
            return self.waha_client.send_text(chat_id, written, reply_to=reply_to), "text", ""

        if citation and self.send_citation_as_text:
            try:
                self.waha_client.send_text(chat_id, citation)
            except Exception as exc:  # noqa: BLE001 — the answer already went out
                print(f"⚠️  sending the sources line failed ({exc})")
        return sent, "voice", spoken

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
        # Keep one timestamp per INBOUND message actually inside the window, not
        # one per line. Each answered turn appends two lines to the session store
        # — the asker's and the bot's — so `history_turns` lines is only half that
        # many inbound messages.
        #
        # Getting this wrong opens a hole rather than a duplicate: the window
        # start would sit further back than what the prompt actually carries, so
        # retrieval would drop chunks covering messages that are NOT held
        # verbatim, and nothing would have them. At history_turns=6 the hole was
        # three messages wide and easy to miss; at 60 it would be thirty.
        del window[: max(0, len(window) - self._inbound_window)]

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
        if self.image_generate is None or not imagegen.allowed_here(
                self.config, scope, chat_id=msg.chat_id, is_group=msg.is_group):
            reply = self.command_replies.get("image_not_allowed", "")
            if reply:
                self._deliver(msg.chat_id, reply)
            return {"chat_id": msg.chat_id, "speaker": speaker, "reply": reply,
                    "user_text": text, "command": "image", "image": "not_allowed"}


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
                    # An edit that produced nothing is reported as an edit that
                    # produced nothing. The bot must not be left guessing why
                    # two turns later — that is how "it got blocked by the
                    # content filters" was invented in the group.
                    reply = self.command_replies.get(
                        "image_unchanged" if mode == "edit" else "image_failed", "")
                    self._deliver(msg.chat_id, reply)
                    self.session_store.append(
                        msg.chat_id, f"Kaya Bot: (a imagem não saiu, pedido: {text[:80]})")
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
        queue = imagegen.get_queue()
        queue.configure(self.config)

        # The acknowledgement goes out BEFORE the job is queued. Submitting
        # first means a job that fails instantly — the feature switched off, an
        # edit with no source photo, both of which return without touching the
        # GPU — can deliver "não consegui" ahead of "vou fazer isso", which
        # reads as the bot answering itself backwards.
        position = queue.depth + 1
        if position > queue.maxsize:
            reply = self.command_replies.get("image_queue_full", "")
            if reply:
                self._deliver(msg.chat_id, reply)
            return {"chat_id": msg.chat_id, "speaker": speaker, "reply": reply,
                    "user_text": text, "command": "image", "image": "queue_full"}

        waiting = imagegen.pause_remaining(self.config)
        if waiting:
            # GPU0 is on loan to a maintenance job. Say so with the wait, rather
            # than acknowledging normally and delivering forty minutes later —
            # the bounded queue exists precisely because a picture nobody still
            # wants is worse than an honest no. If the job releases early the
            # queue drains at once and this was merely pessimistic.
            reply = self.command_replies.get("image_maintenance", "").format(
                minutes=waiting)
        elif position > 1:
            # position is 1-based, so the number of jobs AHEAD is one less —
            # saying "2 pela frente" at position 2 reads as one picture too many.
            reply = self.command_replies.get("image_queued", "").format(
                ahead=position - 1, position=position)
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

        # Only now, with the promise already sent, does the work start. A
        # concurrent request can still have taken the last slot in the gap.
        submitted = queue.submit(work)
        if submitted is None:
            self._deliver(msg.chat_id, self.command_replies.get("image_queue_full", ""))
            return {"chat_id": msg.chat_id, "speaker": speaker, "reply": reply,
                    "user_text": text, "command": "image", "image": "queue_full"}

        return {"chat_id": msg.chat_id, "speaker": speaker, "reply": reply,
                "user_text": text, "command": "image", "image": mode,
                "queue_position": submitted}

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
                name = self.contacts[candidate]
                # Learn the OTHER shapes of the same person while we are here. A
                # member whose phone number is already mapped never reached the
                # learning branch below, so their @lid stayed unknown — and @lid
                # is the only form a mention in the message body comes in. Four
                # of the group were in this state: perfectly identified as
                # speakers, invisible when someone @-ed them.
                if self.learn_contacts:
                    self._learn_contact(candidates, name, verified=True)
                return name

        # Not mapped yet, so fall back to the display name. That is resolved by
        # the SAME resolver the extraction pipeline uses, rather than the
        # whole-name-then-first-token attempt this used to make on its own: "Tomas
        # Carnall" failed because the first token is "tomas" and the alias is
        # "tomás", and the second token — "carnall", an exact alias — was never
        # tried. Five of a real member's messages were logged under a name that is
        # not a member, invisible to RAG person-filtering.
        #
        # It also declines to guess. "Ricardo" matches both Ricardo Romano and
        # Ricardo Alberto, so it resolves to nobody rather than to whichever came
        # first; ``data.sender_aliases`` settles the cases that need settling.
        pushname = (msg.sender_name or "").strip()
        canonical = ""
        if pushname and self.sender_resolver is not None:
            resolved = self.sender_resolver.resolve(pushname)
            if self.sender_resolver.is_member(resolved):
                canonical = resolved
        elif pushname:
            # No resolver injected (tests, older callers): the previous behaviour.
            canonical = self.member_aliases.get(pushname.lower()) or self.member_aliases.get(
                pushname.split()[0].lower(), "")

        if canonical:
            # Remember the number, so the mapping self-populates from real
            # traffic instead of needing a hand-maintained file.
            if self.learn_contacts:
                self._learn_contact(candidates, canonical)
            return canonical

        return pushname or "Alguém"

    def _name_for_jid(self, jid: str) -> str:
        """Best known name for a bare JID. "" when it maps to nobody."""
        normalized = _normalize_jid(jid)
        if not normalized:
            return ""
        local = normalized.split("@", 1)[0]
        # ``@lid`` is in the list because that is how Baileys addresses people and
        # therefore how most of data/whatsapp_contacts.json is keyed. A mention in
        # the message body arrives as the bare number, with no suffix at all, so
        # without this every @lid contact was a miss.
        candidates = (normalized, f"{local}@c.us", f"{local}@lid", local)
        # The bot is matched against the same shapes for the same reason: bot_jids
        # holds "<lid>@lid", a body mention is the bare number, and comparing only
        # the raw form left the bot's own @ unresolved in the message log — which
        # is the log that becomes searchable memory.
        if self.bot_jids.intersection(candidates):
            return "Kaya Bot"
        for candidate in candidates:
            if candidate in self.contacts:
                return self.contacts[candidate]
        return ""

    def quoted_context(self, msg: InboundMessage) -> str:
        """The message this one replies to, as a line to put above it.

        Without it a reply is a fragment. "A culpa é do" was answered "A culpa é
        do quem? Completa lá a frase, papi" because the quoted message was
        parsed for its sender and then discarded. Roughly a third of the group's
        messages are four words or fewer, and most of those are replies.

        Truncated, because the parent is context and not the message being
        answered. Empty when WAHA sent no quoted body, which it does not always.
        """
        if not msg.quoted_text.strip():
            return ""
        who = self._name_for_jid(msg.quoted_sender)
        body = truncate_history_line(msg.quoted_text.strip(), self.quoted_max_words)
        label = f"a responder a {who}" if who else "a responder a"
        return f"[{label}: \"{body}\"]"

    def _learn_contact(self, candidates, name: str, verified: bool = False) -> None:
        """Remember phone -> member name, persisting so it survives a restart.

        A learned mapping is usually a GUESS: it comes from the sender's display
        name, which the person chooses and which two people can share. One member
        is known to the group by a nickname while his display name is another
        member's actual name, so this quietly learned him as that other member
        and every message he sent was attributed to the wrong man for weeks.

        It still learns — that is how most of the file got populated — but a
        newly learned name that ALREADY belongs to a different id is the smell,
        and it is now said out loud. Two ids for one member is normal (a phone
        and a @lid); two *people* behind one member is the failure.

        ``verified`` says the name came from an id already in the file rather
        than from a display name, which is how the other shapes of an
        already-known person get filled in. That case is precisely "two ids for
        one member", so it must not raise the alarm meant for the other one.
        """
        key = next((c for c in candidates if c), None)
        if not key or key in self.contacts:
            return
        existing_ids = [i for i, n in self.contacts.items() if n == name and i != key]
        if existing_ids and not verified:
            print(f"⚠️  learned {key} -> {name} from a display name, but {name} is "
                  f"already {', '.join(existing_ids)}. If this is a different "
                  f"person they need their own entry — check with "
                  f"scripts/map_senders.py.")
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

    def _resolve_mentions(self, text: str) -> str:
        """Turn ``@<lid>`` tokens into the member's name.

        Only the bot's own mention was ever removed; everybody else's arrived at
        the model as a bare number. So "@257487651496102 tas fraquinho. Que
        desilusão" said nothing about Rafa to the model and nothing to
        ``extract_query_persons``, which matches member NAMES — and the roast,
        handed the usual pile of profiles with no one named in the question, went
        to Manuel, who was not in the conversation at all. Both filed reports of
        the bot "referencing the wrong people" are this.

        A lid that maps to nobody is left as it is. An unknown mention is still a
        mention, and deleting it would turn "@X e o @Y" into a sentence about one
        person.
        """
        def replace(match: "re.Match") -> str:
            name = self._name_for_jid(match.group(1))
            return f"@{name}" if name else match.group(0)

        return _MENTION.sub(replace, text)

    def _parse_command(self, text: str) -> Optional[Tuple[str, str]]:
        """Find a slash command anywhere in the message. ``(family, body)`` or None.

        Used for two things: keeping commands out of the memory log, which is
        written *before* the reply gate — so without it a ``/bug`` becomes a
        searchable "message the group sent" and comes back out of retrieval a week
        later — and deciding what to record.

        It scans every token, not just the first, because people do not write the
        way a parser expects. The single most useful note of the first group
        session was typed as "Andas a dizer demasiado foda-se no fim das frases.
        /feedback o problema é a construção frásica ser repetitiva": the command
        arrived mid-message, so it was never recorded at all. The model answered
        it instead, promising to do better, and the whole line went into group
        memory.

        The body is everything after the command word — the run-up is what
        prompted the report, not part of it.
        """
        for index, token in enumerate((text or "").split()):
            family = self._command_families.get(token.strip().lower())
            if family:
                body = (text or "").split(maxsplit=index + 1)
                return family, (body[index + 1].strip() if len(body) > index + 1 else "")
        return None

    def _notify(self, chat_id: str, text: str) -> None:
        """Send a side-channel message. A failed notification never breaks a report."""
        if not chat_id or not text:
            return
        try:
            self.waha_client.send_text(chat_id, text)
        except Exception as exc:  # noqa: BLE001 — the report is already on disk
            print(f"⚠️  could not deliver notification to {chat_id}: {exc}")

    def _reporter_dm(self, msg: InboundMessage) -> str:
        """The reporter's own chat, so a group report can be acknowledged privately."""
        phone = _phone_from_alt(msg.sender_phone) or _phone_from_alt(msg.sender_id)
        if phone:
            return f"{phone}@c.us"
        return msg.sender_id or ""

    def _handle_report(self, msg: InboundMessage, body: str, speaker: str,
                       command: str) -> Dict[str, Any]:
        """Record a ``/bug`` or ``/feedback`` and confirm it, without generating.

        ``body`` is everything after the command word, already split off by
        ``_parse_command``. A bare command explains itself and stores nothing — no
        pending-capture state, so an unrelated next message can never be swallowed
        into somebody's bug report.
        """
        is_bug = command == "bug"
        body = (body or "").strip()

        if not body:
            reply = self.command_replies["bug_usage" if is_bug else "feedback_usage"]
            self.waha_client.send_text(msg.chat_id, reply)
            return {"chat_id": msg.chat_id, "speaker": speaker, "reply": reply,
                    "command": command, "logged": False}

        recent = self.session_store.recent(msg.chat_id, 6) if is_bug else []
        if is_bug:
            feedback.log_bug_report(
                source="whatsapp", description=body, contact=speaker,
                env=os.environ.get("KAYA_ENV"), version=os.environ.get("KAYA_VERSION"),
                recent_turns=list(recent),
                path=feedback.bug_log_path(self.config),
            )
        else:
            feedback.log_note(
                source="whatsapp", text=body, contact=speaker,
                env=os.environ.get("KAYA_ENV"), version=os.environ.get("KAYA_VERSION"),
                path=feedback.feedback_log_path(self.config),
            )

        reply = self.command_replies["bug_logged" if is_bug else "feedback_logged"]
        self.waha_client.send_text(msg.chat_id, reply)

        label = "Bug" if is_bug else "Feedback"
        where = "no grupo" if msg.is_group else "por DM"
        self._notify(self.report_jid, f"{label} de {speaker} ({where}):\n{body}")
        # Reported from the group, the confirmation above is public and scrolls
        # away; a DM is the copy the reporter keeps. From a DM it would be the
        # same message twice, so it is not sent.
        if msg.is_group:
            self._notify(self._reporter_dm(msg), f"{label} registado: {body}")

        return {"chat_id": msg.chat_id, "speaker": speaker, "reply": reply,
                "command": command, "logged": True}

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

        # A slash command is an instruction to the bot, not something the group
        # said, and this log is what gets embedded into long-term memory. It is
        # written BEFORE the reply gate below, so commands have to be excluded
        # here or they come back out of retrieval later as things people "said" —
        # which is exactly what a week of collected bug reports must not become.
        # Mention-stripped first: in a group the text arrives as "@Kaya /bug ...".
        _command = self._parse_command(self._strip_bot_mention(msg.text))

        # Log every message the bot SEES, before deciding whether to reply. Group
        # conversation the bot was not addressed in is exactly the memory worth
        # keeping, and it would otherwise be dropped by the gate below.
        if self.log_messages and not msg.from_me and msg.text.strip() and not _command:
            self.message_log.append(
                chat_id=msg.chat_id,
                message_id=msg.message_id,
                sender=self.resolve_speaker(msg),
                # Named, not numbered. This log is embedded into ChromaDB, and a
                # message stored as "@257487651496102 tas fraquinho" can never be
                # retrieved by a question about Rafa.
                text=self._resolve_mentions(msg.text),
                timestamp=msg.timestamp,
                scope=scope_for_chat(msg.chat_id, self.shared_chats),
                reply_to_id=msg.reply_to_id,
                reply_to_text=truncate_history_line(
                    msg.quoted_text.strip(), self.quoted_max_words),
                # Who actually sent it, not just what they are currently called.
                sender_id=msg.sender_id,
                sender_phone=msg.sender_phone,
            )

        self._note_photo(msg)

        if not self.should_respond(msg):
            return None

        speaker = self.resolve_speaker(msg)
        # Strip the bot's own mention, then name everybody else's: the resolved
        # text is what reaches the model, the retriever's person filter and the
        # session store, and a bare @lid is invisible to all three.
        text = self._resolve_mentions(self._strip_bot_mention(msg.text))
        if not text:
            return None

        # ``/clear`` (or ``/limpar``): wipe this chat's recent context so the bot
        # stops fixating on prior turns. Handled before generation; not stored.
        # ``/bug`` and ``/feedback``: recorded, confirmed, never generated on. The
        # collection channel for the listening week — see src/chat/feedback.py.
        parsed = self._parse_command(text)
        if parsed and parsed[0] in ("bug", "feedback"):
            return self._handle_report(msg, parsed[1], speaker, parsed[0])

        # ``/clear`` stays strict — it must be the whole message. The two mistakes
        # are not symmetric: a missed report can be retyped, while a wrongly
        # triggered wipe silently destroys the context of a live conversation.
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
        kwargs = {"scope": scope, "exclude_from": exclude_from}
        # Older responders (test stubs, the simulators) take no `summary`. Ask the
        # signature rather than catching TypeError, which would swallow a real one
        # raised inside the responder itself.
        if self.summary_writer is not None and self._responder_takes_summary:
            try:
                kwargs["summary"] = self.summary_writer.store.summary_for(msg.chat_id)
            except Exception as exc:  # noqa: BLE001 — a missing summary is not fatal
                logger.warning("could not read the summary for this chat: %s", exc)
        # A reply carries the message it answers, or it is a fragment. This is
        # added AFTER the command checks, so "/bug" quoting something still reads
        # as the command and not as a message about it.
        quoted = self.quoted_context(msg)
        asked = f"{quoted}\n{text}" if quoted else text
        try:
            result = self.responder(asked, speaker, recent, **kwargs)
        finally:
            if self.send_seen:
                self.waha_client.stop_typing(msg.chat_id)

        # The responder returns either a plain string (tests, simulators) or a
        # Reply carrying the routing decision (production). A routed *command* —
        # "responde só em áudio", "esquece o que falámos" — is executed here rather
        # than generated, so the confirmation cannot drift or be refused.
        reply = result if isinstance(result, str) else getattr(result, "text", "")
        route = None if isinstance(result, str) else getattr(result, "route", None)
        citation = "" if isinstance(result, str) else getattr(result, "citation", "")
        telemetry = {} if isinstance(result, str) else getattr(result, "telemetry", {})
        command = getattr(route, "command", None) if route else None

        # A one-off voice request changes only how THIS reply is delivered, so it
        # falls through to normal generation rather than being executed as a
        # state change.
        deliver_as_voice_once = command == "audio_once"
        if deliver_as_voice_once:
            command = None
        # A count was answered by generating, not by changing any state — the
        # engine counted the log and the model phrased it. It reaches here as a
        # command only so the routing decision stays visible in the telemetry.
        if command == "count":
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

        # Persist both sides so the next turn in this chat has context. The
        # quoted line goes in too: the parent is usually NOT in this history
        # (the bot only records turns it answered), so dropping it here would
        # make the follow-up turn as contextless as this one was.
        self.session_store.append(
            msg.chat_id, f"{speaker}: {quoted} {text}" if quoted else f"{speaker}: {text}")
        self.session_store.append(msg.chat_id, f"Kaya Bot: {reply}")

        # Quote the asker's message in groups so it's clear who the bot answers.
        reply_to = msg.message_id if msg.is_group else None
        sent, delivered_as, spoken = self._deliver(
            msg.chat_id, reply, reply_to=reply_to,
            force_voice=deliver_as_voice_once, citation=citation,
        )

        # Keep this chat's rolling summary current. Queued AFTER delivery and run
        # on a worker that skips when the GPU is busy, because a summary must
        # never cost somebody their next message.
        if self.summary_writer is not None:
            try:
                self.summary_writer.maybe_update(
                    msg.chat_id, self.session_store.recent(msg.chat_id, None))
            except Exception as exc:  # noqa: BLE001 — never break a delivered reply
                logger.warning("summary trigger failed: %s", exc)

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
            "citation": citation,
            "user_text": text,
            "is_group": msg.is_group,
            # What medium it actually went out on, and — for a voice note — the
            # exact string Piper was given. Without these the interaction log
            # cannot tell a spoken reply from a written one, let alone show that
            # the sources line was being read aloud.
            "delivered_as": delivered_as,
            "spoken_text": spoken,
            # How the turn was classified and what it retrieved. Merged verbatim
            # into the interaction log so a bad answer can be attributed to the
            # route rather than guessed at from the reply.
            "telemetry": telemetry,
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
