"""Unit tests for the WhatsApp bridge — routing, gating, speaker, history.

No GPU/model/network: the engine is replaced by a stub ``responder`` and WAHA by
``MockWahaClient``, so this exercises the full inbound→reply logic the same way
``scripts/whatsapp_simulator.py`` does, against a temp session dir.
"""
import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.memory import KeyedSessionMemory
from src.chat.scope import scope_for_chat
from src.chat.waha_client import MockWahaClient
from src.chat.whatsapp_adapter import WhatsAppAdapter, parse_waha_message

BOT_JID = "351900000000@c.us"
GROUP = "12036300000000@g.us"
ALICE = "351911111111@c.us"


def make_adapter(tmp_path, **overrides):
    config = {
        "whatsapp": {
            "bot_jid": BOT_JID,
            "group": {"respond_on_mention": True, "respond_on_reply": True},
            "contacts": {"351911111111@c.us": "Alice"},
            "send_seen": False,
            "history_turns": 5,
            **overrides,
        }
    }
    store = KeyedSessionMemory(base_dir=str(tmp_path / "sessions"), max_lines=10)
    client = MockWahaClient(echo=False)

    def responder(message, speaker, recent_lines, scope=None, exclude_from=None):
        return f"reply[{speaker}|{len(recent_lines)}]:{message}"

    adapter = WhatsAppAdapter(responder, client, config, session_store=store)
    return adapter, client


# Real WhatsApp ids are unique per message, and the adapter now relies on that to
# ignore WAHA's post-reconnect replays. A fixture that reused one id made every
# second message in a test look like a replay.
_event_seq = itertools.count(1)


def dm_event(text, sender=ALICE, name="Alice", from_me=False, message_id=None):
    return {
        "event": "message",
        "payload": {"id": message_id or f"dm{next(_event_seq)}", "from": sender,
                    "body": text, "notifyName": name, "fromMe": from_me},
    }


def group_event(text, sender=ALICE, name="Alice", mention=False, reply=False,
                message_id=None):
    payload = {
        "id": message_id or f"g{next(_event_seq)}",
        "from": GROUP,
        "participant": sender,
        "body": text,
        "notifyName": name,
        "mentionedIds": [BOT_JID] if mention else [],
    }
    if reply:
        payload["replyTo"] = {"participant": BOT_JID}
    return {"event": "message", "payload": payload}


# ── parsing ──────────────────────────────────────────────────────────────────
def test_parse_ignores_non_message():
    assert parse_waha_message({"event": "session.status", "payload": {}}) is None


def test_parse_dm_vs_group():
    dm = parse_waha_message(dm_event("hi"))
    assert dm.is_group is False and dm.sender_id == ALICE
    grp = parse_waha_message(group_event("hi", mention=True))
    assert grp.is_group is True
    assert grp.sender_id == ALICE  # participant, not the group id
    assert BOT_JID in grp.mentioned_ids


# ── DM routing: always answer ──────────────────────────────────────────────────
def test_dm_always_responds(tmp_path):
    adapter, client = make_adapter(tmp_path)
    result = adapter.handle_event(dm_event("olá"))
    assert result is not None
    assert len(client.sent) == 1
    assert client.sent[0]["chat_id"] == ALICE
    assert client.sent[0]["reply_to"] is None  # DMs are not quoted


def test_ignores_own_messages(tmp_path):
    adapter, client = make_adapter(tmp_path)
    assert adapter.handle_event(dm_event("echo", from_me=True)) is None
    assert client.sent == []


def test_ignores_empty_text(tmp_path):
    adapter, client = make_adapter(tmp_path)
    assert adapter.handle_event(dm_event("   ")) is None


# ── group routing: only when addressed ─────────────────────────────────────────
def test_group_silent_without_mention(tmp_path):
    adapter, client = make_adapter(tmp_path)
    assert adapter.handle_event(group_event("conversa random")) is None
    assert client.sent == []


def test_group_responds_on_mention(tmp_path):
    adapter, client = make_adapter(tmp_path)
    event = group_event("@bot quem é o Rui?", mention=True)
    result = adapter.handle_event(event)
    assert result is not None
    assert len(client.sent) == 1
    # group replies quote the asker — compare to the event's own id, which is
    # generated per message rather than fixed
    assert client.sent[0]["reply_to"] == event["payload"]["id"]


def test_group_responds_on_reply_to_bot(tmp_path):
    adapter, client = make_adapter(tmp_path)
    result = adapter.handle_event(group_event("e o Tó?", reply=True))
    assert result is not None
    assert len(client.sent) == 1


def test_group_mention_can_be_disabled(tmp_path):
    adapter, client = make_adapter(tmp_path, group={"respond_on_mention": False, "respond_on_reply": True})
    assert adapter.handle_event(group_event("oi", mention=True)) is None


# ── speaker resolution ────────────────────────────────────────────────────────
def test_speaker_from_contacts(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    result = adapter.handle_event(dm_event("oi", sender=ALICE, name="al"))
    assert "Alice" in result["reply"]  # mapped via contacts, not the push name


def test_speaker_falls_back_to_pushname(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    result = adapter.handle_event(dm_event("oi", sender="351999@c.us", name="Zé"))
    assert "Zé" in result["reply"]


def test_bot_mention_token_stripped(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    result = adapter.handle_event(group_event("@351900000000 quem ganhou?", mention=True))
    # the @<number> token is removed before reaching the model
    assert "@351900000000" not in result["reply"]
    assert "quem ganhou?" in result["reply"]


# ── history is per-chat and grows ──────────────────────────────────────────────
def test_history_accumulates_per_chat(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    adapter.handle_event(dm_event("primeira"))
    second = adapter.handle_event(dm_event("segunda"))
    # after one full exchange (2 lines), the second turn sees that history
    assert second["reply"].startswith("reply[Alice|2]")


BOT_LID = "111111111111111@lid"
USER_LID = "222222222222222@lid"
USER_PHONE = "351900000001"


def noweb_dm(text):
    """A NOWEB-shaped DM: @lid addressing, name+phone in _data."""
    return {
        "event": "message",
        "me": {"id": BOT_JID, "lid": BOT_LID},
        "payload": {
            "id": f"false_{USER_LID}_ABC{next(_event_seq)}",
            "from": USER_LID,
            "fromMe": False,
            "body": text,
            "_data": {
                "key": {"remoteJid": USER_LID, "remoteJidAlt": f"{USER_PHONE}@s.whatsapp.net"},
                "pushName": "Gustavo Abreu",
                "message": {"conversation": text},
            },
        },
    }


def noweb_group(text, mention_lid=None, reply_to_lid=None, quoted=None,
                stanza_id="PARENT1"):
    """A NOWEB-shaped group message with nested contextInfo.

    ``quoted`` is the quoted message OBJECT, the shape Baileys actually sends
    (``{"conversation": "..."}``, ``{"extendedTextMessage": {...}}``, a media
    message with a caption), because that nesting is the thing being parsed.
    """
    ext = {"text": text, "contextInfo": {}}
    if mention_lid:
        ext["contextInfo"]["mentionedJid"] = [mention_lid]
    if reply_to_lid:
        ext["contextInfo"]["participant"] = reply_to_lid
    if quoted is not None:
        ext["contextInfo"]["quotedMessage"] = quoted
        ext["contextInfo"]["stanzaId"] = stanza_id
    return {
        "event": "message",
        "me": {"id": BOT_JID, "lid": BOT_LID},
        "payload": {
            "id": f"false_{GROUP}_XYZ{next(_event_seq)}_{USER_LID}",
            "from": GROUP,
            "participant": USER_LID,
            "fromMe": False,
            "body": text,
            "_data": {
                "key": {"participant": USER_LID, "participantAlt": f"{USER_PHONE}@s.whatsapp.net"},
                "pushName": "Gustavo Abreu",
                "message": {"extendedTextMessage": ext},
            },
        },
    }


def test_noweb_dm_parsed_and_named(tmp_path):
    adapter, client = make_adapter(tmp_path, contacts={f"{USER_PHONE}": "Gustavo"})
    result = adapter.handle_event(noweb_dm("Olá Kaya"))
    assert result is not None
    assert len(client.sent) == 1
    assert "Gustavo" in result["reply"]  # mapped via the real phone behind the @lid


def test_noweb_group_mention_by_lid(tmp_path):
    # bot_jid is the @c.us number, but NOWEB mentions the bot by its @lid (learned from me.lid)
    adapter, client = make_adapter(tmp_path)
    silent = adapter.handle_event(noweb_group("conversa qualquer"))
    assert silent is None
    hit = adapter.handle_event(noweb_group("@111111111111111 estás vivo?", mention_lid=BOT_LID))
    assert hit is not None
    assert "@111111111111111" not in hit["reply"]  # bot-lid token stripped


def test_noweb_group_reply_to_bot_lid(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    hit = adapter.handle_event(noweb_group("e depois?", reply_to_lid=BOT_LID))
    assert hit is not None


def test_ignores_stale_backlog(tmp_path):
    adapter, client = make_adapter(tmp_path)
    adapter.ignore_before_ts = 2000
    stale = dm_event("mensagem antiga")
    stale["payload"]["timestamp"] = 1000  # before the cutoff → dropped
    assert adapter.handle_event(stale) is None
    fresh = dm_event("mensagem nova")
    fresh["payload"]["timestamp"] = 3000  # after the cutoff → answered
    assert adapter.handle_event(fresh) is not None


def test_history_isolated_between_dm_and_group(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    adapter.handle_event(dm_event("dm message"))
    grp = adapter.handle_event(group_event("oi", mention=True))
    # the group's first turn must not see the DM history
    assert grp["reply"].startswith("reply[Alice|0]")


def test_history_isolated_between_two_dms(tmp_path):
    """Two different people DMing the bot must never share context.

    Guards the privacy guarantee: KeyedSessionMemory keys history by chat_id, so
    one user's conversation can't bleed into another's.
    """
    adapter, _ = make_adapter(tmp_path)
    # Alice has a 2-line exchange in her DM.
    adapter.handle_event(dm_event("olá sou a Alice", sender=ALICE, name="Alice"))
    # Bob's first DM must see zero prior lines (his own fresh context).
    bob = adapter.handle_event(dm_event("e eu sou o Bob", sender="351922222222@c.us", name="Bob"))
    assert bob["reply"].startswith("reply[Bob|0]")
    # Alice's next turn still sees only her own history (2 lines), not Bob's.
    alice2 = adapter.handle_event(dm_event("ainda aqui", sender=ALICE, name="Alice"))
    assert alice2["reply"].startswith("reply[Alice|2]")


# ── DM whitelist (anti-spam) ────────────────────────────────────────────────────
def _wl(**extra):
    return {"enabled": True, "dm_only": True, "allowed": ["351911111111"], **extra}


def test_whitelist_blocks_non_whitelisted_dm(tmp_path):
    adapter, client = make_adapter(tmp_path, whitelist=_wl())
    # ALICE (351911111111) is allowed; a different number is silently ignored.
    assert adapter.handle_event(dm_event("spam", sender="351999999999@c.us")) is None
    assert client.sent == []


def test_whitelist_allows_whitelisted_dm(tmp_path):
    adapter, client = make_adapter(tmp_path, whitelist=_wl())
    result = adapter.handle_event(dm_event("olá", sender=ALICE))
    assert result is not None
    assert len(client.sent) == 1


def test_whitelist_disabled_allows_all_dms(tmp_path):
    adapter, client = make_adapter(tmp_path, whitelist={"enabled": False, "allowed": []})
    assert adapter.handle_event(dm_event("oi", sender="351999999999@c.us")) is not None


def test_whitelist_does_not_block_group_mentions(tmp_path):
    # A non-whitelisted member @mentioning the bot in the group still gets a reply.
    adapter, client = make_adapter(tmp_path, whitelist=_wl())
    result = adapter.handle_event(group_event("@bot olá", sender="351999999999@c.us", mention=True))
    assert result is not None
    assert len(client.sent) == 1


def test_whitelist_matches_noweb_phone_behind_lid(tmp_path):
    adapter, client = make_adapter(tmp_path, whitelist={"enabled": True, "allowed": [USER_PHONE]})
    result = adapter.handle_event(noweb_dm("Olá"))  # real phone is USER_PHONE behind the @lid
    assert result is not None
    assert len(client.sent) == 1


# ── /clear command ──────────────────────────────────────────────────────────────
def test_clear_command_wipes_history(tmp_path):
    adapter, client = make_adapter(tmp_path)
    adapter.handle_event(dm_event("primeira"))
    adapter.handle_event(dm_event("segunda"))
    result = adapter.handle_event(dm_event("/clear"))
    assert result is not None and result.get("command") == "clear"
    assert "Contexto limpo" in client.sent[-1]["text"]
    # next message starts with zero recent lines
    after = adapter.handle_event(dm_event("terceira"))
    assert after["reply"].startswith("reply[Alice|0]")


def test_clear_command_not_generated_as_reply(tmp_path):
    adapter, client = make_adapter(tmp_path)
    result = adapter.handle_event(dm_event("/limpar"))
    # the stub responder would echo "reply[..." — confirm we short-circuited it
    assert result["command"] == "clear"
    assert not result["reply"].startswith("reply[")


# ── routed commands + sticky output preference (2026-08-09) ──────────────────
class RoutedReply:
    """Mimics ``engine.Reply``: reply text plus how the message was routed."""

    def __init__(self, text="", command=None, mode="banter", citation="", telemetry=None):
        self.text = text
        self.citation = citation
        self.route = type("Route", (), {"mode": mode, "command": command})()
        self.telemetry = telemetry or {}

    @property
    def text_with_citation(self):
        return f"{self.text}\n\n{self.citation}" if self.citation else self.text


def make_routed_adapter(tmp_path, reply, **overrides):
    """Adapter whose responder returns a Reply (production shape) not a string."""
    from src.chat.memory import ChatPreferences

    config = {
        "whatsapp": {
            "bot_jid": BOT_JID,
            "contacts": {"351911111111@c.us": "Alice"},
            "send_seen": False,
            **overrides,
        }
    }
    adapter = WhatsAppAdapter(
        responder=lambda message, speaker, recent_lines, **kw: reply,
        waha_client=MockWahaClient(echo=False),
        config=config,
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "sessions")),
        prefs=ChatPreferences(base_dir=str(tmp_path / "prefs")),
    )
    return adapter


def test_audio_command_sets_sticky_preference(tmp_path):
    adapter = make_routed_adapter(tmp_path, RoutedReply(command="audio"))
    adapter.audio_reply_enabled = True   # TTS present (Phase 4)
    assert adapter.output_mode(ALICE) == "text"

    result = adapter.handle_event(dm_event("responde só em áudio"), system_prompt="")

    assert result["command"] == "audio"
    assert adapter.output_mode(ALICE) == "audio"


def test_output_preference_is_per_chat_and_survives_restart(tmp_path):
    from src.chat.memory import ChatPreferences

    adapter = make_routed_adapter(tmp_path, RoutedReply(command="audio"))
    adapter.audio_reply_enabled = True   # TTS present (Phase 4)
    adapter.handle_event(dm_event("responde só em áudio"), system_prompt="")

    # a different chat keeps the default
    assert adapter.output_mode(GROUP) == "text"
    # and the setting is on disk, so a restart keeps it
    reloaded = ChatPreferences(base_dir=str(tmp_path / "prefs"))
    assert reloaded.output_mode(ALICE) == "audio"


def test_text_command_switches_back(tmp_path):
    adapter = make_routed_adapter(tmp_path, RoutedReply(command="audio"))
    adapter.audio_reply_enabled = True   # TTS present (Phase 4)
    adapter.handle_event(dm_event("responde só em áudio"), system_prompt="")

    adapter.responder = lambda message, speaker, recent, **kw: RoutedReply(command="text")
    adapter.handle_event(dm_event("volta a texto"), system_prompt="")

    assert adapter.output_mode(ALICE) == "text"


def test_command_confirmation_is_not_model_generated(tmp_path):
    """Confirmations come from code, so they cannot drift or be refused."""
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="", command="audio"))
    adapter.audio_reply_enabled = True   # TTS present (Phase 4)
    result = adapter.handle_event(dm_event("só áudio"), system_prompt="")
    assert result["reply"] == adapter.command_replies["audio"]


def test_string_responder_still_supported(tmp_path):
    """Back-compat: the simulator and older tests hand back a bare string."""
    adapter, _ = make_adapter(tmp_path)
    result = adapter.handle_event(dm_event("olá"), system_prompt="")
    assert result["reply"].startswith("reply[")
    assert result.get("command") is None


def test_pushname_matching_an_alias_resolves_to_canonical_member(tmp_path):
    """group_members.json has no phone numbers, so the map is learned from traffic."""
    import json

    contacts_file = tmp_path / "contacts.json"
    adapter = make_routed_adapter(
        tmp_path,
        RoutedReply(text="ok", mode="factual"),
        member_aliases={"piteru": "Peter", "peter": "Peter"},
        contacts_file=str(contacts_file),
        contacts={},
    )
    msg = parse_waha_message(dm_event("olá", sender="351999000111@c.us", name="Piteru"))

    # the alias resolves to the CANONICAL name, so RAG person-filtering matches
    assert adapter.resolve_speaker(msg) == "Peter"
    # and the mapping was persisted for next time
    assert "Peter" in json.loads(contacts_file.read_text()).values()


def test_unknown_pushname_falls_back_to_pushname(tmp_path):
    adapter = make_routed_adapter(
        tmp_path, RoutedReply(text="ok"),
        member_aliases={"peter": "Peter"}, contacts={},
    )
    msg = parse_waha_message(dm_event("olá", sender="351999000222@c.us", name="Estranho"))
    assert adapter.resolve_speaker(msg) == "Estranho"


def test_audio_command_is_honest_when_tts_missing(tmp_path):
    """Confirming voice replies we cannot produce is worse than declining.

    Without TTS the command must NOT store a preference either — a stored mode
    that changes nothing is how "it said yes and then kept typing" happens.
    """
    adapter = make_routed_adapter(
        tmp_path, RoutedReply(command="audio"),
    )
    assert adapter.audio_reply_enabled is False

    result = adapter.handle_event(dm_event("responde só em áudio"), system_prompt="")

    assert result["reply"] == adapter.command_replies["audio_unavailable"]
    assert adapter.output_mode(ALICE) == "text"   # not stored


def test_audio_command_works_once_tts_is_enabled(tmp_path):
    adapter = make_routed_adapter(
        tmp_path, RoutedReply(command="audio"),
    )
    adapter.audio_reply_enabled = True

    result = adapter.handle_event(dm_event("responde só em áudio"), system_prompt="")

    assert result["reply"] == adapter.command_replies["audio"]
    assert adapter.output_mode(ALICE) == "audio"


def test_text_command_still_works_without_tts(tmp_path):
    """Going back to text is always honourable, TTS or not."""
    adapter = make_routed_adapter(tmp_path, RoutedReply(command="text"))
    result = adapter.handle_event(dm_event("volta a texto"), system_prompt="")
    assert result["reply"] == adapter.command_replies["text"]
    assert adapter.output_mode(ALICE) == "text"


# ── voice replies + incoming voice notes (Phase 4) ───────────────────────────
def test_voice_reply_sent_when_chat_prefers_audio(tmp_path):
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="Olá!", mode="banter"))
    adapter.audio_reply_enabled = True
    adapter.tts_synthesize = lambda text: b"FAKE_OGG_BYTES"
    adapter.prefs.set_output_mode(ALICE, "audio")

    adapter.handle_event(dm_event("olá"), system_prompt="")

    sent = adapter.waha_client.sent[-1]
    assert "voice_bytes" in sent and sent["voice_bytes"] == len(b"FAKE_OGG_BYTES")


def test_text_reply_when_chat_prefers_text(tmp_path):
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="Olá!", mode="banter"))
    adapter.tts_synthesize = lambda text: b"SHOULD_NOT_BE_USED"

    adapter.handle_event(dm_event("olá"), system_prompt="")

    assert adapter.waha_client.sent[-1].get("text") == "Olá!"


def test_falls_back_to_text_when_synthesis_fails(tmp_path):
    """A silent non-reply is far worse than the wrong medium."""
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="Olá!", mode="banter"))
    adapter.audio_reply_enabled = True
    adapter.tts_synthesize = lambda text: None      # synthesis failed
    adapter.prefs.set_output_mode(ALICE, "audio")

    adapter.handle_event(dm_event("olá"), system_prompt="")

    assert adapter.waha_client.sent[-1].get("text") == "Olá!"


def test_falls_back_to_text_when_sending_voice_raises(tmp_path):
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="Olá!", mode="banter"))
    adapter.audio_reply_enabled = True
    adapter.tts_synthesize = lambda text: b"OGG"
    adapter.prefs.set_output_mode(ALICE, "audio")

    def boom(*a, **kw):
        raise RuntimeError("WAHA rejected the voice note")
    adapter.waha_client.send_voice = boom

    adapter.handle_event(dm_event("olá"), system_prompt="")

    assert adapter.waha_client.sent[-1].get("text") == "Olá!"


def test_incoming_voice_note_is_transcribed_then_answered(tmp_path):
    """A voice note arrives with EMPTY text and would otherwise be dropped."""
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="Percebi!", mode="banter"))
    adapter.transcribe = lambda url, mime: "isto foi dito em voz alta"

    event = dm_event("")                       # no text, as WhatsApp sends it
    event["payload"]["media"] = {"url": "http://waha/f.oga", "mimetype": "audio/ogg"}

    result = adapter.handle_event(event, system_prompt="")

    assert result is not None, "voice note was dropped instead of transcribed"
    assert result["reply"] == "Percebi!"


def test_voice_note_without_stt_is_still_dropped(tmp_path):
    """Without transcription there is genuinely nothing to answer."""
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="x", mode="banter"))
    adapter.transcribe = None

    event = dm_event("")
    event["payload"]["media"] = {"url": "http://waha/f.oga", "mimetype": "audio/ogg"}

    assert adapter.handle_event(event, system_prompt="") is None


def test_one_off_audio_request_speaks_without_changing_the_default(tmp_path):
    """"explica isso num áudio" answers by voice but leaves the chat on text."""
    adapter = make_routed_adapter(
        tmp_path, RoutedReply(text="Aqui vai a explicação.", command="audio_once", mode="factual"))
    adapter.audio_reply_enabled = True
    adapter.tts_synthesize = lambda text: b"OGG"

    assert adapter.output_mode(ALICE) == "text"
    adapter.handle_event(dm_event("explica isso num áudio"), system_prompt="")

    assert "voice_bytes" in adapter.waha_client.sent[-1]
    # the sticky default must NOT have changed
    assert adapter.output_mode(ALICE) == "text"


def test_one_off_audio_still_generates_a_real_answer(tmp_path):
    """Unlike other commands, this one is a delivery hint, not a state change."""
    adapter = make_routed_adapter(
        tmp_path, RoutedReply(text="Resposta real.", command="audio_once", mode="factual"))
    adapter.audio_reply_enabled = True
    adapter.tts_synthesize = lambda text: b"OGG"

    result = adapter.handle_event(dm_event("manda um áudio a explicar"), system_prompt="")

    assert result["reply"] == "Resposta real."
    assert result.get("command") is None   # not treated as a pure command


# ── language-aware voice selection ───────────────────────────────────────────
# A Piper voice speaks one language. Reading an English reply with the pt_PT
# model produces Portuguese phonetics applied to English words — intelligible to
# nobody. These pin the sentence-level split that fixes it.
def test_english_reply_is_spoken_by_the_english_voice():
    from src.chat.tts import split_by_language

    assert split_by_language("The dinner is at eight, mate.") == [
        ("en", "The dinner is at eight, mate.")]


def test_portuguese_reply_stays_on_the_portuguese_voice():
    from src.chat.tts import split_by_language

    runs = split_by_language("O Peter chegou atrasado outra vez.")
    assert [lang for lang, _ in runs] == ["pt"]


def test_mixed_reply_switches_voice_per_sentence():
    """This group code-switches mid-reply; one voice for the lot sounds wrong."""
    from src.chat.tts import split_by_language

    runs = split_by_language(
        "Bora lá pessoal, o jantar é às oito. Honestly mate, that is a terrible idea. "
        "Mas se quiseres, eu vou na mesma.")
    assert [lang for lang, _ in runs] == ["pt", "en", "pt"]
    assert "Honestly" in runs[1][1]


def test_consecutive_same_language_sentences_are_one_run():
    """Fewer runs = fewer voice loads and no seam inside a single language."""
    from src.chat.tts import split_by_language

    runs = split_by_language("Quem é o Peter? É o mais alto do grupo.")
    assert len(runs) == 1


def test_unmarked_sentence_inherits_the_surrounding_language():
    """"Absolutely brutal." carries no marker; defaulting it to PT mid-English
    reply is the exact mispronunciation this feature removes."""
    from src.chat.tts import split_by_language

    runs = split_by_language("The dinner is at eight. Absolutely brutal. Everyone is coming.")
    assert [lang for lang, _ in runs] == ["en"]


def test_reassembled_runs_lose_no_text():
    from src.chat.tts import split_by_language

    text = "Olá! Ready? Vamos embora, pá."
    assert "".join(chunk for _, chunk in split_by_language(text)).strip() == text.strip()


def test_voice_paths_fall_back_to_the_portuguese_voice(tmp_path):
    """A missing English model must degrade to PT audio, never to no reply."""
    from src.chat import tts

    config = {"chat": {"audio": {"reply_enabled": True,
                                 "voices": {"en": str(tmp_path / "missing.onnx")}}}}
    # The fallback itself is pure path logic and is always checked — it is what
    # this test is named for.
    assert tts._voice_paths(config)["pt"] == tts.DEFAULT_VOICES["pt"]

    # `is_available` additionally stats the Portuguese model on disk. The Piper
    # voices are gitignored, so a clean checkout (CI) does not have them; that is
    # a missing fixture, not a defect, and it must not read as one.
    if not tts._resolve(tts.DEFAULT_VOICES["pt"]).exists():
        pytest.skip("Piper PT voice model not present (gitignored); "
                    "fetch it to exercise the availability check")
    assert tts.is_available(config) is True


def image_group_event(text, media_url="", mimetype=""):
    event = group_event(text, mention=True)
    if media_url:
        event["payload"]["media"] = {"url": media_url, "mimetype": mimetype}
    return event


# ── asking for a picture (removed 2026-09-04) ────────────────────────────────
# Generation and editing are gone. CMD_IMAGE is deliberately kept so the ask
# gets a fixed answer rather than falling through to the model, which used to
# promise a picture and then never send one.
def make_declining_adapter(tmp_path):
    from src.chat.memory import ChatPreferences

    config = {"whatsapp": {"bot_jid": BOT_JID, "send_seen": False,
                           "shared_chats": [GROUP]}}
    return WhatsAppAdapter(
        responder=lambda message, speaker, recent_lines, **kw: RoutedReply(
            command="image", mode="factual"),
        waha_client=MockWahaClient(echo=False),
        config=config,
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "sessions")),
        prefs=ChatPreferences(base_dir=str(tmp_path / "prefs")),
    )


def test_an_image_request_is_declined_not_attempted(tmp_path):
    adapter = make_declining_adapter(tmp_path)

    result = adapter.handle_event(
        image_group_event("faz uma imagem de um gato astronauta"),
        system_prompt="")

    assert result["command"] == "image"
    assert "não faço" in result["reply"].lower()
    assert len(adapter.waha_client.sent) == 1


def test_an_edit_request_with_a_photo_is_declined_too(tmp_path):
    """An attached photo used to make this the unambiguous edit path."""
    adapter = make_declining_adapter(tmp_path)

    result = adapter.handle_event(
        image_group_event("põe-lhe uma coroa", media_url="http://waha:3000/f.jpg",
                          mimetype="image/jpeg"),
        system_prompt="")

    assert result["command"] == "image"
    assert "não faço" in result["reply"].lower()


def test_the_decline_never_promises_a_picture(tmp_path):
    """The failure this replaces: "Vou fazer isso, dá-me um bocado." followed by
    nothing. Nothing in the reply may suggest one is coming."""
    adapter = make_declining_adapter(tmp_path)

    result = adapter.handle_event(
        image_group_event("faz uma imagem de um gato astronauta"),
        system_prompt="")

    lowered = result["reply"].lower()
    for promise in ("já mando", "dá-me", "uns minutos", "em fila", "assim que"):
        assert promise not in lowered, promise


def test_no_image_is_ever_sent(tmp_path):
    """send_image is gone from the client; nothing may try to call it."""
    adapter = make_declining_adapter(tmp_path)

    adapter.handle_event(image_group_event("faz uma imagem de um cão"),
                         system_prompt="")

    assert not any("image_bytes" in s for s in adapter.waha_client.sent)
    assert not hasattr(adapter.waha_client, "send_image")


# ── reading inbound photos ───────────────────────────────────────────────────
# A photo carries no text. Without a description the message is either dropped or
# answered as though nothing were attached — "não recebi nenhuma imagem" while
# the picture sits in the chat. Described, it becomes ordinary text and
# everything downstream (memory, ingestion, routing) works unchanged.
def make_vision_adapter(tmp_path, description="dois homens num barco com cervejas"):
    from src.chat.memory import ChatPreferences

    seen = []

    def describe(url, mimetype):
        seen.append({"url": url, "mimetype": mimetype})
        return description

    adapter = WhatsAppAdapter(
        responder=lambda message, speaker, recent_lines, **kw: f"vi: {message}",
        waha_client=MockWahaClient(echo=False),
        config={"whatsapp": {"bot_jid": BOT_JID, "send_seen": False,
                             "shared_chats": [GROUP]}},
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "sessions")),
        prefs=ChatPreferences(base_dir=str(tmp_path / "prefs")),
        describe_image=describe,
    )
    adapter.described = seen
    return adapter


def photo_event(caption="", mimetype="image/jpeg", timestamp=1700000000):
    event = group_event(caption, mention=True)
    event["payload"]["media"] = {"url": "http://waha:3000/photo.jpg", "mimetype": mimetype}
    # MessageLog.read() only yields records newer than its cutoff, so a message
    # with no timestamp is written and never read back.
    event["payload"]["timestamp"] = timestamp
    return event


def test_photo_without_a_caption_is_still_understood(tmp_path):
    adapter = make_vision_adapter(tmp_path)

    result = adapter.handle_event(photo_event(), system_prompt="")

    assert result is not None, "a photo with no caption must not be dropped"
    assert "barco" in result["reply"]


def test_caption_is_kept_alongside_the_description(tmp_path):
    """"quem é este?" is the question; the description is the evidence."""
    adapter = make_vision_adapter(tmp_path)

    result = adapter.handle_event(photo_event("quem é este?"), system_prompt="")

    assert "quem é este?" in result["reply"]
    assert "barco" in result["reply"]


def test_a_photo_is_not_sent_to_the_transcriber(tmp_path):
    """Whisper on a JPEG wastes a GPU load and returns nothing useful."""
    from src.chat.memory import ChatPreferences

    transcribed = []
    adapter = WhatsAppAdapter(
        responder=lambda message, speaker, recent_lines, **kw: "ok",
        waha_client=MockWahaClient(echo=False),
        config={"whatsapp": {"bot_jid": BOT_JID, "send_seen": False}},
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "sessions")),
        prefs=ChatPreferences(base_dir=str(tmp_path / "prefs")),
        transcribe=lambda url, mimetype: transcribed.append(url) or "nope",
        describe_image=lambda url, mimetype: "uma foto",
    )

    adapter.handle_event(photo_event(), system_prompt="")

    assert transcribed == []


def test_described_photo_is_logged_as_memory(tmp_path):
    """This is what makes "aquela foto do barco" findable a week later."""
    from src.chat.memory import ChatPreferences
    from src.data.message_log import MessageLog

    log = MessageLog(base_dir=str(tmp_path / "log"))
    adapter = WhatsAppAdapter(
        responder=lambda message, speaker, recent_lines, **kw: "ok",
        waha_client=MockWahaClient(echo=False),
        config={"whatsapp": {"bot_jid": BOT_JID, "send_seen": False,
                             "log_messages": True, "shared_chats": [GROUP]}},
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "sessions")),
        prefs=ChatPreferences(base_dir=str(tmp_path / "prefs")),
        message_log=log,
        describe_image=lambda url, mimetype: "dois homens num barco",
    )

    adapter.handle_event(photo_event(), system_prompt="")

    logged = [m["text"] for m in log.read("shared")]
    assert any("barco" in text for text in logged)


def test_vision_failure_falls_back_to_the_old_behaviour(tmp_path):
    """A dead vision server must not start dropping every photo message."""
    adapter = make_vision_adapter(tmp_path, description=None)

    result = adapter.handle_event(photo_event("olhem isto"), system_prompt="")

    assert result is not None
    assert "olhem isto" in result["reply"]


# ── replay protection ────────────────────────────────────────────────────────

def test_a_replayed_message_is_answered_only_once(tmp_path):
    """WAHA replays its backlog after a reconnect. Answering twice is visible to
    everyone in the group and was what the simulator caught."""
    adapter, client = make_adapter(tmp_path)
    event = dm_event("quem é o Peter?")
    event["payload"]["id"] = "replayed-1"

    first = adapter.handle_event(event)
    second = adapter.handle_event(event)

    assert first is not None
    assert second is None, "the replay should have been ignored"
    assert len(client.sent) == 1


def test_different_messages_are_both_answered(tmp_path):
    """The dedup must key on the id, not suppress everything after the first."""
    adapter, client = make_adapter(tmp_path)
    for index, text in enumerate(("olá", "tudo bem?")):
        event = dm_event(text)
        event["payload"]["id"] = f"distinct-{index}"
        assert adapter.handle_event(event) is not None
    assert len(client.sent) == 2


# ── what is SPOKEN vs what is WRITTEN ────────────────────────────────────────
# Every voice test above stubs TTS as `lambda text: b"OGG"` and asserts on byte
# counts, so nothing noticed that a web-grounded reply was handing Piper its
# "🌐 Fontes: x.com, play.google.com" line to read out domain by domain.

def _capture_tts(adapter):
    """Record the exact string handed to the synthesiser."""
    seen = {}

    def tts(text):
        seen["text"] = text
        return b"OGG"

    adapter.tts_synthesize = tts
    return seen


def test_voice_note_never_speaks_the_sources_line(tmp_path):
    from src.chat.tts import sanitize_for_speech

    adapter = make_routed_adapter(
        tmp_path,
        RoutedReply(text="O Benfica ganhou 6-1.", mode="factual",
                    citation="🌐 Fontes: espn.com.br, pt.uefa.com"),
    )
    adapter.audio_reply_enabled = True
    adapter.speech_text = sanitize_for_speech
    adapter.prefs.set_output_mode(ALICE, "audio")
    spoken = _capture_tts(adapter)

    adapter.handle_event(dm_event("quem ganhou?"), system_prompt="")

    assert spoken["text"] == "O Benfica ganhou 6-1."
    assert "Fontes" not in spoken["text"]
    assert "espn" not in spoken["text"]


def test_sources_follow_the_voice_note_as_text(tmp_path):
    from src.chat.tts import sanitize_for_speech

    adapter = make_routed_adapter(
        tmp_path,
        RoutedReply(text="O Benfica ganhou 6-1.", mode="factual",
                    citation="🌐 Fontes: espn.com.br"),
    )
    adapter.audio_reply_enabled = True
    adapter.speech_text = sanitize_for_speech
    adapter.prefs.set_output_mode(ALICE, "audio")
    _capture_tts(adapter)

    adapter.handle_event(dm_event("quem ganhou?"), system_prompt="")

    sent = adapter.waha_client.sent
    assert any("voice_bytes" in item for item in sent)
    assert any(item.get("text") == "🌐 Fontes: espn.com.br" for item in sent)


def test_written_reply_still_carries_the_sources_line(tmp_path):
    adapter = make_routed_adapter(
        tmp_path,
        RoutedReply(text="O Benfica ganhou 6-1.", mode="factual",
                    citation="🌐 Fontes: espn.com.br"),
    )

    adapter.handle_event(dm_event("quem ganhou?"), system_prompt="")

    assert adapter.waha_client.sent[-1]["text"] == (
        "O Benfica ganhou 6-1.\n\n🌐 Fontes: espn.com.br"
    )


def test_spoken_text_is_recorded_for_the_voice_note(tmp_path):
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="Olá!", mode="banter"))
    adapter.audio_reply_enabled = True
    adapter.prefs.set_output_mode(ALICE, "audio")
    adapter.tts_synthesize = lambda text: b"OGG"

    result = adapter.handle_event(dm_event("olá"), system_prompt="")

    assert result["delivered_as"] == "voice"
    assert result["spoken_text"] == "Olá!"
    assert adapter.waha_client.sent[-1]["spoken_text"] == "Olá!"


def test_text_delivery_is_reported_as_text(tmp_path):
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="Olá!", mode="banter"))

    result = adapter.handle_event(dm_event("olá"), system_prompt="")

    assert result["delivered_as"] == "text"
    assert result["spoken_text"] == ""


def test_no_canned_reply_carries_a_dash(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    for key, reply in adapter.command_replies.items():
        assert "—" not in reply and "–" not in reply, key
        assert " - " not in reply, key


# ── the verbatim window vs retrieval ─────────────────────────────────────────
# `exclude_from` is what stops retrieval re-injecting turns the prompt already
# carries word for word. It had no test at all, which matters more now that
# history_turns is 60 rather than 6: the window start moved a long way back, so
# a mistake here silently changes what the model can recall.

def _capture_responder(tmp_path, **overrides):
    """An adapter whose responder records the kwargs it was handed."""
    seen = {}

    def responder(message, speaker, recent_lines, scope=None, exclude_from=None,
                  summary=""):
        seen["scope"] = scope
        seen["exclude_from"] = exclude_from
        seen["summary"] = summary
        seen["recent"] = list(recent_lines or [])
        return "ok"

    config = {"whatsapp": {"bot_jid": BOT_JID, "send_seen": False,
                           "contacts": {ALICE: "Alice"}, **overrides}}
    adapter = WhatsAppAdapter(
        responder, MockWahaClient(echo=False), config,
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "s"), max_lines=200),
    )
    return adapter, seen


def _dm_at(text, ts):
    event = dm_event(text)
    event["payload"]["timestamp"] = ts
    return event


def test_no_window_start_before_anything_has_been_said(tmp_path):
    adapter, seen = _capture_responder(tmp_path)
    adapter.handle_event(_dm_at("olá", 1_700_000_000), system_prompt="")
    # One message in: the window starts at that message, not before it.
    assert seen["exclude_from"] is not None


def test_the_window_start_follows_history_turns(tmp_path):
    adapter, seen = _capture_responder(tmp_path, history_turns=6)
    base = 1_700_000_000
    for i in range(10):
        adapter.handle_event(_dm_at(f"mensagem {i}", base + i), system_prompt="")
    # In a DM with one reply per message, 6 lines is 3 answered turns: three
    # inbound lines plus the message being answered, so the window covers four
    # messages — the 10th, 9th, 8th and 7th. It starts at the 7th (base + 6).
    from datetime import datetime, timezone

    expected = datetime.fromtimestamp(base + 6, tz=timezone.utc).replace(
        tzinfo=None).isoformat()
    assert seen["exclude_from"] == expected, (
        "the window must be measured in inbound messages, not session lines")


def test_a_bigger_window_reaches_further_back(tmp_path):
    """Raising history_turns must move the boundary earlier, not later."""
    base = 1_700_000_000
    starts = {}
    for turns in (3, 10):
        adapter, seen = _capture_responder(tmp_path / f"t{turns}", history_turns=turns)
        for i in range(12):
            adapter.handle_event(_dm_at(f"m{i}", base + i), system_prompt="")
        starts[turns] = seen["exclude_from"]
    assert starts[10] < starts[3], "a longer verbatim window must start earlier"


def test_messages_without_a_timestamp_do_not_corrupt_the_window(tmp_path):
    adapter, seen = _capture_responder(tmp_path, history_turns=3)
    adapter.handle_event(_dm_at("com tempo", 1_700_000_000), system_prompt="")
    event = dm_event("sem tempo")
    event["payload"]["timestamp"] = 0
    adapter.handle_event(event, system_prompt="")
    # The zero is ignored rather than becoming 1970 and excluding everything.
    assert seen["exclude_from"].startswith("2023-")


def test_the_rolling_summary_reaches_the_responder(tmp_path):
    class Writer:
        class store:
            @staticmethod
            def summary_for(chat_id):
                return "Ficou combinado jantar no sábado."

        @staticmethod
        def maybe_update(chat_id, history):
            return False

    adapter, seen = _capture_responder(tmp_path)
    adapter.summary_writer = Writer()
    adapter._responder_takes_summary = True
    adapter.handle_event(_dm_at("e então?", 1_700_000_000), system_prompt="")
    assert seen["summary"] == "Ficou combinado jantar no sábado."


def test_a_responder_that_takes_no_summary_still_works(tmp_path):
    """The simulators and older stubs must keep working unchanged."""
    def old_style(message, speaker, recent_lines, scope=None, exclude_from=None):
        return f"reply:{message}"

    config = {"whatsapp": {"bot_jid": BOT_JID, "send_seen": False}}
    adapter = WhatsAppAdapter(
        old_style, MockWahaClient(echo=False), config,
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "s2")),
    )
    assert adapter._responder_takes_summary is False
    result = adapter.handle_event(_dm_at("olá", 1_700_000_000), system_prompt="")
    assert result["reply"] == "reply:olá"


def test_the_exclusion_window_counts_inbound_messages_not_lines(tmp_path):
    """The window start must not reach back further than the prompt actually goes.

    How many inbound messages `history_turns` lines represent is not a fixed
    fraction: every message the bot SEES is a line now, so a burst of chatter
    fills the window with inbound lines and a quiet exchange with alternating
    ones. The count has to come from the lines actually being sent. Guessing it
    pushes the start too far back, and retrieval then drops chunks covering
    messages that are NOT held verbatim — a hole, not a duplicate, and silent.
    """
    adapter, seen = _capture_responder(tmp_path, history_turns=6)
    base = 1_700_000_000
    for i in range(12):
        adapter.handle_event(_dm_at(f"m{i}", base + i), system_prompt="")

    lines = seen["recent"]
    inbound = [ln for ln in lines if not ln.startswith("Kaya Bot:")]
    assert len(lines) <= 6

    from datetime import datetime, timezone

    start = seen["exclude_from"]
    # The prompt carries the history lines PLUS the message being answered.
    carried = len(inbound) + 1
    oldest_inbound_ts = base + 12 - carried
    earliest_allowed = datetime.fromtimestamp(
        oldest_inbound_ts, tz=timezone.utc).replace(tzinfo=None).isoformat()
    assert start >= earliest_allowed, (
        f"window starts at {start}, earlier than the oldest message actually in "
        f"the prompt ({earliest_allowed}) — chunks in between would be excluded "
        f"from retrieval without being carried verbatim")
    assert start == earliest_allowed, (
        "the boundary is counted from the lines being sent, so it should land "
        "exactly on the oldest message carried, not short of it")


# ── /bug and /feedback (2026-08-13) ──────────────────────────────────────────
# The collection channel for the listening week. Two things are load-bearing:
# the reports must reach disk, and they must NOT reach the memory log — that log
# is embedded into ChromaDB, so a logged "/bug" comes back out of retrieval later
# as something the group supposedly said.

def make_report_adapter(tmp_path, client=None, **overrides):
    """An adapter whose feedback sinks and message log live under tmp_path."""
    from src.chat.memory import ChatPreferences
    from src.data.message_log import MessageLog

    log = MessageLog(base_dir=str(tmp_path / "msglog"))
    config = {
        "whatsapp": {
            "bot_jid": BOT_JID,
            "group": {"respond_on_mention": True, "respond_on_reply": True},
            "send_seen": False,
            "log_messages": True,
            "shared_chats": [GROUP],
            "report_to": "351999999999@c.us",
            **overrides,
        },
        "chat": {
            "bug_report": {"log_file": str(tmp_path / "bugs.jsonl")},
            "feedback": {"log_file": str(tmp_path / "feedback.jsonl")},
        },
    }
    adapter = WhatsAppAdapter(
        responder=lambda message, speaker, recent_lines, **kw: f"reply[{speaker}]",
        waha_client=client or MockWahaClient(echo=False),
        config=config,
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "sessions")),
        prefs=ChatPreferences(base_dir=str(tmp_path / "prefs")),
        message_log=log,
    )
    return adapter, adapter.waha_client, log


def _rows(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_bug_command_records_the_report(tmp_path):
    adapter, client, _ = make_report_adapter(tmp_path)

    result = adapter.handle_event(dm_event("/bug o áudio não funcionou"))

    assert result["command"] == "bug" and result["logged"] is True
    rows = _rows(tmp_path / "bugs.jsonl")
    assert len(rows) == 1
    assert rows[0]["description"] == "o áudio não funcionou"
    assert rows[0]["source"] == "whatsapp"
    # never generated on: the stub responder would have echoed "reply["
    assert not result["reply"].startswith("reply[")


def test_feedback_command_records_a_note(tmp_path):
    adapter, _, _ = make_report_adapter(tmp_path)

    result = adapter.handle_event(dm_event("/feedback devias ser mais curto"))

    assert result["command"] == "feedback" and result["logged"] is True
    rows = _rows(tmp_path / "feedback.jsonl")
    assert len(rows) == 1 and rows[0]["type"] == "note"
    assert rows[0]["text"] == "devias ser mais curto"


# ── an unknown /command (2026-09-04) ─────────────────────────────────────────
# "/feature have better update of facts, give more importance or update facts
# with new information" was not in the 7-token command table, so it fell through
# to the model. It routed GENERAL and answered "Understood. I will prioritize and
# integrate new information more aggressively... Expect more relevant updates in
# our future interactions" — a promise it has no state to keep — and the whole
# line went into the memory log, and from there into ChromaDB as a searchable
# thing "the group said".
def test_an_unknown_command_is_not_answered_by_the_model(tmp_path):
    adapter, _, _ = make_report_adapter(tmp_path)

    result = adapter.handle_event(dm_event("/feature have better update of facts"))

    assert result["command"] == "unknown"
    # the stub responder echoes "reply["; the model must never have run
    assert not result["reply"].startswith("reply[")
    assert "/bug" in result["reply"] and "/feedback" in result["reply"]


def test_an_unknown_command_never_promises_anything(tmp_path):
    adapter, _, _ = make_report_adapter(tmp_path)

    reply = adapter.handle_event(
        dm_event("/feature keep a counter of swear words"))["reply"].lower()

    for promise in ("vou ", "understood", "expect", "prometo", "a partir de agora"):
        assert promise not in reply, promise


def test_an_unknown_command_is_kept_out_of_the_memory_log(tmp_path):
    """The half that matters: it is embedded into ChromaDB by the ingester."""
    adapter, _, _ = make_report_adapter(tmp_path)

    adapter.handle_event(dm_event("/feature have better update of facts"))

    written = "".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "msglog").rglob("*.jsonl"))
    assert "/feature" not in written


def test_an_unknown_command_stores_nothing_and_captures_nothing(tmp_path):
    """No pending-capture state: the NEXT message must not be swallowed."""
    adapter, _, _ = make_report_adapter(tmp_path)

    adapter.handle_event(dm_event("/feature"))
    after = adapter.handle_event(dm_event("boa tarde"))

    assert not after.get("command")
    assert not _rows(tmp_path / "bugs.jsonl")
    assert not _rows(tmp_path / "feedback.jsonl")


def test_a_known_command_is_still_matched_mid_message(tmp_path):
    """The unknown-command branch must not shadow the mid-message scan."""
    adapter, _, _ = make_report_adapter(tmp_path)

    result = adapter.handle_event(
        dm_event("Andas a repetir-te. /feedback varia mais as respostas"))

    assert result["command"] == "feedback"
    assert _rows(tmp_path / "feedback.jsonl")[0]["text"] == "varia mais as respostas"


def test_ordinary_prose_containing_a_slash(tmp_path):
    """Only a LEADING /word counts. A date or a "sim/não" is not a command."""
    adapter, _, _ = make_report_adapter(tmp_path)

    for text in ("vamos dia 12/09 ao jantar", "sim/não, decide lá",
                 "olha isto https://x.com/algo"):
        result = adapter.handle_event(dm_event(text))
        assert result.get("command") != "unknown", text


def test_the_general_mode_prompt_carries_the_no_promises_clause():
    """general was the ONLY mode prompt without {maintainer_clause} — which is
    why the promise got through, since /feature routed GENERAL."""
    from src.config_loader import load_config

    modes = load_config("config.yaml")["chat"]["modes"]
    for name, mode in modes.items():
        prompt = mode.get("system_prompt")
        if prompt is None:
            continue  # inherits data.system_prompt, which has it
        assert "{maintainer_clause}" in prompt, name


def test_confirmations_carry_no_emoji(tmp_path):
    """Explicitly asked for: these replies are plain text."""
    adapter, client, _ = make_report_adapter(tmp_path)
    adapter.handle_event(dm_event("/bug partiu-se"))
    adapter.handle_event(dm_event("/feedback mais piadas"))

    for sent in client.sent:
        assert all(ord(ch) < 0x2000 for ch in sent["text"]), sent["text"]


def test_a_bare_command_explains_itself_and_stores_nothing(tmp_path):
    """No pending-capture state, so an unrelated next message is never swallowed."""
    adapter, client, _ = make_report_adapter(tmp_path)

    result = adapter.handle_event(dm_event("/bug"))

    assert result["command"] == "bug" and result["logged"] is False
    assert "/bug" in client.sent[-1]["text"]
    assert _rows(tmp_path / "bugs.jsonl") == []


def test_reports_never_reach_the_memory_log(tmp_path):
    """The whole point: these must not become searchable group memory."""
    adapter, _, log = make_report_adapter(tmp_path)

    adapter.handle_event(dm_event("/bug isto está partido"))
    adapter.handle_event(dm_event("/feedback sê mais curto"))
    adapter.handle_event(dm_event("/clear"))
    adapter.handle_event(dm_event("uma mensagem normal"))

    # after_ts=-1 because dm_event carries no timestamp and read() filters on
    # `> after_ts`; a real WhatsApp message always has one.
    logged = [m["text"] for m in log.read(scope_for_chat(ALICE, {GROUP}), after_ts=-1)]
    assert "uma mensagem normal" in logged
    assert not any(text.strip().startswith("/") for text in logged), logged


def test_bug_in_the_group_is_recognised_behind_a_mention(tmp_path):
    """In a group the text arrives as "@<bot> /bug ..." and must still parse."""
    adapter, _, log = make_report_adapter(tmp_path)

    result = adapter.handle_event(
        group_event(f"@{BOT_JID.split('@')[0]} /bug não respondeu", mention=True))

    assert result["command"] == "bug" and result["logged"] is True
    assert _rows(tmp_path / "bugs.jsonl")[0]["description"] == "não respondeu"
    assert not any("/bug" in m["text"] for m in log.read("shared", after_ts=-1))


def test_group_report_also_reaches_the_reporter_privately(tmp_path):
    adapter, client, _ = make_report_adapter(tmp_path)

    adapter.handle_event(
        group_event(f"@{BOT_JID.split('@')[0]} /bug não respondeu", mention=True))

    targets = [s["chat_id"] for s in client.sent]
    assert GROUP in targets                      # the public confirmation
    assert "351999999999@c.us" in targets        # Gustavo
    assert ALICE in targets                      # the reporter's own copy


def test_a_dm_report_is_not_confirmed_twice(tmp_path):
    adapter, client, _ = make_report_adapter(tmp_path)

    adapter.handle_event(dm_event("/bug partiu-se"))

    assert [s["chat_id"] for s in client.sent].count(ALICE) == 1


def test_a_failed_notification_still_records_the_report(tmp_path):
    """The report is already on disk; a dead side-channel must not lose it."""
    class HalfDeadClient(MockWahaClient):
        def send_text(self, chat_id, text):
            if chat_id == "351999999999@c.us":
                raise RuntimeError("WAHA is down")
            return super().send_text(chat_id, text)

    adapter, client, _ = make_report_adapter(tmp_path, client=HalfDeadClient(echo=False))

    result = adapter.handle_event(dm_event("/bug partiu-se"))

    assert result["logged"] is True
    assert len(_rows(tmp_path / "bugs.jsonl")) == 1
    assert "Registado" in client.sent[-1]["text"]


# ── routing telemetry (2026-08-13) ───────────────────────────────────────────
def test_the_result_carries_the_routing_telemetry(tmp_path):
    """How a turn was routed has to reach the interaction log.

    The log recorded latency and delivery medium but never the route, so the
    "it is obsessed with the Gil" complaint could not be measured — only
    re-read. whatsapp_server merges this dict into log_interaction's extras.
    """
    telemetry = {"route_mode": "factual", "route_fallback": False,
                 "reply_members": ["Gil"], "query_members": []}
    adapter = make_routed_adapter(
        tmp_path, RoutedReply(text="o Gil outra vez", mode="factual", telemetry=telemetry))

    result = adapter.handle_event(dm_event("quem é o mais paneleiro?"), system_prompt="")

    assert result["telemetry"] == telemetry


def test_a_plain_string_responder_still_works(tmp_path):
    """Tests and the simulators return a bare string, not a Reply."""
    adapter, _ = make_adapter(tmp_path)

    result = adapter.handle_event(dm_event("olá"))

    assert result["telemetry"] == {}


# ── commands typed mid-message (2026-08-13) ──────────────────────────────────
def test_feedback_mid_message_is_recorded(tmp_path):
    """The most useful note of the first group session was typed mid-sentence.

    "Andas a dizer demasiado foda-se no fim das frases. /feedback o problema é
    a construção frásica" matched nothing, because only the first token was
    checked. The model answered it — promising to do better — and the line went
    into group memory instead of the feedback log.
    """
    adapter, client, tmp = make_report_adapter(tmp_path)

    result = adapter.handle_event(dm_event(
        "Andas a dizer demasiado foda se no fim das frases. "
        "/feedback o problema é a construção frásica ser repetitiva"))

    assert result["logged"] is True
    rows = _rows(tmp_path / "feedback.jsonl")
    assert rows[-1]["text"] == "o problema é a construção frásica ser repetitiva"


def test_the_run_up_is_context_not_part_of_the_report(tmp_path):
    """Only what follows the command word is the report."""
    adapter, _, _ = make_report_adapter(tmp_path)

    adapter.handle_event(dm_event("isto está estranho /bug o áudio corta a meio"))

    assert _rows(tmp_path / "bugs.jsonl")[-1]["description"] == "o áudio corta a meio"


def test_an_inline_command_stays_out_of_the_memory_log(tmp_path):
    """The log is written before the reply gate — a report must not become
    something 'the group said' and come back out of retrieval a week later."""
    adapter, _, _ = make_report_adapter(tmp_path)
    logged = []
    adapter.log_messages = True
    adapter.message_log = type("Log", (), {"append": lambda self, **kw: logged.append(kw)})()

    adapter.handle_event(dm_event("a voz é podre /feedback experimenta outro TTS"))

    assert logged == []


def test_clear_still_has_to_be_the_whole_message(tmp_path):
    """A wrongly triggered wipe destroys a live conversation's context; a missed
    one is retyped. The two mistakes are not symmetric, so /clear stays strict."""
    adapter, client = make_adapter(tmp_path)

    result = adapter.handle_event(dm_event("podes fazer /clear a isso?"))

    assert result.get("command") != "clear"


# ── concurrent writes to one chat's history (2026-08-13) ─────────────────────
def test_concurrent_appends_do_not_lose_lines(tmp_path):
    """append() is a read-modify-write and more than one thread does it.

    The webhook thread appends "(a preparar uma imagem…)" while the image
    queue worker appends "(imagem enviada)" for the previous job. Unlocked,
    whichever loaded first won and the other line was simply gone — measured at
    60-95% of lines lost under three writers, plus save failures, because the
    atomic write used one fixed ".tmp" path that both threads replaced.
    """
    import threading

    store = KeyedSessionMemory(base_dir=str(tmp_path / "sessions"), max_lines=500)

    def writer(tag):
        for index in range(30):
            store.append("chat@g.us", f"{tag}{index}")

    threads = [threading.Thread(target=writer, args=(tag,)) for tag in "ABC"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(store.recent("chat@g.us", None)) == 90


def test_a_save_leaves_no_scratch_file_behind(tmp_path):
    session_dir = tmp_path / "sessions"
    store = KeyedSessionMemory(base_dir=str(session_dir), max_lines=50)

    store.append("chat@g.us", "Gustavo: olá")

    assert [p.name for p in session_dir.iterdir()] == ["chat_g.us.json"]


def test_each_chat_locks_independently(tmp_path):
    """One chat's write must not serialise another's."""
    store = KeyedSessionMemory(base_dir=str(tmp_path / "sessions"), max_lines=50)

    assert store._chat_lock("a@g.us") is not store._chat_lock("b@g.us")
    assert store._chat_lock("a@g.us") is store._chat_lock("a@g.us")


# ── an edit that changed nothing (2026-08-13) ────────────────────────────────
def _wait_for_text(adapter, needle, timeout=5.0):
    """The failure message is sent from the image thread, like the image is."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(needle in (s.get("text") or "") for s in adapter.waha_client.sent):
            return True
        time.sleep(0.02)
    return False


# ── the message a reply is answering (bug 7acbc092, 2026-08-13) ──────────────
# Gil replied to an earlier message with "A culpa é do" and the bot answered
# "A culpa é do quem? Completa lá a frase, papi." parse_waha_message read the
# quoted message for its SENDER — enough to decide whether to answer — and threw
# the body away, so the model had nothing to attach the fragment to.
PARENT = "Os judeus é que mandam nisto tudo"


def test_the_quoted_message_is_parsed_from_noweb():
    msg = parse_waha_message(
        noweb_group("A culpa é do", quoted={"conversation": PARENT}))

    assert msg.quoted_text == PARENT
    assert msg.reply_to_id == "PARENT1"


def test_a_quoted_message_with_context_is_unwrapped():
    """A quoted message that itself had context nests one level deeper."""
    msg = parse_waha_message(
        noweb_group("sim", quoted={"extendedTextMessage": {"text": "quem vem ao jantar?"}}))

    assert msg.quoted_text == "quem vem ao jantar?"


def test_a_quoted_photo_contributes_its_caption():
    msg = parse_waha_message(
        noweb_group("Esse é meu dog", quoted={"imageMessage": {"caption": "o meu cão novo"}}))

    assert msg.quoted_text == "o meu cão novo"


def test_the_webjs_shape_is_read_too():
    """The bridge has to work with either WAHA engine."""
    event = group_event("Ao contrário de outros", mention=True)
    event["payload"]["quotedMsg"] = {"body": PARENT, "id": "P2", "participant": ALICE}

    msg = parse_waha_message(event)

    assert msg.quoted_text == PARENT
    assert msg.reply_to_id == "P2"


def test_a_reply_with_no_quoted_body_is_not_an_error():
    """WAHA does not always send one — a reply to something it never saw
    arrives with the participant and nothing else."""
    msg = parse_waha_message(noweb_group("yah", reply_to_lid=USER_LID))

    assert msg.quoted_text == ""
    assert msg.reply_to_id == ""


def test_the_model_is_shown_what_is_being_replied_to(tmp_path):
    """The actual bug: the fragment must arrive with its parent attached."""
    adapter, _ = make_adapter(tmp_path)
    seen = {}
    adapter.responder = lambda message, speaker, recent, **kw: seen.setdefault("msg", message) or "ok"

    adapter.handle_event(
        noweb_group("A culpa é do", mention_lid=BOT_LID, quoted={"conversation": PARENT}))

    assert PARENT in seen["msg"]
    assert "A culpa é do" in seen["msg"]


def test_an_ordinary_message_gains_no_prefix(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    seen = {}
    adapter.responder = lambda message, speaker, recent, **kw: seen.setdefault("msg", message) or "ok"

    adapter.handle_event(noweb_group("boas malta", mention_lid=BOT_LID))

    assert seen["msg"] == "boas malta"


def test_the_quote_survives_into_the_history(tmp_path):
    """The parent is usually NOT in this history — the bot only records turns it
    answered — so dropping it here leaves the follow-up turn as contextless."""
    adapter, _ = make_adapter(tmp_path)

    adapter.handle_event(
        noweb_group("A culpa é do", mention_lid=BOT_LID, quoted={"conversation": PARENT}))

    assert any(PARENT in line for line in adapter.session_store.recent(GROUP, 10))


def test_a_quoted_bot_message_is_named_as_the_bot(tmp_path):
    adapter, _ = make_adapter(tmp_path)

    msg = parse_waha_message(
        noweb_group("mentira", reply_to_lid=BOT_JID, quoted={"conversation": "O Gil é o pior"}))

    assert "Kaya Bot" in adapter.quoted_context(msg)


def test_a_command_quoting_something_is_still_a_command(tmp_path):
    """The prefix is added after the command check, so "/bug" used as a reply
    still records a report rather than becoming a message about one."""
    adapter, client, _ = make_report_adapter(tmp_path)

    result = adapter.handle_event(dm_event("/bug o áudio corta a meio"))

    assert result["logged"] is True


# ── the live path resolves display names like the pipeline does (2026-08-13) ─
def _kaya_resolver(tmp_path):
    from src.data.identity_resolver import SenderResolver

    path = tmp_path / "members.json"
    path.write_text(json.dumps({"members": [
        {"name": "Carnall", "aliases": ["carnall", "tomás"]},
        {"name": "Romano", "aliases": ["romano", "ricardo romano"]},
        {"name": "Ricky", "aliases": ["ricky", "ricardo alberto"]},
        {"name": "Frederico", "aliases": ["frederico", "fred"]},
    ]}), encoding="utf-8")
    return SenderResolver(path, {"fredericop167": "Frederico"})


def test_a_display_name_the_old_matcher_missed_now_resolves(tmp_path):
    """"Tomas Carnall": the first token is "tomas", the alias is "tomás", and
    the second token — an exact alias — was never tried. Five of a real
    member's messages were logged under a name that is not a member."""
    adapter, _ = make_adapter(tmp_path, contacts={})
    adapter.sender_resolver = _kaya_resolver(tmp_path)

    msg = parse_waha_message(
        dm_event("olá", sender="351999999999@c.us", name="Tomas Carnall"))

    assert adapter.resolve_speaker(msg) == "Carnall"


def test_a_sender_alias_override_reaches_the_live_path(tmp_path):
    adapter, _ = make_adapter(tmp_path, contacts={})
    adapter.sender_resolver = _kaya_resolver(tmp_path)

    msg = parse_waha_message(
        dm_event("olá", sender="351999999998@c.us", name="fredericop167"))

    assert adapter.resolve_speaker(msg) == "Frederico"


def test_an_ambiguous_first_name_is_not_guessed(tmp_path):
    """Two members answer to Ricardo. The bot keeps the display name rather
    than attributing the message to whichever it happened to match."""
    adapter, _ = make_adapter(tmp_path, contacts={})
    adapter.sender_resolver = _kaya_resolver(tmp_path)

    msg = parse_waha_message(
        dm_event("olá", sender="351999999997@c.us", name="Ricardo"))

    assert adapter.resolve_speaker(msg) == "Ricardo"


def test_the_phone_mapping_still_wins_over_the_display_name(tmp_path):
    """A number identifies a person; a display name does not. That is what
    settles the ambiguous cases in production."""
    adapter, _ = make_adapter(tmp_path, contacts={"351999999997@c.us": "Ricky"})
    adapter.sender_resolver = _kaya_resolver(tmp_path)

    msg = parse_waha_message(
        dm_event("olá", sender="351999999997@c.us", name="Ricardo"))

    assert adapter.resolve_speaker(msg) == "Ricky"


def test_without_a_resolver_the_previous_behaviour_holds(tmp_path):
    """Tests and older callers inject none."""
    adapter, _ = make_adapter(tmp_path, contacts={})
    adapter.sender_resolver = None
    adapter.member_aliases = {"gustavo": "Gustavo"}

    msg = parse_waha_message(dm_event("olá", sender="351999999996@c.us", name="Gustavo"))

    assert adapter.resolve_speaker(msg) == "Gustavo"


# ── mentions ─────────────────────────────────────────────────────────────────
# Only the bot's own mention was ever stripped; everybody else's reached the model
# as a bare number. "@257487651496102 tas fraquinho. Que desilusão" said nothing
# about Rafa to the model and nothing to the retriever's person filter, and the
# roast went to Manuel, who was not in the conversation. Both filed reports of the
# bot "referencing the wrong people" are this.
RAFA_LID = "257487651496102@lid"
GIL_LID = "34815055245324@lid"


def _mention_adapter(tmp_path):
    return make_adapter(tmp_path, contacts={
        RAFA_LID: "Rafa", GIL_LID: "Gil", "351911111111@c.us": "Alice"})


def test_a_mentioned_lid_becomes_a_name(tmp_path):
    adapter, _ = _mention_adapter(tmp_path)
    assert adapter._resolve_mentions("@257487651496102 tas fraquinho") == \
        "@Rafa tas fraquinho"


def test_every_mention_in_the_message_is_resolved(tmp_path):
    adapter, _ = _mention_adapter(tmp_path)
    assert adapter._resolve_mentions("@257487651496102 e @34815055245324 bora") == \
        "@Rafa e @Gil bora"


def test_an_unknown_lid_is_left_alone(tmp_path):
    """An unknown mention is still a mention. Deleting it would turn "@X e o @Y"
    into a sentence about one person."""
    adapter, _ = _mention_adapter(tmp_path)
    assert adapter._resolve_mentions("@999999999999999 quem és tu") == \
        "@999999999999999 quem és tu"


def test_ordinary_text_is_untouched(tmp_path):
    adapter, _ = _mention_adapter(tmp_path)
    for text in ("email@dominio.pt", "custa 50@ euros", "sem mentions nenhumas"):
        assert adapter._resolve_mentions(text) == text


def test_the_resolved_name_reaches_the_responder(tmp_path):
    adapter, _ = _mention_adapter(tmp_path)
    result = adapter.handle_event(
        group_event("@257487651496102 tas fraquinho", mention=True),
        system_prompt="")
    assert "@Rafa" in result["reply"]
    assert "257487651496102" not in result["reply"]


def test_the_bot_mention_is_still_stripped(tmp_path):
    """Resolution must not resurrect the bot's own @ token."""
    adapter, _ = _mention_adapter(tmp_path)
    text = adapter._resolve_mentions(adapter._strip_bot_mention(
        f"@{BOT_JID.split('@')[0]} olá"))
    assert text == "olá"


def test_the_logged_message_carries_names_not_numbers(tmp_path):
    """This log is embedded into ChromaDB. A message stored as
    "@257487651496102 tas fraquinho" can never be retrieved by a question
    about Rafa."""
    adapter, _ = _mention_adapter(tmp_path)
    logged = []
    adapter.log_messages = True
    adapter.message_log = type("L", (), {"append": lambda self, **kw: logged.append(kw)})()
    adapter.handle_event(group_event("@257487651496102 tas fraquinho", mention=True),
                         system_prompt="")
    assert logged and "@Rafa" in logged[0]["text"]


def test_mentioned_ids_are_untouched_so_the_reply_gate_still_works(tmp_path):
    """The gate keys off mentioned_ids, not the text."""
    adapter, _ = _mention_adapter(tmp_path)
    msg = parse_waha_message(group_event("@257487651496102 olá", mention=True))
    assert BOT_JID in msg.mentioned_ids
    assert adapter.should_respond(msg) is True


def test_a_known_member_learns_their_other_ids(tmp_path, capsys):
    """A member mapped by phone never reached the learning branch, so their @lid
    stayed unknown — and @lid is the only shape a mention comes in. Four of the
    group were in that state: identified as speakers, invisible when @-ed."""
    adapter, _ = make_adapter(tmp_path, contacts={"351913227550": "Gustavo"})
    event = group_event("olá", sender="64622145081581@lid", name="Gustavo Abreu",
                        mention=True)
    # Baileys carries the real phone alongside the @lid, under _data.key.
    event["payload"]["_data"] = {"key": {"participantAlt": "351913227550@c.us"}}
    msg = parse_waha_message(event)
    assert msg.sender_id == "64622145081581@lid" and msg.sender_phone == "351913227550"

    assert adapter.resolve_speaker(msg) == "Gustavo"
    assert adapter.contacts["64622145081581@lid"] == "Gustavo"
    assert adapter._resolve_mentions("@64622145081581 anda") == "@Gustavo anda"
    # Two ids for one member is normal and must not raise the mismatch alarm.
    assert "already" not in capsys.readouterr().out


def test_a_display_name_collision_still_warns(tmp_path, capsys):
    """The guard that matters is untouched: a GUESSED name that already belongs
    to another id is the failure that misattributed weeks of messages."""
    adapter, _ = make_adapter(
        tmp_path, contacts={"351913227550": "Gustavo"},
        member_aliases={"gustavo": "Gustavo"})
    msg = parse_waha_message(
        dm_event("olá", sender="351900000009@c.us", name="Gustavo"))

    assert adapter.resolve_speaker(msg) == "Gustavo"
    assert "already" in capsys.readouterr().out


def test_the_bots_own_mention_resolves_from_a_bare_number(tmp_path):
    """bot_jids holds "<lid>@lid"; a mention in the body is the bare number. The
    two were compared directly, so the bot's own @ stayed a raw number in the
    message log — the log that becomes searchable memory."""
    adapter, _ = _mention_adapter(tmp_path)
    adapter.bot_jids = {"237065786642635@lid", "48453977310@c.us"}

    assert adapter._name_for_jid("237065786642635") == "Kaya Bot"
    assert adapter._name_for_jid("237065786642635@lid") == "Kaya Bot"
    assert adapter._name_for_jid("48453977310") == "Kaya Bot"
    assert adapter._resolve_mentions("@237065786642635 quantas vezes?") == \
        "@Kaya Bot quantas vezes?"


def test_a_member_still_wins_over_the_bot_check(tmp_path):
    """The bot short-circuit must not shadow a real contact."""
    adapter, _ = _mention_adapter(tmp_path)
    adapter.bot_jids = {"237065786642635@lid"}
    assert adapter._name_for_jid("257487651496102") == "Rafa"


# ── the bot reads the room (2026-08-17) ──────────────────────────────────────
# In a group it replies on a mention or a reply, and it used to record only the
# turns it answered. One live morning that was 46 messages in the room and 29 in
# the prompt: it never saw "Bruh nunca vi programador tão fraco" or "Bruv is
# hallucinating hard", which is why a "toma aí" between them came back as an
# unrelated stock insult. The durable log had all of it; it just never reached
# the model.

def test_chatter_the_bot_was_not_addressed_in_reaches_the_next_turn(tmp_path):
    adapter, seen = _capture_responder(tmp_path, history_turns=20)
    adapter.handle_event(group_event("Bruh nunca vi programador tão fraco"),
                         system_prompt="")
    adapter.handle_event(group_event("toma aí", mention=True), system_prompt="")
    assert any("programador tão fraco" in line for line in seen["recent"]), (
        "a message the bot was not addressed in must still be in the context of "
        "the next one it answers")


def test_the_answered_message_is_not_also_handed_back_as_history(tmp_path):
    """It is appended before the reply gate, so it would otherwise arrive twice —
    once as the question and once as something already said."""
    adapter, seen = _capture_responder(tmp_path, history_turns=20)
    adapter.handle_event(group_event("toma aí", mention=True), system_prompt="")
    assert not any("toma aí" in line for line in seen["recent"])


def test_the_asker_is_stored_exactly_once(tmp_path):
    adapter, _ = _capture_responder(tmp_path, history_turns=20)
    adapter.handle_event(group_event("olá bot", mention=True), system_prompt="")
    lines = adapter.session_store.recent(GROUP, None)
    assert sum(1 for line in lines if "olá bot" in line) == 1


def test_chatter_is_named_not_numbered(tmp_path):
    """This window is also what the rolling summary is built from, and a line
    stored as a bare @lid says nothing to the model or to the person filter."""
    adapter, seen = _capture_responder(tmp_path, history_turns=20)
    adapter.contacts[ALICE] = "Alice"
    other = "351922222222@c.us"
    event = group_event("@" + ALICE.split("@")[0] + " tas fraquinho", sender=other,
                        name="Outro")
    adapter.handle_event(event, system_prompt="")
    adapter.handle_event(group_event("e então", mention=True), system_prompt="")
    joined = "\n".join(seen["recent"])
    assert "Alice" in joined and "tas fraquinho" in joined


def test_a_slash_command_never_enters_the_verbatim_window(tmp_path):
    """Same rule as the durable log: a week of bug reports must not become
    things "the group said" — and this window feeds the rolling summary."""
    adapter, seen = _capture_responder(tmp_path, history_turns=20)
    adapter.handle_event(group_event("/bug não respondeu ao meu áudio"),
                         system_prompt="")
    adapter.handle_event(group_event("e então", mention=True), system_prompt="")
    assert not any("não respondeu ao meu áudio" in line for line in seen["recent"])


def test_the_bots_own_messages_are_not_read_back_as_chatter(tmp_path):
    adapter, seen = _capture_responder(tmp_path, history_turns=20)
    event = group_event("isto sou eu")
    event["payload"]["fromMe"] = True
    adapter.handle_event(event, system_prompt="")
    adapter.handle_event(group_event("e então", mention=True), system_prompt="")
    assert not any("isto sou eu" in line for line in seen["recent"])


# ── documents ────────────────────────────────────────────────────────────────
def _document_event(filename="labour.pdf", mimetype="application/pdf", body="",
                    timestamp=1700000000, group=True):
    event = group_event(body, mention=True) if group else dm_event(body)
    event["payload"]["media"] = {"url": "http://waha:3000/f.pdf",
                                 "mimetype": mimetype, "filename": filename}
    # MessageLog.read() only yields records newer than its cutoff, so a message
    # with no timestamp is written and never read back.
    event["payload"]["timestamp"] = timestamp
    return event


def test_a_pdf_is_not_sent_to_the_transcriber(tmp_path):
    """The live bug: a PDF was not an image, so it went to Whisper and vanished.

    Four papers shared in the group on 2026-09-05 left no trace on disk because
    the transcribe branch was gated on "not an image" rather than on "audio".
    """
    transcribed = []
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="ok", mode="banter"))
    adapter.transcribe = lambda url, mimetype: transcribed.append(url) or "nope"
    adapter.ingest_document = lambda msg: "labour.pdf, 51 páginas — sobre sindicatos"

    adapter.handle_event(_document_event(), system_prompt="")

    assert transcribed == [], "a PDF must never reach Whisper"


def _logging_adapter(tmp_path, synopsis):
    """An adapter whose message log can be read back, as the photo test does."""
    from src.chat.memory import ChatPreferences
    from src.data.message_log import MessageLog

    log = MessageLog(base_dir=str(tmp_path / "log"))
    adapter = WhatsAppAdapter(
        responder=lambda message, speaker, recent_lines, **kw: "ok",
        waha_client=MockWahaClient(echo=False),
        config={"whatsapp": {"bot_jid": BOT_JID, "send_seen": False,
                             "log_messages": True, "shared_chats": [GROUP]}},
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "sessions")),
        prefs=ChatPreferences(base_dir=str(tmp_path / "prefs")),
        message_log=log,
        ingest_document=synopsis,
    )
    return adapter, log


def test_a_document_becomes_text_the_group_can_search(tmp_path):
    """This is what makes "aquele doc que o Bana mandou" findable later."""
    adapter, log = _logging_adapter(
        tmp_path, lambda msg: "labour.pdf, 51 páginas — sobre sindicatos")

    adapter.handle_event(_document_event(), system_prompt="")

    logged = [m["text"] for m in log.read("shared")]
    assert any("[Documento: labour.pdf" in text for text in logged), logged


def test_a_document_caption_is_kept(tmp_path):
    adapter, log = _logging_adapter(
        tmp_path, lambda msg: "labour.pdf, 51 páginas — sobre sindicatos")

    adapter.handle_event(
        _document_event(body="argue with the paper boys"), system_prompt="")

    logged = "\n".join(m["text"] for m in log.read("shared"))
    assert "argue with the paper boys" in logged
    assert "[Documento:" in logged


def test_an_unreadable_document_is_not_announced(tmp_path):
    """A scanned PDF must not be logged as one the bot has read."""
    adapter, log = _logging_adapter(tmp_path, lambda msg: "")

    adapter.handle_event(_document_event(), system_prompt="")

    logged = "\n".join(m["text"] for m in log.read("shared"))
    assert "[Documento:" not in logged


def test_a_failing_document_reader_does_not_break_the_message(tmp_path):
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="ok", mode="banter"))

    def boom(msg):
        raise RuntimeError("pypdf exploded")

    adapter.ingest_document = boom

    # Must not raise.
    adapter.handle_event(
        _document_event(body="olhem este doc"), system_prompt="")


def test_the_document_filename_survives_parsing():
    msg = parse_waha_message(_document_event(filename="Página 51.pdf"))
    assert msg.media_filename == "Página 51.pdf"


def test_a_voice_note_is_still_transcribed(tmp_path):
    """The regression guard for the fix itself."""
    transcribed = []
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="ok", mode="banter"))
    adapter.transcribe = lambda url, mimetype: (transcribed.append(url)
                                                or "disse isto em voz alta")

    event = dm_event("")
    event["payload"]["media"] = {"url": "http://waha/f.oga",
                                 "mimetype": "audio/ogg; codecs=opus"}
    adapter.handle_event(event, system_prompt="")

    assert transcribed, "voice notes must still be transcribed"


def test_resolve_speaker_is_idempotent(tmp_path):
    """`whatsapp_server._ingest_document` calls it before the reply path does.

    A document's chunks store who shared it, and that has to be the canonical
    member name — "enviado por Tomas Carnall" is invisible to a person filter
    matching "Carnall". Resolving early is only safe if resolving twice gives the
    same answer and does not corrupt the learned contact map.
    """
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="ok", mode="banter"))
    msg = parse_waha_message(_document_event())

    first = adapter.resolve_speaker(msg)
    second = adapter.resolve_speaker(msg)

    assert first == second
    assert first, "a speaker must always resolve to something usable"


def test_a_duplicate_delivery_does_no_work_twice(tmp_path):
    """WAHA sends every event to the webhook TWICE.

    Confirmed in WAHA's own logs: the same `event.id` dispatched by two
    WebhookPlugin instances, because the hook is registered both globally
    (WHATSAPP_HOOK_URL) and in the session config. `should_respond` already
    guarded the reply, which is why nobody ever saw two answers — but it is
    checked after the media branches, so the expensive work ran twice.
    """
    reads = []
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="ok", mode="banter"))
    adapter.ingest_document = lambda msg: (reads.append(msg.media_filename)
                                           or "paper.pdf, 5 páginas — uma sinopse")

    event = _document_event()
    adapter.handle_event(event, system_prompt="")
    second = adapter.handle_event(event, system_prompt="")

    assert reads == ["labour.pdf"], "the document was read twice"
    assert second is None, "the duplicate should be dropped, not answered"


def test_a_duplicate_photo_is_described_once(tmp_path):
    """Same bug, and it was costing a vision call on every photo in the group."""
    described = []
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="ok", mode="banter"))
    adapter.describe_image = lambda url, mimetype: (described.append(url)
                                                    or "dois homens num barco")

    event = photo_event()
    adapter.handle_event(event, system_prompt="")
    adapter.handle_event(event, system_prompt="")

    assert len(described) == 1, "the photo was described twice"


def test_a_duplicate_voice_note_is_transcribed_once(tmp_path):
    transcribed = []
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="ok", mode="banter"))
    adapter.transcribe = lambda url, mimetype: (transcribed.append(url) or "disse isto")

    event = dm_event("")
    event["payload"]["media"] = {"url": "http://waha/f.oga",
                                 "mimetype": "audio/ogg; codecs=opus"}
    adapter.handle_event(event, system_prompt="")
    adapter.handle_event(event, system_prompt="")

    assert len(transcribed) == 1


def test_distinct_messages_are_still_both_processed(tmp_path):
    """The guard is on the message id, not on the content."""
    replies = []
    adapter = make_routed_adapter(tmp_path, RoutedReply(text="ok", mode="banter"))

    for index in range(2):
        result = adapter.handle_event(
            dm_event("olá", message_id=f"MSG{index}"), system_prompt="")
        replies.append(result)

    assert all(r is not None for r in replies), "two real messages must both answer"
