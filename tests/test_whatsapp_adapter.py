"""Unit tests for the WhatsApp bridge — routing, gating, speaker, history.

No GPU/model/network: the engine is replaced by a stub ``responder`` and WAHA by
``MockWahaClient``, so this exercises the full inbound→reply logic the same way
``scripts/whatsapp_simulator.py`` does, against a temp session dir.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.memory import KeyedSessionMemory
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


def dm_event(text, sender=ALICE, name="Alice", from_me=False):
    return {
        "event": "message",
        "payload": {"id": "x", "from": sender, "body": text, "notifyName": name, "fromMe": from_me},
    }


def group_event(text, sender=ALICE, name="Alice", mention=False, reply=False):
    payload = {
        "id": "g1",
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
    result = adapter.handle_event(group_event("@bot quem é o Rui?", mention=True))
    assert result is not None
    assert len(client.sent) == 1
    assert client.sent[0]["reply_to"] == "g1"  # group replies quote the asker


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
            "id": f"false_{USER_LID}_ABC",
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


def noweb_group(text, mention_lid=None, reply_to_lid=None):
    """A NOWEB-shaped group message with nested contextInfo."""
    ext = {"text": text, "contextInfo": {}}
    if mention_lid:
        ext["contextInfo"]["mentionedJid"] = [mention_lid]
    if reply_to_lid:
        ext["contextInfo"]["participant"] = reply_to_lid
    return {
        "event": "message",
        "me": {"id": BOT_JID, "lid": BOT_LID},
        "payload": {
            "id": f"false_{GROUP}_XYZ_{USER_LID}",
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

    def __init__(self, text="", command=None, mode="banter"):
        self.text = text
        self.route = type("Route", (), {"mode": mode, "command": command})()


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
    assert tts.is_available(config) is True
    assert tts._voice_paths(config)["pt"] == tts.DEFAULT_VOICES["pt"]
