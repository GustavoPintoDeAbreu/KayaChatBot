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

@pytest.fixture(autouse=True)
def _fresh_image_queue():
    """The image queue is process-wide, so one test's leftovers change the next
    test's behaviour — a full queue from one case silently refused another's job."""
    from src.chat import imagegen

    previous = imagegen._default_queue
    imagegen._default_queue = imagegen.ImageQueue()
    yield
    imagegen._default_queue = previous


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
    assert tts.is_available(config) is True
    assert tts._voice_paths(config)["pt"] == tts.DEFAULT_VOICES["pt"]


# ── image generation / editing (Phase 5) ─────────────────────────────────────
# Making a picture takes minutes, so the webhook must return immediately and the
# image arrive later. These pin the acknowledgement, the choice of subject, and
# the refusals — a bot that promises a picture it will never send is worse than
# one that says no.
def make_image_adapter(tmp_path, reply, imagegen_result=b"PNGDATA", **overrides):
    from src.chat.memory import ChatPreferences

    config = {
        "whatsapp": {"bot_jid": BOT_JID, "send_seen": False,
                     "shared_chats": [GROUP], **overrides},
        "chat": {"imagegen": {"enabled": True, "allowed_scopes": ["shared"]}},
    }
    calls = []

    def fake_imagegen(mode, prompt, image_path=None):
        calls.append({"mode": mode, "prompt": prompt, "image_path": image_path})
        return imagegen_result

    adapter = WhatsAppAdapter(
        responder=lambda message, speaker, recent_lines, **kw: reply,
        waha_client=MockWahaClient(echo=False),
        config=config,
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "sessions")),
        prefs=ChatPreferences(base_dir=str(tmp_path / "prefs")),
        image_generate=fake_imagegen,
        fetch_media=lambda url, mimetype: str(tmp_path / "photo.jpg"),
    )
    adapter.imagegen_calls = calls
    return adapter


def _wait_for_image(adapter, timeout=5.0):
    """The image is produced on a background thread; wait for it to land."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if any("image_bytes" in s for s in adapter.waha_client.sent):
            return True
        time.sleep(0.02)
    return False


def image_group_event(text, media_url="", mimetype=""):
    event = group_event(text, mention=True)
    if media_url:
        event["payload"]["media"] = {"url": media_url, "mimetype": mimetype}
    return event


def test_image_request_acknowledges_immediately(tmp_path):
    """Five minutes of silence reads as a broken bot."""
    adapter = make_image_adapter(tmp_path, RoutedReply(command="image", mode="factual"))

    result = adapter.handle_event(image_group_event("faz uma imagem de um gato astronauta"),
                                  system_prompt="")

    assert result["command"] == "image"
    assert result["reply"], "the request must be acknowledged before the work starts"
    assert _wait_for_image(adapter)


def test_request_without_a_photo_generates_from_text(tmp_path):
    adapter = make_image_adapter(tmp_path, RoutedReply(command="image", mode="factual"))

    adapter.handle_event(image_group_event("faz uma imagem de um gato astronauta"),
                         system_prompt="")

    assert _wait_for_image(adapter)
    assert adapter.imagegen_calls[0]["mode"] == "generate"
    assert adapter.imagegen_calls[0]["image_path"] is None


def test_attached_photo_is_edited(tmp_path):
    adapter = make_image_adapter(tmp_path, RoutedReply(command="image", mode="factual"))

    adapter.handle_event(
        image_group_event("põe-lhe uma coroa", media_url="http://waha:3000/f.jpg",
                          mimetype="image/jpeg"),
        system_prompt="")

    assert _wait_for_image(adapter)
    assert adapter.imagegen_calls[0]["mode"] == "edit"
    assert adapter.imagegen_calls[0]["image_path"]


def test_the_last_photo_in_the_chat_is_the_implicit_subject(tmp_path):
    """"põe-lhe uma coroa" right after someone posts a picture must edit it."""
    adapter = make_image_adapter(tmp_path, RoutedReply(command="image", mode="factual"))

    # A photo posted to the group without addressing the bot: not replied to, but
    # seen — and it is what the next request refers to.
    adapter.handle_event(image_group_event("olhem esta", media_url="http://waha:3000/f.jpg",
                                           mimetype="image/jpeg"),
                         system_prompt="")
    adapter.handle_event(image_group_event("põe-lhe uma coroa"), system_prompt="")

    assert _wait_for_image(adapter)
    assert adapter.imagegen_calls[-1]["mode"] == "edit"


def test_a_dm_may_not_ask_for_an_edit(tmp_path):
    """Editing puts a real member's face in an invented scene; the group opted in
    to having the bot in the room, a random DM did not."""
    adapter = make_image_adapter(tmp_path, RoutedReply(command="image", mode="factual"))

    result = adapter.handle_event(dm_event("põe o Rafa de rei"), system_prompt="")

    assert result["image"] == "not_allowed"
    assert not adapter.imagegen_calls
    assert "só faço imagens no grupo" in adapter.waha_client.sent[-1]["text"].lower()


def test_failed_generation_says_so(tmp_path):
    """Silence after a promise is the failure mode to avoid."""
    adapter = make_image_adapter(tmp_path, RoutedReply(command="image", mode="factual"),
                                 imagegen_result=None)

    adapter.handle_event(image_group_event("faz uma imagem"), system_prompt="")

    # Wait for the QUEUE to drain, not for a message count. The job runs on a
    # background thread, so counting sends races with the scheduler — this failed
    # once on a box that was busy rendering a bake-off, and only then.
    import time

    from src.chat import imagegen

    deadline = time.time() + 10.0
    while time.time() < deadline and (
            imagegen.get_queue().depth > 0 or len(adapter.waha_client.sent) < 2):
        time.sleep(0.02)
    assert "não consegui" in adapter.waha_client.sent[-1]["text"].lower()


def test_without_a_worker_the_bot_declines(tmp_path):
    """image_generate=None (feature off) must decline, not hang."""
    from src.chat.memory import ChatPreferences

    adapter = WhatsAppAdapter(
        responder=lambda message, speaker, recent_lines, **kw: RoutedReply(
            command="image", mode="factual"),
        waha_client=MockWahaClient(echo=False),
        config={"whatsapp": {"bot_jid": BOT_JID, "send_seen": False,
                             "shared_chats": [GROUP]},
                "chat": {"imagegen": {"enabled": True, "allowed_scopes": ["shared"]}}},
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "sessions")),
        prefs=ChatPreferences(base_dir=str(tmp_path / "prefs")),
    )

    result = adapter.handle_event(image_group_event("faz uma imagem"), system_prompt="")

    assert result["image"] == "not_allowed"


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


# ── which picture an image request is about ──────────────────────────────────
# The last photo in a chat is a good implicit subject for "põe-lhe uma coroa" and
# a terrible one for "faz uma imagem de um gato astronauta". Treating every
# request as an edit meant any image request after any photo silently edited
# somebody's holiday snap instead of drawing what was asked for.
def test_a_request_describing_something_new_generates():
    from src.chat.whatsapp_adapter import refers_to_existing_image

    for text in ("faz uma imagem de um gato astronauta",
                 "gera uma imagem de um cão a conduzir",
                 "desenha um robot gigante",
                 "make me a picture of a dragon"):
        assert refers_to_existing_image(text) is False, text


def test_a_request_pointing_at_a_photo_edits():
    from src.chat.whatsapp_adapter import refers_to_existing_image

    for text in ("põe-lhe uma coroa",
                 "edita esta foto e mete-lhe um chapéu",
                 "põe este gajo de rei medieval",
                 "nesta foto mete-lhe óculos",
                 "put a crown on him"):
        assert refers_to_existing_image(text) is True, text


def test_last_photo_is_only_used_when_the_request_refers_to_it(tmp_path):
    """End to end: a generate-style request must not consume the last photo."""
    adapter = make_image_adapter(tmp_path, RoutedReply(command="image", mode="factual"))

    adapter.handle_event(image_group_event("olhem esta", media_url="http://waha:3000/f.jpg",
                                           mimetype="image/jpeg"), system_prompt="")
    adapter.handle_event(image_group_event("faz uma imagem de um gato astronauta"),
                         system_prompt="")

    assert _wait_for_image(adapter)
    assert adapter.imagegen_calls[-1]["mode"] == "generate"
    assert adapter.imagegen_calls[-1]["image_path"] is None


def test_an_attached_photo_still_edits_regardless_of_wording(tmp_path):
    """An attachment is unambiguous and never consults the wording."""
    adapter = make_image_adapter(tmp_path, RoutedReply(command="image", mode="factual"))

    adapter.handle_event(
        image_group_event("faz uma imagem gira", media_url="http://waha:3000/f.jpg",
                          mimetype="image/jpeg"),
        system_prompt="")

    assert _wait_for_image(adapter)
    assert adapter.imagegen_calls[-1]["mode"] == "edit"


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


# ── the image queue ──────────────────────────────────────────────────────────
# One GPU means one render at a time, but refusing the second request loses it:
# a long simulator run had two edits asked for, told "estou ocupado", and never
# made. They queue now. The queue is separate from the text path — generation
# runs on the other card — so conversation continues at full speed meanwhile.
def test_a_second_image_request_is_queued_not_refused(tmp_path):
    import threading

    from src.chat import imagegen

    release = threading.Event()
    adapter = make_image_adapter(tmp_path, RoutedReply(command="image", mode="factual"))

    def slow_imagegen(mode, prompt, image_path=None):
        release.wait(timeout=5)
        return b"PNGDATA"

    adapter.image_generate = slow_imagegen
    imagegen._default_queue = imagegen.ImageQueue()

    first = adapter.handle_event(image_group_event("faz uma imagem de um gato"),
                                 system_prompt="")
    second = adapter.handle_event(image_group_event("faz uma imagem de um cão"),
                                  system_prompt="")

    assert first["queue_position"] == 1
    assert second["queue_position"] == 2, "the second request must wait, not be dropped"
    assert "fila" in second["reply"].lower()
    # position 2 means ONE job ahead, not two
    assert "1 à frente" in second["reply"]
    release.set()


def test_a_full_queue_declines_rather_than_promising(tmp_path):
    import threading

    from src.chat import imagegen

    adapter = make_image_adapter(tmp_path, RoutedReply(command="image", mode="factual"))
    imagegen._default_queue = imagegen.ImageQueue(maxsize=1)
    # The job must still be occupying the slot when the second request arrives,
    # otherwise the first finishes instantly and the queue is empty again.
    release = threading.Event()
    adapter.image_generate = lambda mode, prompt, image_path=None: (
        release.wait(timeout=5) and b"PNG")

    adapter.handle_event(image_group_event("faz uma imagem 1"), system_prompt="")
    result = adapter.handle_event(image_group_event("faz uma imagem 2"), system_prompt="")
    release.set()

    assert result["image"] == "queue_full"


def test_a_pending_image_is_visible_in_the_conversation(tmp_path):
    """Asked "então e a foto?" two turns later, the bot must know one is coming."""
    from src.chat import imagegen

    adapter = make_image_adapter(tmp_path, RoutedReply(command="image", mode="factual"))
    imagegen._default_queue = imagegen.ImageQueue()

    adapter.handle_event(image_group_event("faz uma imagem de um gato astronauta"),
                         system_prompt="")

    history = adapter.session_store.recent(GROUP, 10)
    assert any("a preparar uma imagem" in line for line in history)
    assert any("gato astronauta" in line for line in history)


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


def test_image_acknowledgements_carry_no_dashes(tmp_path):
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
    # 6 lines is 3 answered turns, so the window covers the last THREE inbound
    # messages — the 10th, 9th and 8th. It starts at the 8th (base + 7).
    from datetime import datetime, timezone

    expected = datetime.fromtimestamp(base + 7, tz=timezone.utc).replace(
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

    Each answered turn writes TWO session lines (the asker's and the bot's), so
    `history_turns` lines is half that many inbound messages. Measuring the
    exclusion window in lines pushes its start too far back, and retrieval then
    drops chunks covering messages that are NOT held verbatim — a hole, not a
    duplicate, and silent.
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
    oldest_inbound_ts = base + 12 - len(inbound)
    earliest_allowed = datetime.fromtimestamp(
        oldest_inbound_ts, tz=timezone.utc).replace(tzinfo=None).isoformat()
    assert start >= earliest_allowed, (
        f"window starts at {start}, earlier than the oldest message actually in "
        f"the prompt ({earliest_allowed}) — chunks in between would be excluded "
        f"from retrieval without being carried verbatim")


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
