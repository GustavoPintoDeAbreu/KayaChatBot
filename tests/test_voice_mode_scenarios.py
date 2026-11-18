"""Switching between voice and text replies, the way Gustavo uses it (2026-11-18).

Any message, typed or spoken, may end with how he wants the answer. "Answer in audio" sticks
until "answer in text"; the instruction alone is confirmed and nothing else is answered; a
question in front of it is answered, in the new mode. In a group the mode is per person.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.memory import ChatPreferences, KeyedSessionMemory
from src.chat.waha_client import MockWahaClient
from src.chat.whatsapp_adapter import WhatsAppAdapter

from tests.test_whatsapp_adapter import ALICE, BOT_JID, GROUP, RoutedReply, dm_event, group_event

BOB = "351922222222@c.us"


class Engine:
    """The responder: records what it was asked, answers with a routed reply."""

    def __init__(self, route_command=None):
        self.asked = []
        self.route_command = route_command

    def __call__(self, message, speaker, recent_lines, **kw):
        self.asked.append(message)
        if self.route_command:
            return RoutedReply(command=self.route_command)
        return RoutedReply(text=f"answer to: {message}", mode="general")


def rig(tmp_path, engine=None):
    engine = engine or Engine()
    adapter = WhatsAppAdapter(
        responder=engine, waha_client=MockWahaClient(echo=False),
        config={"whatsapp": {"bot_jid": BOT_JID, "send_seen": False,
                             "contacts": {ALICE: "Alice", BOB: "Bob"},
                             "group": {"respond_on_mention": True, "respond_on_reply": True}}},
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "sessions")),
        prefs=ChatPreferences(base_dir=str(tmp_path / "prefs")),
    )
    adapter.audio_reply_enabled = True
    adapter.tts_synthesize = lambda text: b"OGG"
    adapter.transcribe = lambda url, mime: adapter._next_transcript
    return adapter, engine


def voice_note(adapter, transcript, **kw):
    adapter._next_transcript = transcript
    event = dm_event("", **kw)
    event["payload"]["media"] = {"url": "http://waha/f.oga", "mimetype": "audio/ogg"}
    return adapter.handle_event(event, system_prompt="")


def last(adapter):
    sent = adapter.waha_client.sent[-1]
    return ("voice" if "voice_bytes" in sent else "text"), sent.get("text", "")


def test_answer_in_text_alone_mid_voice_conversation_only_confirms(tmp_path):
    adapter, engine = rig(tmp_path)
    adapter.prefs.set_output_mode(ALICE, "audio")
    adapter.handle_event(dm_event("who won the derby yesterday?"), system_prompt="")
    assert last(adapter)[0] == "voice"
    sends, asks = len(adapter.waha_client.sent), len(engine.asked)

    result = adapter.handle_event(dm_event("Answer in text."), system_prompt="")

    assert result["command"] == "text"
    assert len(engine.asked) == asks, "the switch must not re-answer the conversation"
    assert len(adapter.waha_client.sent) == sends + 1
    assert last(adapter) == ("text", "Got it, text only from now on.")
    adapter.handle_event(dm_event("and who scored?"), system_prompt="")
    assert last(adapter)[0] == "text"


def test_answer_in_text_said_in_a_voice_note_replies_in_text(tmp_path):
    adapter, engine = rig(tmp_path)
    adapter.prefs.set_output_mode(ALICE, "audio")

    voice_note(adapter, "Answer in text.")

    assert engine.asked == []
    assert last(adapter) == ("text", "Got it, text only from now on.")
    assert adapter.output_mode(ALICE) == "text"


def test_a_question_ending_in_answer_in_audio_is_answered_by_voice_and_it_sticks(tmp_path):
    adapter, engine = rig(tmp_path)

    adapter.handle_event(dm_event("who's coming to dinner tonight? answer in audio"), system_prompt="")

    assert engine.asked == ["who's coming to dinner tonight?"]
    assert last(adapter)[0] == "voice"
    assert adapter.output_mode(ALICE) == "audio"
    adapter.handle_event(dm_event("and on Saturday?"), system_prompt="")
    assert last(adapter)[0] == "voice"


def test_the_same_in_a_voice_note_and_in_portuguese(tmp_path):
    adapter, engine = rig(tmp_path)

    voice_note(adapter, "quem vem ao jantar logo? responde em áudio")

    assert engine.asked == ["quem vem ao jantar logo?"]
    assert last(adapter)[0] == "voice"
    voice_note(adapter, "responde em texto")
    assert last(adapter) == ("text", "Feito, volto a responder por texto.")


def test_a_voice_note_in_text_mode_is_answered_in_text(tmp_path):
    adapter, engine = rig(tmp_path)

    voice_note(adapter, "what's the plan for Saturday?")

    assert last(adapter)[0] == "text"


def test_answer_in_audio_alone_confirms_by_voice_and_answers_nothing(tmp_path):
    adapter, engine = rig(tmp_path)

    adapter.handle_event(dm_event("answer in audio"), system_prompt="")

    assert engine.asked == []
    assert last(adapter)[0] == "voice"
    assert adapter.output_mode(ALICE) == "audio"


def test_just_this_one_speaks_once_and_keeps_text(tmp_path):
    adapter, engine = rig(tmp_path)

    adapter.handle_event(dm_event("explain inflation, answer in audio, just this one"), system_prompt="")

    assert engine.asked == ["explain inflation"]
    assert last(adapter)[0] == "voice"
    assert adapter.output_mode(ALICE) == "text"
    adapter.handle_event(dm_event("and deflation?"), system_prompt="")
    assert last(adapter)[0] == "text"


def test_switching_twice_in_a_row_is_idempotent(tmp_path):
    adapter, engine = rig(tmp_path)
    adapter.handle_event(dm_event("answer in text"), system_prompt="")
    adapter.handle_event(dm_event("answer in text"), system_prompt="")
    assert engine.asked == [] and adapter.output_mode(ALICE) == "text"
    assert [s.get("text") for s in adapter.waha_client.sent] == ["Got it, text only from now on."] * 2


def test_in_a_group_the_mode_is_per_person(tmp_path):
    adapter, engine = rig(tmp_path)

    adapter.handle_event(group_event("@bot answer in audio", mention=True), system_prompt="")
    adapter.handle_event(group_event("@bot what's the score?", sender=BOB, name="Bob", mention=True),
                         system_prompt="")
    assert last(adapter)[0] == "text", "Alice's switch must not give Bob voice notes"
    adapter.handle_event(group_event("@bot and the next game?", mention=True), system_prompt="")
    assert last(adapter)[0] == "voice"


def test_talk_about_audio_is_answered_normally(tmp_path):
    adapter, engine = rig(tmp_path)

    adapter.handle_event(dm_event("o Rafa respondeu em áudio"), system_prompt="")

    assert engine.asked == ["o Rafa respondeu em áudio"]
    assert adapter.output_mode(ALICE) == "text"


def test_the_routers_free_phrasing_still_switches_per_person(tmp_path):
    """Phrasings the trailing check doesn't know still reach the router's CMD_AUDIO."""
    adapter, engine = rig(tmp_path, Engine(route_command="audio"))

    adapter.handle_event(group_event("@bot daqui para a frente quero ouvir-te", mention=True),
                         system_prompt="")

    assert adapter.output_mode(adapter._mode_key(type("M", (), {
        "is_group": True, "chat_id": GROUP, "sender_id": ALICE})())) == "audio"
    assert adapter.output_mode(GROUP) == "text"


def test_whisper_picks_portuguese_or_english_never_a_third():
    """Forced to "pt", an English voice note came out translated into Portuguese (2026-11-18)."""
    from src.chat.stt import pick_language
    assert pick_language([("es", 0.6), ("pt", 0.3), ("en", 0.1)], ["pt", "en"]) == "pt"
    assert pick_language([("en", 0.06), ("pt", 0.01), ("nl", 0.4)], ["pt", "en"]) == "en"
    assert pick_language([], ["pt", "en"]) == "pt"
    assert pick_language([("en", 0.9)], []) is None


def test_the_config_transcribes_portuguese_and_english_with_mode_hotwords():
    from src.config_loader import load_config
    audio = load_config(str(Path(__file__).parent.parent / "config.yaml"))["chat"]["audio"]
    assert audio["language"] == "auto" and audio["languages"] == ["pt", "en"]
    assert "responde em texto" in audio["hotwords"] and "answer in audio" in audio["hotwords"]
