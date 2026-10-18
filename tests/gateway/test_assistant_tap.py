"""The assistant tap: the owner's DM voice notes and plain text go to the lab assistant."""
import base64
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.app import Gateway, GatewaySettings
from src.gateway.assistant_tap import ASLEEP_TEXT, AssistantTap
from src.gateway.idea_tap import IdeaTap
from src.gateway.monitor import PcMonitor, PcState
from tests.gateway.test_idea_tap import ALICE, BOT, GROUP, OWNER, Inbox, _message


@pytest.fixture
def rig(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "whatsapp": {"group": {"respond_on_mention": True, "respond_on_reply": True},
                     "whitelist": {"enabled": True, "allowed": ["351911111111", "351922222222"]}},
        "power": {"timezone": "Europe/Lisbon", "wake_time": "07:00", "wol_lead_minutes": 5,
                  "shutdown": {day: "23:00" for day in ("sun", "mon", "tue", "wed", "thu", "fri", "sat")}},
    }))
    settings = GatewaySettings.from_env({
        "GATEWAY_DATA_DIR": str(tmp_path / "data"), "GATEWAY_CONFIG": str(config),
        "GATEWAY_WHITELIST": str(tmp_path / "none.json"), "KAYA_RELAY_TOKEN": "relay",
        "KAYA_WHATSAPP_WEBHOOK_TOKEN": "hook",
    })
    monitor = PcMonitor(settings.pc_url, probe=lambda: PcState.ONLINE)
    monitor.observe(PcState.ONLINE)
    ideas, assistant, sent = Inbox(), Inbox(), []
    env = {"IDEA_INBOX_URL": "http://idea-inbox:8000", "IDEA_OWNER_NUMBERS": "+351922222222",
           "ASSISTANT_INBOX_URL": "http://assistant-inbox:8000/", "ASSISTANT_INBOX_TOKEN": "atok"}
    idea_tap, tap = IdeaTap.from_env(env), AssistantTap.from_env(env)
    idea_tap.post, tap.post = ideas, assistant

    async def fetch(url):
        return b"OGGDATA", "audio/ogg; codecs=opus"

    gateway = Gateway(settings, monitor=monitor, idea_tap=idea_tap, assistant_tap=tap, fetch_media=fetch,
                      send_text=lambda chat, text, reply_to=None: sent.append((chat, text, reply_to)))
    return gateway, ideas, assistant, sent, monitor


def _post(gateway, event):
    return TestClient(gateway.internal_app).post("/waha/webhook", json=event,
                                                 headers={"X-Webhook-Token": "hook"}).json()


def _voice(message_id, chat=OWNER):
    event = _message(message_id, "", chat=chat)
    event["payload"].update(hasMedia=True, media={"url": "http://waha:3000/api/files/v.oga",
                                                  "mimetype": "audio/ogg; codecs=opus"})
    return event


def test_a_voice_note_goes_to_the_assistant_with_its_audio(rig):
    gateway, _, assistant, sent, _ = rig
    assert _post(gateway, _voice("v1")) == {"assistant": "forwarded"}
    assert gateway.journal.pending_count() == 0
    url, body, headers = assistant.received[0]
    assert url == "http://assistant-inbox:8000/ingest" and headers == {"X-Assistant-Token": "atok"}
    assert base64.b64decode(body["audio_b64"]) == b"OGGDATA" and body["audio_mime"].startswith("audio/ogg")
    assert body["event"]["payload"]["id"] == "v1" and body["sender"] == "351922222222"
    assert sent == []


def test_plain_text_goes_but_commands_and_replies_to_kaya_stay(rig):
    gateway, ideas, assistant, _, _ = rig
    assert _post(gateway, _message("t1", "is the PC up?")) == {"assistant": "forwarded"}
    assert _post(gateway, _message("t2", "yes", quoted="🤖 → prices: follow item\nReply *yes* or *no*.")) == \
        {"assistant": "forwarded"}
    assert "assistant" not in _post(gateway, _message("t3", "/start"))
    assert "assistant" not in _post(gateway, _message("t4", "e depois?", quoted="Claro! O jantar é às 8."))
    assert _post(gateway, _message("t5", "/idea add a flag")) == {"idea": "forwarded"}
    assert [b["message_id"] for _, b, _ in assistant.received] == ["t1", "t2"]
    assert len(ideas.received) == 1


def test_groups_other_people_and_echoes(rig):
    gateway, _, assistant, _, _ = rig
    assert "assistant" not in _post(gateway, _message("g1", "is the PC up?", chat=GROUP, sender=OWNER))
    assert "assistant" not in _post(gateway, _message("a1", "is the PC up?", chat=ALICE))
    assert _post(gateway, _message("e1", "🤖 The PC is up.", from_me=True)) == {"assistant": "echo"}
    assert assistant.received == []


def test_inbox_down_leaves_the_message_to_kaya(rig):
    gateway, _, assistant, _, _ = rig
    assistant.raises = True
    result = _post(gateway, _message("d1", "hello"))
    assert "assistant" not in result and gateway.journal.pending_count() == 1


def test_pc_asleep_says_so_once(rig):
    gateway, _, assistant, sent, monitor = rig
    monitor.observe(PcState.OFFLINE)
    _post(gateway, _message("s1", "is the PC up?"))
    _post(gateway, _message("s1", "is the PC up?"))
    assert sent == [(OWNER, ASLEEP_TEXT, "s1")] and len(assistant.received) == 2


def test_handback_journals_the_text_for_kaya_without_the_taps(rig):
    gateway, _, assistant, _, _ = rig
    _post(gateway, _voice("h1"))
    event = assistant.received[0][1]["event"]
    client = TestClient(gateway.internal_app)
    assert client.post("/assistant/handback", json={"event": event, "text": "what did Bana say?"}).status_code == 401
    r = client.post("/assistant/handback", json={"event": event, "text": "what did Bana say?"},
                    headers={"X-Relay-Token": "relay"})
    assert r.status_code == 200 and "seq" in r.json()
    assert len(assistant.received) == 1 and gateway.journal.pending_count() == 1
    alice = _message("x1", "hi", chat=ALICE)
    assert client.post("/assistant/handback", json={"event": alice, "text": "hi"},
                       headers={"X-Relay-Token": "relay"}).status_code == 400


def test_off_without_url():
    assert AssistantTap.from_env({}) is None
    tap = AssistantTap.from_env({"ASSISTANT_INBOX_URL": "http://x", "ASSISTANT_OWNER_NUMBERS": "351933333333",
                                 "IDEA_OWNER_NUMBERS": "351922222222"})
    assert tap.owner_numbers == {"351933333333"}
