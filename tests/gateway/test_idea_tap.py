"""The idea tap: the owner's /idea DMs go to the idea inbox and never reach Kaya."""
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.app import Gateway, GatewaySettings
from src.gateway.idea_tap import OFFLINE_TEXT, IdeaTap
from src.gateway.monitor import PcMonitor, PcState

BOT = "351900000000@c.us"
OWNER = "351922222222@c.us"
OWNER_LID = "64622145081581@lid"
ALICE = "351911111111@c.us"
GROUP = "120363000000000000@g.us"


class Inbox:
    def __init__(self):
        self.received, self.status, self.raises = [], 200, False

    async def __call__(self, url, body, headers):
        if self.raises:
            raise OSError("connection refused")
        self.received.append((url, body, headers))
        return self.status


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
    inbox, sent = Inbox(), []
    tap = IdeaTap.from_env({"IDEA_INBOX_URL": "http://idea-inbox:8000/", "IDEA_INBOX_TOKEN": "tok",
                            "IDEA_OWNER_NUMBERS": "+351922222222"})
    tap.post = inbox
    gateway = Gateway(settings, monitor=monitor, idea_tap=tap,
                      send_text=lambda chat, text, reply_to=None: sent.append((chat, text, reply_to)))
    return gateway, inbox, sent


def _message(message_id, body, *, chat=OWNER, sender=None, from_me=False, quoted=None, alt=None):
    payload = {"id": message_id, "from": chat, "body": body, "timestamp": 1_790_000_000,
               "fromMe": from_me, "notifyName": "Gustavo"}
    if chat.endswith("@g.us"):
        payload["participant"] = sender or OWNER
    if quoted is not None:
        payload["replyTo"] = {"id": "q1", "body": quoted}
    if alt:
        payload["_data"] = {"key": {"remoteJidAlt": alt, "participantAlt": alt}}
    return {"event": "message", "me": {"id": BOT}, "payload": payload}


def _post(gateway, event):
    return TestClient(gateway.internal_app).post("/waha/webhook", json=event,
                                                 headers={"X-Webhook-Token": "hook"}).json()


def test_an_idea_goes_to_the_inbox_and_not_to_kaya(rig):
    gateway, inbox, _ = rig
    assert _post(gateway, _message("i1", "/idea add a --version flag")) == {"idea": "forwarded"}
    assert gateway.journal.pending_count() == 0
    url, body, headers = inbox.received[0]
    assert url == "http://idea-inbox:8000/ingest" and headers == {"X-Idea-Token": "tok"}
    assert body["message_id"] == "i1" and body["text"] == "/idea add a --version flag"
    assert body["sender"] == "351922222222"


def test_status_and_numbered_answers_are_ideas_too(rig):
    gateway, inbox, _ = rig
    assert _post(gateway, _message("i2", "/ideas"))["idea"] == "forwarded"
    assert _post(gateway, _message("i3", "/IDEA 4 use sqlite"))["idea"] == "forwarded"
    assert len(inbox.received) == 2


def test_a_quoted_reply_to_a_pipeline_question_is_an_answer(rig):
    gateway, inbox, _ = rig
    event = _message("a1", "sqlite, and keep it local", quoted="💡[idea-4] 1. Which database?")
    assert _post(gateway, event) == {"idea": "forwarded"}
    assert inbox.received[0][1]["quoted_text"].startswith("💡[idea-4]")
    assert gateway.journal.pending_count() == 0


def test_the_owner_is_recognised_behind_a_lid(rig):
    gateway, inbox, _ = rig
    event = _message("l1", "/idea something", chat=OWNER_LID, alt="351922222222@s.whatsapp.net")
    assert _post(gateway, event) == {"idea": "forwarded"}


def test_anyone_else_is_left_to_kaya(rig):
    gateway, inbox, _ = rig
    assert "seq" in _post(gateway, _message("n1", "/idea free pizza", chat=ALICE))
    assert "seq" in _post(gateway, _message("n2", "olá", quoted="💡[idea-4] 1. Which database?", chat=ALICE))
    assert inbox.received == []


def test_a_group_message_is_never_tapped(rig):
    gateway, inbox, _ = rig
    assert "seq" in _post(gateway, _message("g1", "/idea in the group", chat=GROUP))
    assert inbox.received == []


def test_ordinary_owner_chat_is_left_to_kaya(rig):
    gateway, inbox, _ = rig
    assert "seq" in _post(gateway, _message("o1", "olá kaya"))
    assert "seq" in _post(gateway, _message("o2", "the /idea of it"))
    assert inbox.received == []


def test_the_pipelines_own_messages_are_dropped(rig):
    gateway, inbox, _ = rig
    assert _post(gateway, _message("e1", "💡[idea-4] PR ready", from_me=True)) == {"idea": "echo"}
    assert gateway.journal.pending_count() == 0 and inbox.received == []
    assert "seq" in _post(gateway, _message("e2", "an ordinary reply", from_me=True))


def test_an_offline_inbox_is_said_once_and_kaya_still_hears_nothing(rig):
    gateway, inbox, sent = rig
    inbox.raises = True
    assert _post(gateway, _message("x1", "/idea lost?")) == {"idea": "inbox-offline"}
    assert _post(gateway, _message("x1", "/idea lost?")) == {"idea": "inbox-offline"}
    assert sent == [(OWNER, OFFLINE_TEXT, "x1")]
    assert gateway.journal.pending_count() == 0


def test_a_refusing_inbox_counts_as_offline(rig):
    gateway, inbox, sent = rig
    inbox.status = 401
    assert _post(gateway, _message("r1", "/idea x")) == {"idea": "inbox-offline"}
    assert len(sent) == 1


def test_the_tap_is_off_without_an_inbox_url():
    assert IdeaTap.from_env({}) is None
    tap = IdeaTap.from_env({"IDEA_INBOX_URL": "http://x", "IDEA_OWNER_NUMBERS": "351922222222@c.us, 351933333333"})
    assert tap.owner_numbers == {"351922222222", "351933333333"}
