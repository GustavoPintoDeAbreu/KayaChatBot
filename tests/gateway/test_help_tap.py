"""The owner's /help DM: a fixed command list answered by the Pi, never journaled."""
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.app import Gateway, GatewaySettings
from src.gateway.help_tap import HELP_TAG, HELP_TEXT, HelpTap
from src.gateway.monitor import PcMonitor, PcState

BOT = "351900000000@c.us"
OWNER = "351922222222@c.us"
OWNER_LID = "64622145081581@lid"
ALICE = "351911111111@c.us"
GROUP = "120363000000000000@g.us"


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
    sent = []
    tap = HelpTap.from_env({"HELP_OWNER_NUMBERS": "+351922222222",
                            "GATEWAY_HELP_URL": "http://help.example/"})
    gateway = Gateway(settings, monitor=monitor, help_tap=tap,
                      send_text=lambda chat, text, reply_to=None: sent.append((chat, text, reply_to)))
    return gateway, sent


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


def test_help_is_answered_from_the_pi_and_not_journaled(rig):
    gateway, sent = rig
    assert _post(gateway, _message("m1", "/help")) == {"help": "sent"}
    assert gateway.journal.pending_count() == 0
    assert sent == [(OWNER, f"{HELP_TAG} {HELP_TEXT}\n\n📖 Página de ajuda: http://help.example", "m1")]


def test_help_answers_while_the_pc_is_off(rig):
    gateway, sent = rig
    gateway.monitor.observe(PcState.OFFLINE)
    assert _post(gateway, _message("m2", "/help")) == {"help": "sent"}
    assert len(sent) == 1


def test_help_omits_the_link_when_no_url_is_set():
    tap = HelpTap(help_url="", owner_numbers={"351922222222"})
    assert tap.reply_text() == HELP_TEXT
    assert "Página de ajuda" not in tap.reply_text()


def test_help_url_is_stripped_of_a_trailing_slash():
    tap = HelpTap.from_env({"HELP_OWNER_NUMBERS": "+351922222222", "GATEWAY_HELP_URL": "http://help.example/"})
    assert tap.help_url == "http://help.example"


def test_help_is_owner_only(rig):
    gateway, sent = rig
    assert "seq" in _post(gateway, _message("n1", "/help", chat=ALICE))
    assert "seq" in _post(gateway, _message("g1", "/help", chat=GROUP))
    assert sent == []


def test_the_owner_is_recognised_behind_a_lid(rig):
    gateway, _ = rig
    event = _message("l1", "/help", chat=OWNER_LID, alt="351922222222@s.whatsapp.net")
    assert _post(gateway, event) == {"help": "sent"}


def test_help_is_case_insensitive_but_not_mid_message(rig):
    gateway, sent = rig
    assert _post(gateway, _message("c1", "/HELP")) == {"help": "sent"}
    assert "seq" in _post(gateway, _message("c2", "the /help of it"))
    assert len(sent) == 1


def test_the_help_reply_echo_is_dropped(rig):
    gateway, sent = rig
    assert _post(gateway, _message("e1", f"{HELP_TAG} Kaya — o que podes pedir", from_me=True)) == {"help": "echo"}
    assert gateway.journal.pending_count() == 0 and sent == []
    assert "seq" in _post(gateway, _message("e2", "an ordinary reply", from_me=True))


def test_the_tap_is_off_without_an_owner():
    assert HelpTap.from_env({}) is None
    tap = HelpTap.from_env({"HELP_OWNER_NUMBERS": "351922222222@c.us, 351933333333"})
    assert tap.owner_numbers == {"351922222222", "351933333333"}
    assert HelpTap.from_env({"IDEA_OWNER_NUMBERS": "351922222222"}) is not None
