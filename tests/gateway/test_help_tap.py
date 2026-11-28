"""/help from the Pi: per chat, from the command registry, never journaled (2026-11-25)."""
import json
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.app import Gateway, GatewaySettings
from src.chat import commands as registry
from src.gateway.help_tap import HELP_TAG, CommandSource, HelpTap, _commands_url
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
    tap.source = CommandSource(url="")   # Kaya's own lab.yaml only, no network
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


def _reply(sent):
    assert len(sent) == 1
    return sent[0][1]


def test_help_is_answered_from_the_pi_and_not_journaled(rig):
    gateway, sent = rig
    assert _post(gateway, _message("m1", "/help")) == {"help": "sent"}
    assert gateway.journal.pending_count() == 0
    text = _reply(sent)
    assert text.startswith(f"{HELP_TAG} Kaya — o que podes pedir")
    assert "/homelaboff" in text and "/bug" in text
    assert text.endswith("📖 Página de ajuda: http://help.example")


def test_help_answers_while_the_pc_is_off(rig):
    gateway, sent = rig
    gateway.monitor.observe(PcState.OFFLINE)
    assert _post(gateway, _message("m2", "/help")) == {"help": "sent"}
    assert len(sent) == 1


def test_help_in_the_group_lists_only_group_commands_even_from_the_owner(rig):
    gateway, sent = rig
    assert _post(gateway, _message("g1", "@351900000000 /help", chat=GROUP)) == {"help": "sent"}
    text = _reply(sent)
    assert "/bug" in text and "sondagem" in text
    assert "/homelab" not in text and "/gamingoff" not in text and "Página de ajuda" not in text
    assert gateway.journal.pending_count() == 0


def test_a_members_dm_is_left_to_kaya(rig):
    gateway, sent = rig
    assert "seq" in _post(gateway, _message("n1", "/help", chat=ALICE))
    assert "seq" in _post(gateway, _message("n2", "/hepl", chat=ALICE))
    assert sent == []


def test_the_owners_wrong_command_gets_the_closest_one(rig):
    gateway, sent = rig
    assert _post(gateway, _message("u1", "/homelabof")) == {"help": "unknown"}
    assert "Querias dizer /homelaboff?" in _reply(sent)
    assert gateway.journal.pending_count() == 0


def test_the_owners_kaya_commands_still_reach_kaya(rig):
    gateway, sent = rig
    assert "seq" in _post(gateway, _message("k1", "/bug não respondeu"))
    assert "seq" in _post(gateway, _message("k2", "olá, tudo bem?"))
    assert sent == []


def test_help_omits_the_link_when_no_url_is_set():
    tap = HelpTap(help_url="", owner_numbers={"351922222222"}, source=CommandSource(url=""))
    assert "Página de ajuda" not in tap.reply_text()
    assert "/homelaboff" in tap.reply_text()


def test_help_url_is_stripped_of_a_trailing_slash():
    tap = HelpTap.from_env({"HELP_OWNER_NUMBERS": "+351922222222", "GATEWAY_HELP_URL": "http://help.example/"})
    assert tap.help_url == "http://help.example"


def test_the_registry_url_comes_from_the_help_page_origin():
    assert _commands_url({"GATEWAY_HELP_URL": "http://hub.example:8095/help"}) == \
        "http://hub.example:8095/api/commands"
    assert _commands_url({"GATEWAY_COMMANDS_URL": "http://x/api/commands",
                          "GATEWAY_HELP_URL": "http://hub/help"}) == "http://x/api/commands"
    assert _commands_url({}) == ""


def test_the_hub_list_is_used_cached_and_survives_the_hub_going_down(tmp_path):
    clock = [1000.0]
    rows = [{"product": "idea-pipeline", "cmd": "/idea", "args": "<texto>", "where": ["dm"],
             "who": "owner", "summary": "uma ideia para o pipeline"}]
    calls = []

    def fetch(url):
        calls.append(url)
        if len(calls) > 1:
            raise OSError("hub down")
        return rows

    source = CommandSource(url="http://hub/api/commands", cache_path=tmp_path / "commands.json",
                           fetch=fetch, now=lambda: clock[0])
    assert "/idea" in registry.tokens(source.commands())
    assert json.loads((tmp_path / "commands.json").read_text()) == rows
    assert "/bug" in registry.tokens(source.commands())          # Kaya's own, always
    clock[0] += 3600
    restarted = CommandSource(url="http://hub/api/commands", cache_path=tmp_path / "commands.json",
                              fetch=fetch, now=lambda: clock[0])
    assert "/idea" in registry.tokens(restarted.commands())       # the cached copy
    assert len(calls) == 2


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
