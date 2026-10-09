"""The owner's /homelaboff and /homelabon DMs go to the PC's power listener, not Kaya."""
import asyncio
import sys
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.app import Gateway, GatewaySettings
from src.gateway.homelab import (
    ALREADY_OFF, ALREADY_ON, ALREADY_STOPPING, ALREADY_WAITING, ASK_BUSY, ASK_IDLE,
    CANCEL_FAILED, CANCELLING, DECLINED, HOMELAB_TAG, NOTHING_TO_CANCEL, PC_ONLINE,
    RC_ALREADY, RC_FAILED, RC_NOT_RUNNING, RC_PC_OFF, RC_STARTED, RC_STARTED_NO_LINK, RC_STOPPED,
    RC_USAGE, STARTED, SUDO_ALLOWED, SUDO_ASK, SUDO_REFUSED, SUDO_TTL_SECONDS, TOO_LATE, UNREACHABLE, WAKE_FAILED, WAKE_SENT, WAKE_TIMEOUT,
    HomelabPower,
)
from src.gateway.monitor import PcMonitor, PcState

BOT = "351900000000@c.us"
OWNER = "351922222222@c.us"
OWNER_LID = "64622145081581@lid"
ALICE = "351911111111@c.us"
GROUP = "120363000000000000@g.us"
RC_URL = "https://claude.ai/code/session_01QfLguPaAQUGmXdHNVgK5pw"


class Clock:
    def __init__(self, start: float = 1_790_000_000.0):
        self.value = start

    def __call__(self) -> float:
        return self.value


class FakePc:
    """The PC's power listener, by hand: one (status, body) per path."""

    def __init__(self):
        self.calls = []
        self.status_by_path = {
            "/power/status": (200, {"busy": [], "notes": [], "manual": "idle"}),
            "/power/off": (202, {"started": True}),
            "/power/cancel": (200, {"cancelled": True}),
            "/rc/start": (200, {"started": True, "name": "homelab-home", "url": RC_URL}),
            "/rc/stop": (200, {"stopped": True}),
        }
        self.raises = False

    async def __call__(self, method, url, headers):
        if self.raises:
            raise httpx.ConnectError("refused")
        path = urlsplit(url).path
        self.calls.append((method, url, headers))
        return self.status_by_path[path]


@pytest.fixture
def rig(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "whatsapp": {"group": {"respond_on_mention": True, "respond_on_reply": True},
                     "whitelist": {"enabled": True, "allowed": ["351911111111", "351922222222"]}},
        "power": {"timezone": "Europe/Lisbon", "wake_time": "07:00", "wol_lead_minutes": 5,
                  "shutdown": {day: "23:00" for day in ("sun", "mon", "tue", "wed", "thu", "fri", "sat")}},
    }))
    landing = tmp_path / "landing.html"
    landing.write_text("<html>kaya</html>")
    settings = GatewaySettings.from_env({
        "GATEWAY_DATA_DIR": str(tmp_path / "data"), "GATEWAY_CONFIG": str(config),
        "GATEWAY_LANDING_HTML": str(landing), "GATEWAY_WHITELIST": str(tmp_path / "none.json"),
        "KAYA_RELAY_TOKEN": "relay", "KAYA_WAHA_API_KEY": "waha-key",
        "KAYA_WHATSAPP_WEBHOOK_TOKEN": "hook",
    })
    clock = Clock()
    monitor = PcMonitor(settings.pc_url, now=clock, probe=lambda: PcState.ONLINE)
    monitor.observe(PcState.ONLINE)
    sent = []
    homelab = HomelabPower.from_env({
        "HOMELAB_OWNER_NUMBERS": "+351922222222",
        "KAYA_PC_URL": "http://192.168.1.149:7860",
        "KAYA_RELAY_TOKEN": "relay",
    })
    pc = FakePc()
    homelab.call = pc
    gateway = Gateway(settings, monitor=monitor, now=clock, homelab=homelab,
                      send_text=lambda chat, text, reply_to=None: sent.append((chat, text, reply_to)))
    return gateway, clock, monitor, sent, pc, homelab


def _message(message_id, body, *, chat=OWNER, sender=None, from_me=False, timestamp=1_790_000_000, alt=None,
             quoting=None):
    payload = {"id": message_id, "from": chat, "body": body, "timestamp": timestamp,
               "fromMe": from_me, "notifyName": "Gustavo"}
    if quoting is not None:
        payload["replyTo"] = {"id": "q1", "participant": BOT, "body": quoting}
    if chat.endswith("@g.us"):
        payload["participant"] = sender or OWNER
    if alt:
        payload["_data"] = {"key": {"remoteJidAlt": alt, "participantAlt": alt}}
    return {"event": "message", "me": {"id": BOT}, "payload": payload}


def _post(gateway, event):
    return TestClient(gateway.internal_app).post("/waha/webhook", json=event,
                                                 headers={"X-Webhook-Token": "hook"}).json()


def test_off_without_owner_numbers_is_off():
    assert HomelabPower.from_env({}) is None
    homelab = HomelabPower.from_env({"HOMELAB_OWNER_NUMBERS": "351922222222",
                                     "KAYA_RELAY_TOKEN": "relay"})
    assert homelab.pc_power_url == "http://192.168.1.149:8099"
    homelab = HomelabPower.from_env({"HOMELAB_OWNER_NUMBERS": "351922222222",
                                     "HOMELAB_PC_POWER_URL": "http://10.0.0.5:9999/",
                                     "KAYA_RELAY_TOKEN": "relay"})
    assert homelab.pc_power_url == "http://10.0.0.5:9999"


def test_homelaboff_lists_what_is_running_and_asks(rig):
    gateway, _, _, sent, pc, _ = rig
    pc.status_by_path["/power/status"] = (200, {"busy": ["a CI job is running", "a GPU is 90% busy"],
                                                "notes": ["Kaya is finishing 1 reply"], "manual": "idle"})
    assert _post(gateway, _message("h1", "/homelaboff")) == {"homelab": "off-asked"}
    assert gateway.journal.pending_count() == 0
    method, url, headers = pc.calls[0]
    assert method == "GET" and url == "http://192.168.1.149:8099/power/status"
    assert headers == {"X-Relay-Token": "relay"}
    chat, text, reply_to = sent[0]
    assert chat == OWNER and reply_to == "h1"
    expected = (f"{HOMELAB_TAG} " + ASK_BUSY.format(items="- a CI job is running\n- a GPU is 90% busy")
                + "\n\nNotes:\n- Kaya is finishing 1 reply")
    assert text == expected


def test_nothing_running_still_asks(rig):
    gateway, _, _, sent, _, _ = rig
    assert _post(gateway, _message("h2", "/homelaboff")) == {"homelab": "off-asked"}
    assert sent[0][1] == f"{HOMELAB_TAG} " + ASK_IDLE


def test_notes_are_listed(rig):
    gateway, _, _, sent, pc, _ = rig
    pc.status_by_path["/power/status"] = (200, {"busy": [], "notes": ["kernel 7.0.0-35 has no nvidia module"],
                                                "manual": "idle"})
    assert _post(gateway, _message("h3", "/homelaboff")) == {"homelab": "off-asked"}
    assert ASK_IDLE in sent[0][1] and "kernel 7.0.0-35 has no nvidia module" in sent[0][1]


def test_yes_starts_the_shutdown(rig):
    gateway, _, _, sent, pc, homelab = rig
    _post(gateway, _message("h4", "/homelaboff"))
    assert _post(gateway, _message("h5", "yes")) == {"homelab": "off-started"}
    assert pc.calls[1][0] == "POST" and pc.calls[1][1].endswith("/power/off")
    assert homelab.active_chat() == OWNER
    assert sent[1][1] == f"{HOMELAB_TAG} " + STARTED


def test_yes_with_nothing_pending_goes_to_kaya(rig):
    gateway, _, _, sent, _, _ = rig
    result = _post(gateway, _message("h6", "yes"))
    assert "seq" in result and sent == []


def test_confirmation_expires(rig):
    gateway, clock, _, sent, _, _ = rig
    _post(gateway, _message("h7", "/homelaboff"))
    clock.value += 301
    result = _post(gateway, _message("h8", "yes", timestamp=int(clock.value)))
    assert "seq" in result and len(sent) == 1


def test_no_keeps_the_pc_on(rig):
    gateway, _, _, sent, pc, _ = rig
    _post(gateway, _message("h9", "/homelaboff"))
    assert _post(gateway, _message("h10", "no")) == {"homelab": "off-declined"}
    assert sent[1][1] == f"{HOMELAB_TAG} " + DECLINED
    assert [call[0] for call in pc.calls] == ["GET"]


def test_cancel_while_waiting(rig):
    gateway, _, _, sent, pc, homelab = rig
    pc.status_by_path["/power/status"] = (200, {"busy": [], "notes": [], "manual": "waiting"})
    _post(gateway, _message("h11", "/homelaboff"))
    assert homelab.active_chat() == OWNER
    assert _post(gateway, _message("h12", "cancel")) == {"homelab": "cancel-requested"}
    assert pc.calls[1][0] == "POST" and pc.calls[1][1].endswith("/power/cancel")
    assert sent[1][1] == f"{HOMELAB_TAG} " + CANCELLING


def test_cancel_too_late(rig):
    gateway, _, _, sent, pc, _ = rig
    pc.status_by_path["/power/status"] = (200, {"busy": [], "notes": [], "manual": "waiting"})
    _post(gateway, _message("h13", "/homelaboff"))
    pc.status_by_path["/power/cancel"] = (409, {"reason": "too late"})
    assert _post(gateway, _message("h14", "cancel")) == {"homelab": "cancel-too-late"}
    assert sent[1][1] == f"{HOMELAB_TAG} " + TOO_LATE


def test_cancel_with_nothing_running_clears_the_state(rig):
    gateway, _, _, sent, pc, homelab = rig
    pc.status_by_path["/power/status"] = (200, {"busy": [], "notes": [], "manual": "waiting"})
    _post(gateway, _message("h15", "/homelaboff"))
    pc.status_by_path["/power/cancel"] = (409, {"reason": "not running"})
    assert _post(gateway, _message("h16", "cancel")) == {"homelab": "cancel-none"}
    assert sent[1][1] == f"{HOMELAB_TAG} " + NOTHING_TO_CANCEL
    assert homelab.active_chat() == ""
    result = _post(gateway, _message("h16b", "cancel"))
    assert "seq" in result and len(sent) == 2


def test_unreachable_pc_does_nothing(rig):
    gateway, _, _, sent, pc, _ = rig
    pc.raises = True
    assert _post(gateway, _message("h17", "/homelaboff")) == {"homelab": "off-unreachable"}
    assert sent[0][1] == f"{HOMELAB_TAG} " + UNREACHABLE.format(error="ConnectError")
    assert gateway.journal.pending_count() == 0
    result = _post(gateway, _message("h17b", "yes"))
    assert "seq" in result and len(sent) == 1


def test_off_when_already_off(rig):
    gateway, _, monitor, sent, pc, _ = rig
    pc.raises = True
    monitor.observe(PcState.OFFLINE)
    assert _post(gateway, _message("h18", "/homelaboff")) == {"homelab": "off-already"}
    assert sent[0][1] == f"{HOMELAB_TAG} " + ALREADY_OFF


def test_already_waiting(rig):
    gateway, _, _, sent, pc, _ = rig
    pc.status_by_path["/power/status"] = (200, {"busy": [], "notes": [], "manual": "waiting"})
    assert _post(gateway, _message("h19", "/homelaboff")) == {"homelab": "off-waiting"}
    assert sent[0][1] == f"{HOMELAB_TAG} " + ALREADY_WAITING


def test_already_stopping(rig):
    gateway, _, _, sent, pc, _ = rig
    pc.status_by_path["/power/status"] = (200, {"busy": [], "notes": [], "manual": "stopping"})
    assert _post(gateway, _message("h20", "/homelaboff")) == {"homelab": "off-stopping"}
    assert sent[0][1] == f"{HOMELAB_TAG} " + ALREADY_STOPPING


def test_only_the_owner_in_a_dm(rig):
    gateway, _, _, sent, _, _ = rig
    result = _post(gateway, _message("h21", "/homelaboff", chat=ALICE))
    assert "seq" in result and sent == []


def test_owner_behind_a_lid(rig):
    gateway, _, _, sent, _, _ = rig
    event = _message("h22", "/homelaboff", chat=OWNER_LID, alt="351922222222@s.whatsapp.net")
    assert _post(gateway, event) == {"homelab": "off-asked"}
    assert sent[0][0] == OWNER_LID


def test_a_group_message_is_never_tapped(rig):
    gateway, _, _, sent, _, _ = rig
    result = _post(gateway, _message("h23", "/homelaboff", chat=GROUP, sender=OWNER))
    assert "seq" in result and sent == []


def test_duplicate_delivery_is_handled_once(rig):
    gateway, _, _, sent, pc, _ = rig
    assert _post(gateway, _message("h24", "/homelaboff")) == {"homelab": "off-asked"}
    assert _post(gateway, _message("h24", "/homelaboff")) == {"homelab": "duplicate"}
    assert len(sent) == 1 and len(pc.calls) == 1


def test_stale_backlog_is_ignored(rig):
    gateway, clock, _, sent, pc, _ = rig
    stale = int(clock.value) - 400
    assert _post(gateway, _message("h25", "/homelaboff", timestamp=stale)) == {"homelab": "stale"}
    assert sent == [] and pc.calls == []


def test_own_echo_is_dropped(rig):
    gateway, _, _, sent, pc, _ = rig
    assert _post(gateway, _message("h26", f"{HOMELAB_TAG} {ASK_IDLE}", from_me=True)) == {"homelab": "echo"}
    assert sent == [] and pc.calls == [] and gateway.journal.pending_count() == 0


def test_homelabon_requests_wake_and_reports_online(rig):
    gateway, clock, monitor, sent, _, homelab = rig
    monitor.observe(PcState.OFFLINE)
    assert _post(gateway, _message("h27", "/homelabon")) == {"homelab": "on-requested"}
    assert sent[0][1] == f"{HOMELAB_TAG} " + WAKE_SENT
    request = Path(gateway.settings.data_dir) / "pc-wake.request"
    assert request.exists()
    monitor.observe(PcState.ONLINE)
    homelab.tick()
    assert sent[1][1] == f"{HOMELAB_TAG} " + PC_ONLINE
    assert OWNER not in homelab._waking


def test_homelabon_when_already_on(rig):
    gateway, _, _, sent, _, _ = rig
    assert _post(gateway, _message("h28", "/homelabon")) == {"homelab": "on-already"}
    assert sent[0][1] == f"{HOMELAB_TAG} " + ALREADY_ON


def test_homelabon_times_out(rig):
    gateway, clock, monitor, sent, _, homelab = rig
    monitor.observe(PcState.OFFLINE)
    _post(gateway, _message("h29", "/homelabon"))
    clock.value += 901
    homelab.tick()
    assert sent[1][1] == f"{HOMELAB_TAG} " + WAKE_TIMEOUT
    assert OWNER not in homelab._waking


def test_power_update_needs_the_token(rig):
    gateway, *_ = rig
    client = TestClient(gateway.internal_app)
    assert client.post("/pc/power/update", json={"text": "x"}).status_code == 401
    assert client.post("/pc/power/update", json={"text": "x"},
                       headers={"X-Relay-Token": "wrong"}).status_code == 401


def test_power_update_goes_to_the_active_chat(rig):
    gateway, _, _, sent, pc, homelab = rig
    _post(gateway, _message("h30", "/homelaboff"))
    _post(gateway, _message("h31", "yes"))
    client = TestClient(gateway.internal_app)
    assert client.post("/pc/power/update", json={"text": "Waiting for:\n- a CI job is running"},
                       headers={"X-Relay-Token": "relay"}).json() == {"ok": True, "sent": True}
    chat, text, _ = sent[2]
    assert chat == OWNER and text == f"{HOMELAB_TAG} Waiting for:\n- a CI job is running"
    assert homelab.active_chat() == OWNER
    assert client.post("/pc/power/update", json={"text": "Powering off now.", "final": True},
                       headers={"X-Relay-Token": "relay"}).json() == {"ok": True, "sent": True}
    assert homelab.active_chat() == ""


def test_power_update_without_a_request_goes_to_the_owner_dm(rig):
    gateway, _, _, sent, _, _ = rig
    client = TestClient(gateway.internal_app)
    assert client.post("/pc/power/update", json={"text": "hello"},
                       headers={"X-Relay-Token": "relay"}).json() == {"ok": True, "sent": True}
    assert sent[0][0] == "351922222222@c.us"


def test_power_update_is_404_when_homelab_is_off(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "whatsapp": {"whitelist": {"enabled": True, "allowed": ["351922222222"]}},
        "power": {"timezone": "Europe/Lisbon", "wake_time": "07:00", "wol_lead_minutes": 5, "shutdown": {}},
    }))
    settings = GatewaySettings.from_env({
        "GATEWAY_DATA_DIR": str(tmp_path / "data"), "GATEWAY_CONFIG": str(config),
        "GATEWAY_WHITELIST": str(tmp_path / "none.json"), "KAYA_RELAY_TOKEN": "relay",
        "KAYA_WHATSAPP_WEBHOOK_TOKEN": "hook",
    })
    monitor = PcMonitor(settings.pc_url, probe=lambda: PcState.ONLINE)
    monitor.observe(PcState.ONLINE)
    gateway = Gateway(settings, monitor=monitor,
                      send_text=lambda chat, text, reply_to=None: None)
    client = TestClient(gateway.internal_app)
    assert client.post("/pc/power/update", json={"text": "x"},
                       headers={"X-Relay-Token": "relay"}).status_code == 404


def test_background_tasks_include_the_watcher(rig):
    gateway, *_ = rig

    async def cycle():
        await gateway.start_background()
        await asyncio.sleep(0.05)
        await asyncio.wait_for(gateway.stop_background(), timeout=10)

    asyncio.run(cycle())


def test_homelabrc_starts_a_session_in_desktop_by_default(rig):
    gateway, _, _, sent, pc, _ = rig
    pc.status_by_path["/rc/start"] = (200, {"started": True, "name": "homelab-desktop", "url": RC_URL})
    assert _post(gateway, _message("r1", "/homelabrc")) == {"homelab": "rc-started"}
    method, url, headers = pc.calls[0]
    assert method == "POST" and url == "http://192.168.1.149:8099/rc/start?dir=desktop"
    assert headers == {"X-Relay-Token": "relay"}
    assert sent == [(OWNER, f"{HOMELAB_TAG} " + RC_STARTED.format(name="homelab-desktop", url=RC_URL), "r1")]
    assert gateway.journal.pending_count() == 0


@pytest.mark.parametrize("text,target", [("/homelabrc desk", "desktop"), ("/homelabrc Desktop", "desktop"),
                                         ("/HomelabRC home", "home"), ("/homelabrc ~", "home")])
def test_homelabrc_picks_the_directory(rig, text, target):
    gateway, _, _, _, pc, _ = rig
    assert _post(gateway, _message("r2", text)) == {"homelab": "rc-started"}
    assert pc.calls[0][1].endswith(f"/rc/start?dir={target}")


def test_homelabrc_without_a_link_still_names_the_session(rig):
    gateway, _, _, sent, pc, _ = rig
    pc.status_by_path["/rc/start"] = (200, {"started": True, "name": "homelab-desktop", "url": None})
    assert _post(gateway, _message("r3", "/homelabrc desk")) == {"homelab": "rc-started"}
    assert sent[0][1] == f"{HOMELAB_TAG} " + RC_STARTED_NO_LINK.format(name="homelab-desktop")


def test_homelabrc_already_running(rig):
    gateway, _, _, sent, pc, _ = rig
    pc.status_by_path["/rc/start"] = (409, {"started": False, "name": "homelab-desktop"})
    assert _post(gateway, _message("r4", "/homelabrc")) == {"homelab": "rc-already"}
    assert sent[0][1] == f"{HOMELAB_TAG} " + RC_ALREADY.format(name="homelab-desktop", suffix="")
    pc.status_by_path["/rc/start"] = (409, {"started": False, "name": "homelab-home"})
    assert _post(gateway, _message("r4b", "/homelabrc home")) == {"homelab": "rc-already"}
    assert sent[1][1] == f"{HOMELAB_TAG} " + RC_ALREADY.format(name="homelab-home", suffix=" home")


def test_homelabrc_failure_carries_the_error(rig):
    gateway, _, _, sent, pc, _ = rig
    pc.status_by_path["/rc/start"] = (500, {"started": False, "error": "workspace not trusted"})
    assert _post(gateway, _message("r5", "/homelabrc home")) == {"homelab": "rc-failed"}
    assert sent[0][1] == f"{HOMELAB_TAG} " + RC_FAILED.format(name="homelab-home",
                                                              error="workspace not trusted")


def test_homelabrc_stop(rig):
    gateway, _, _, sent, pc, _ = rig
    assert _post(gateway, _message("r6", "/homelabrc stop")) == {"homelab": "rc-stopped"}
    assert pc.calls[0][0] == "POST" and pc.calls[0][1].endswith("/rc/stop?dir=desktop")
    assert sent[0][1] == f"{HOMELAB_TAG} " + RC_STOPPED.format(name="homelab-desktop")
    pc.status_by_path["/rc/stop"] = (404, {"stopped": False})
    assert _post(gateway, _message("r7", "/homelabrc stop home")) == {"homelab": "rc-not-running"}
    assert pc.calls[1][1].endswith("/rc/stop?dir=home")
    assert sent[1][1] == f"{HOMELAB_TAG} " + RC_NOT_RUNNING.format(name="homelab-home")


@pytest.mark.parametrize("text", ["/homelabrc /etc", "/homelabrc desk extra", "/homelabrc stop nowhere"])
def test_homelabrc_bad_arguments_get_usage_and_touch_nothing(rig, text):
    gateway, _, _, sent, pc, _ = rig
    assert _post(gateway, _message("r8", text)) == {"homelab": "rc-usage"}
    assert pc.calls == []
    assert sent[0][1] == f"{HOMELAB_TAG} " + RC_USAGE


def test_homelabrc_when_the_pc_is_off_or_unreachable(rig):
    gateway, _, monitor, sent, pc, _ = rig
    pc.raises = True
    assert _post(gateway, _message("r9", "/homelabrc")) == {"homelab": "rc-unreachable"}
    assert sent[0][1] == f"{HOMELAB_TAG} " + UNREACHABLE.format(error="ConnectError")
    monitor.observe(PcState.OFFLINE)
    assert _post(gateway, _message("r10", "/homelabrc")) == {"homelab": "rc-pc-off"}
    assert sent[1][1] == f"{HOMELAB_TAG} " + RC_PC_OFF


def test_homelabrc_from_anyone_else_goes_to_kaya(rig):
    gateway, _, _, sent, pc, _ = rig
    result = _post(gateway, _message("r11", "/homelabrc", chat=ALICE))
    assert "seq" in result and pc.calls == [] and sent == []
    result = _post(gateway, _message("r12", "/homelabrc", chat=GROUP, sender=OWNER))
    assert "seq" in result and pc.calls == []


def _sudo_request(gateway, token="relay", **body):
    payload = {"command": "sudo -A deploy/power/install.sh", "cwd": "/home/gustavo/Desktop/KayaChatBot",
               "origin": "bash ← claude"}
    payload.update(body)
    return TestClient(gateway.internal_app).post("/pc/sudo/request", json=payload,
                                                 headers={"X-Relay-Token": token})


def _sudo_status(gateway, key, token="relay"):
    return TestClient(gateway.internal_app).get(f"/pc/sudo/{key}", headers={"X-Relay-Token": token})


def _code(text):
    return text.split('"yes ')[1].split('"')[0]


def test_sudo_request_needs_the_token(rig):
    gateway, _, _, sent, _, _ = rig
    assert _sudo_request(gateway, token="wrong").status_code == 401
    assert _sudo_request(gateway, command="").status_code == 400
    assert _sudo_status(gateway, "nope").status_code == 404
    assert sent == []


def test_sudo_yes_with_the_code_approves_it(rig):
    gateway, _, _, sent, _, _ = rig
    key = _sudo_request(gateway).json()["id"]
    chat, text, _ = sent[0]
    code = _code(text)
    assert chat == OWNER and text == f"{HOMELAB_TAG} " + SUDO_ASK.format(
        command="sudo -A deploy/power/install.sh", cwd="/home/gustavo/Desktop/KayaChatBot",
        origin="bash ← claude", code=code)
    assert _sudo_status(gateway, key).json() == {"status": "pending"}
    assert _post(gateway, _message("s1", f"Yes {code}")) == {"homelab": "sudo-approved"}
    assert _sudo_status(gateway, key).json() == {"status": "approved"}
    assert sent[1][1] == f"{HOMELAB_TAG} " + SUDO_ALLOWED.format(command="sudo -A deploy/power/install.sh")
    assert gateway.journal.pending_count() == 0


def test_sudo_no_refuses_and_a_wrong_code_goes_to_kaya(rig):
    gateway, _, _, sent, _, _ = rig
    key = _sudo_request(gateway).json()["id"]
    code = _code(sent[0][1])
    wrong = "0000" if code != "0000" else "1111"
    assert "seq" in _post(gateway, _message("s2", f"yes {wrong}"))
    assert _sudo_status(gateway, key).json() == {"status": "pending"}
    assert _post(gateway, _message("s3", f"no {code}")) == {"homelab": "sudo-denied"}
    assert _sudo_status(gateway, key).json() == {"status": "denied"}
    assert sent[-1][1] == f"{HOMELAB_TAG} " + SUDO_REFUSED.format(command="sudo -A deploy/power/install.sh")
    assert "seq" in _post(gateway, _message("s4", f"yes {code}"))


def test_sudo_only_the_owner_can_answer(rig):
    gateway, _, _, sent, _, _ = rig
    key = _sudo_request(gateway).json()["id"]
    code = _code(sent[0][1])
    assert "seq" in _post(gateway, _message("s5", f"yes {code}", chat=ALICE))
    assert "seq" in _post(gateway, _message("s6", f"yes {code}", chat=GROUP, sender=OWNER))
    assert _sudo_status(gateway, key).json() == {"status": "pending"}


def test_sudo_request_expires(rig):
    gateway, clock, _, sent, _, _ = rig
    key = _sudo_request(gateway).json()["id"]
    code = _code(sent[0][1])
    clock.value += SUDO_TTL_SECONDS + 1
    assert _sudo_status(gateway, key).json() == {"status": "expired"}
    assert "seq" in _post(gateway, _message("s7", f"yes {code}", timestamp=int(clock.value)))
    assert _sudo_status(gateway, key).json() == {"status": "expired"}


def test_sudo_bare_yes_quoting_the_request_approves_it(rig):
    gateway, _, _, sent, _, _ = rig
    key = _sudo_request(gateway).json()["id"]
    asked = sent[0][1]
    assert _post(gateway, _message("q2", "Yes", quoting=asked)) == {"homelab": "sudo-approved"}
    assert _sudo_status(gateway, key).json() == {"status": "approved"}


def test_sudo_bare_no_quoting_the_request_refuses_it(rig):
    gateway, _, _, sent, _, _ = rig
    key = _sudo_request(gateway).json()["id"]
    assert _post(gateway, _message("q3", "não", quoting=sent[0][1])) == {"homelab": "sudo-denied"}
    assert _sudo_status(gateway, key).json() == {"status": "denied"}


def test_a_bare_yes_without_a_quote_or_quoting_something_else_is_not_sudo(rig):
    gateway, _, _, sent, _, _ = rig
    key = _sudo_request(gateway).json()["id"]
    code = _code(sent[0][1])
    assert "seq" in _post(gateway, _message("q4", "yes"))
    assert "seq" in _post(gateway, _message("q5", "yes", quoting=f'sudo on the PC: "yes {code}"'))
    assert _sudo_status(gateway, key).json() == {"status": "pending"}
