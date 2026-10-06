"""The PC's power listener: an authenticated LAN trigger for the manual shutdown."""
import importlib.util
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parent.parent.parent / "deploy" / "power" / "kaya_power_listener.py"
spec = importlib.util.spec_from_file_location("kaya_power_listener", MODULE_PATH)
listener = importlib.util.module_from_spec(spec)
sys.modules["kaya_power_listener"] = listener
spec.loader.exec_module(listener)


class FakeRun:
    """Records argv and returns a preconfigured CompletedProcess per argv prefix."""

    def __init__(self, report_stdout=""):
        self.calls = []
        self.report_stdout = report_stdout
        self.active_state = "inactive"
        self.start_rc = 0
        self.timeout = False

    def __call__(self, argv, timeout):
        self.calls.append(list(argv))
        if self.timeout and argv[0].endswith("kaya-shutdown.sh"):
            raise subprocess.TimeoutExpired(argv, timeout)
        if argv[0] == "systemctl" and argv[1] == "show":
            return subprocess.CompletedProcess(argv, 0, stdout=self.active_state + "\n")
        if argv[0] == "systemctl" and argv[1] == "start":
            return subprocess.CompletedProcess(argv, self.start_rc)
        return subprocess.CompletedProcess(argv, 0, stdout=self.report_stdout)


def _config(tmp_path, **overrides):
    values = {"token": "relay", "allowed_ips": {"127.0.0.1"}, "port": 0,
              "script": "/usr/local/sbin/kaya-shutdown.sh",
              "state_dir": str(tmp_path / "state")}
    values.update(overrides)
    return listener.ListenerConfig(**values)


def test_config_from_env_file_and_defaults(tmp_path):
    env_file = tmp_path / "kaya-power.env"
    env_file.write_text('# comment\nKAYA_RELAY_TOKEN="tok"\nPOWER_ALLOWED_IPS=1.2.3.4, 5.6.7.8\n')
    config = listener.ListenerConfig.load(str(env_file), {})
    assert config.token == "tok"
    assert config.allowed_ips == {"1.2.3.4", "5.6.7.8"}
    assert config.port == 8099 and config.script == "/usr/local/sbin/kaya-shutdown.sh"
    assert config.state_dir == "/run/kaya-power"
    config = listener.ListenerConfig.load(str(tmp_path / "missing"), {"KAYA_RELAY_TOKEN": "env",
                                                                      "POWER_LISTEN_PORT": "9999"})
    assert config.token == "env" and config.port == 9999


def test_status_parses_the_report(tmp_path):
    run = FakeRun(report_stdout="note\tkernel 7.0.0-35 has no nvidia module\n"
                                "busy\ta CI job is running\nbusy\ta GPU is 90% busy\n")
    control = listener.PowerControl(_config(tmp_path), run)
    status, body = control.status()
    assert status == 200
    assert body == {"busy": ["a CI job is running", "a GPU is 90% busy"],
                    "notes": ["kernel 7.0.0-35 has no nvidia module"], "manual": "idle"}


def test_status_times_out(tmp_path):
    run = FakeRun()
    run.timeout = True
    control = listener.PowerControl(_config(tmp_path), run)
    assert control.status() == (504, {"error": "report timed out"})


def test_off_starts_the_unit_and_clears_a_stale_cancel(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "cancel").write_text("cancel")
    run = FakeRun()
    control = listener.PowerControl(_config(tmp_path), run)
    status, body = control.start_off()
    assert status == 202 and body == {"started": True}
    assert not (state / "cancel").exists()
    assert ["systemctl", "start", "--no-block", listener.UNIT] in run.calls


def test_off_refused_while_running(tmp_path):
    run = FakeRun()
    run.active_state = "activating"
    control = listener.PowerControl(_config(tmp_path), run)
    assert control.start_off() == (409, {"started": False, "phase": "waiting"})
    assert ["systemctl", "start", "--no-block", listener.UNIT] not in run.calls


def test_off_refused_while_the_oneshot_is_still_activating(tmp_path):
    # A Type=oneshot without RemainAfterExit sits in "activating" for its whole
    # ExecStart, and is-active reports that as not-active. phase() must not.
    state = tmp_path / "state"
    state.mkdir()
    (state / "phase").write_text("stopping")
    run = FakeRun()
    run.active_state = "activating"
    control = listener.PowerControl(_config(tmp_path), run)
    assert control.phase() == "stopping"
    assert control.start_off() == (409, {"started": False, "phase": "stopping"})
    assert ["systemctl", "show", "-p", "ActiveState", "--value", listener.UNIT] in run.calls
    assert ["systemctl", "start", "--no-block", listener.UNIT] not in run.calls


def test_cancel_only_while_waiting(tmp_path):
    config = _config(tmp_path)
    state = Path(config.state_dir)
    state.mkdir(parents=True)
    run = FakeRun()
    control = listener.PowerControl(config, run)
    assert control.cancel() == (409, {"cancelled": False, "reason": "not running"})
    run.active_state = "activating"
    (state / "phase").write_text("waiting")
    status, body = control.cancel()
    assert status == 200 and body == {"cancelled": True}
    assert (state / "cancel").read_text() == "cancel"
    (state / "phase").write_text("stopping")
    (state / "cancel").unlink()
    assert control.cancel() == (409, {"cancelled": False, "reason": "too late"})


def _request(server, path, method="GET", token="relay"):
    url = f"http://127.0.0.1:{server.server_address[1]}{path}"
    request = urllib.request.Request(url, method=method, headers={"X-Relay-Token": token})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_http_checks_ip_then_token(tmp_path):
    run = FakeRun(report_stdout="busy\ta CI job is running\n")
    config = _config(tmp_path, allowed_ips={"127.0.0.1"})
    server = listener.make_server(config, listener.PowerControl(config, run))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert _request(server, "/power/status") == (200, {"busy": ["a CI job is running"],
                                                          "notes": [], "manual": "idle"})
        assert _request(server, "/power/status", token="wrong") == (401, {"error": "invalid token"})
        assert _request(server, "/power/off", method="POST") == (202, {"started": True})
        assert _request(server, "/nope") == (404, {"error": "not found"})
    finally:
        server.shutdown()
        thread.join()

    other = listener.make_server(_config(tmp_path, allowed_ips={"10.9.9.9"}),
                                 listener.PowerControl(_config(tmp_path), FakeRun()))
    other_thread = threading.Thread(target=other.serve_forever, daemon=True)
    other_thread.start()
    try:
        assert _request(other, "/power/status") == (403, {"error": "forbidden"})
    finally:
        other.shutdown()
        other_thread.join()
