"""The PAM hook that lets the owner approve a terminal-less sudo from WhatsApp."""
import importlib.util
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

MODULE_PATH = Path(__file__).parent.parent.parent / "deploy" / "power" / "kaya_sudo_approve.py"
spec = importlib.util.spec_from_file_location("kaya_sudo_approve", MODULE_PATH)
approver = importlib.util.module_from_spec(spec)
sys.modules["kaya_sudo_approve"] = approver
spec.loader.exec_module(approver)

ENV = {"PAM_SERVICE": "sudo", "PAM_TYPE": "auth", "PAM_USER": "gustavo", "PAM_TTY": ""}
CONFIG = {"POWER_USER": "gustavo", "KAYA_RELAY_TOKEN": "relay", "POWER_GATEWAY_URL": "http://pi:8088/"}
REQUEST = {"command": "sudo -A deploy/power/install.sh", "cwd": "/home/gustavo", "origin": "bash ← claude"}


class Gateway:
    def __init__(self, statuses, request_id="abc", fail=False):
        self.statuses = list(statuses)
        self.request_id = request_id
        self.fail = fail
        self.calls = []

    def __call__(self, method, url, token, body=None):
        self.calls.append((method, url, token, body))
        if self.fail:
            raise urllib.error.URLError("refused")
        if method == "POST":
            return {"id": self.request_id}
        return {"status": self.statuses.pop(0) if self.statuses else "pending"}


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def _approve(gateway, environ=ENV, config=CONFIG, request=REQUEST, clock=None):
    clock = clock or Clock()
    return approver.approve(environ, config, request, call=gateway, sleep=clock.sleep, now=clock)


def test_approved_after_polling():
    gateway = Gateway(["pending", "pending", "approved"])
    assert _approve(gateway) == 0
    method, url, token, body = gateway.calls[0]
    assert (method, url, token, body) == ("POST", "http://pi:8088/pc/sudo/request", "relay", REQUEST)
    assert gateway.calls[-1][:2] == ("GET", "http://pi:8088/pc/sudo/abc")


def test_denied_or_expired_falls_through():
    assert _approve(Gateway(["denied"])) == 1
    assert _approve(Gateway(["expired"])) == 1


def test_times_out():
    clock = Clock()
    assert _approve(Gateway([]), clock=clock) == 1
    assert clock.value >= approver.WAIT_SECONDS


def test_never_asks_with_a_terminal_another_user_or_service():
    for environ in ({**ENV, "PAM_TTY": "/dev/pts/3"}, {**ENV, "PAM_USER": "root"},
                    {**ENV, "PAM_SERVICE": "su"}, {**ENV, "PAM_TYPE": "account"}):
        gateway = Gateway(["approved"])
        assert _approve(gateway, environ=environ) == 1
        assert gateway.calls == []


def test_missing_config_or_gateway_down_falls_through():
    gateway = Gateway(["approved"])
    assert _approve(gateway, config={**CONFIG, "KAYA_RELAY_TOKEN": ""}) == 1
    assert gateway.calls == []
    assert _approve(Gateway([], fail=True)) == 1


def test_a_poll_error_keeps_waiting():
    class Flaky(Gateway):
        def __call__(self, method, url, token, body=None):
            if method == "GET" and len(self.calls) == 1:
                self.calls.append((method, url, token, body))
                raise urllib.error.URLError("blip")
            return super().__call__(method, url, token, body)

    assert _approve(Flaky(["approved"])) == 0


def test_describe_reads_the_sudo_process(tmp_path):
    shell = subprocess.Popen(["sh", "-c", "sleep 5; true"], cwd=tmp_path)
    try:
        sleeper = 0
        for _ in range(100):
            children = Path(f"/proc/{shell.pid}/task/{shell.pid}/children").read_text().split()
            if children:
                sleeper = int(children[0])
                break
            time.sleep(0.02)
        info = approver.describe(sleeper)
    finally:
        shell.kill()
        shell.wait()
    assert info["command"] == "sleep 5"
    assert info["cwd"] == str(tmp_path)
    assert info["origin"].split(" ← ")[0] == "sh"


def test_terminal_detection():
    assert approver.has_terminal("/dev/pts/0") and approver.has_terminal("/dev/tty1")
    assert not approver.has_terminal("") and not approver.has_terminal("/dev/null")
    assert not approver.has_terminal("ssh")
