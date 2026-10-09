#!/usr/bin/python3 -I
"""Approve a terminal-less sudo from the owner's phone.

Run by pam_exec as the first line of /etc/pam.d/sudo's auth stack, as root:

    auth [success=done default=ignore] pam_exec.so quiet /usr/local/sbin/kaya-sudo-approve

It acts only for POWER_USER, only for sudo, and only when there is no
terminal (Claude Code and Remote Control sessions). It posts the command to the
Pi gateway, which DMs the owner a 4-digit code, then polls until he answers
"yes <code>" (exit 0: sudo goes ahead without a password) or "no <code>", or
two minutes pass. Every other outcome exits non-zero, and default=ignore sends
sudo on to the usual password, so a missing gateway can never lock sudo out.
Stdlib only; reads the relay token from /etc/kaya-power.env.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

WAIT_SECONDS = 125.0
POLL_SECONDS = 2.0


def read_env_file(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def has_terminal(pam_tty: str) -> bool:
    return pam_tty.startswith("/dev/") and pam_tty != "/dev/null"


def _proc(pid: int, name: str) -> str:
    try:
        return Path(f"/proc/{pid}/{name}").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _parent(pid: int) -> int:
    for line in _proc(pid, "status").splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    return 0


def describe(sudo_pid: int) -> Dict[str, str]:
    """The sudo command line, its directory, and the processes that started it."""
    command = " ".join(part for part in _proc(sudo_pid, "cmdline").split("\0") if part)
    try:
        cwd = os.readlink(f"/proc/{sudo_pid}/cwd")
    except OSError:
        cwd = ""
    chain: List[str] = []
    pid = _parent(sudo_pid)
    while pid > 1 and len(chain) < 4:
        chain.append(_proc(pid, "comm").strip() or str(pid))
        pid = _parent(pid)
    return {"command": command, "cwd": cwd, "origin": " ← ".join(chain)}


def _call(method: str, url: str, token: str, body: Optional[dict] = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"X-Relay-Token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read() or b"{}")


def approve(environ: Mapping[str, str], config: Mapping[str, str], request: Dict[str, str],
            call: Callable[..., dict] = _call, sleep: Callable[[float], None] = time.sleep,
            now: Callable[[], float] = time.monotonic) -> int:
    """0 only when the owner allowed this sudo; 1 otherwise."""
    if environ.get("PAM_SERVICE") != "sudo" or environ.get("PAM_TYPE") != "auth":
        return 1
    if environ.get("PAM_USER") != config.get("POWER_USER", "gustavo"):
        return 1
    if has_terminal(environ.get("PAM_TTY", "")):
        return 1
    token = config.get("KAYA_RELAY_TOKEN", "")
    gateway = config.get("POWER_GATEWAY_URL", "").rstrip("/")
    if not token or not gateway or not request.get("command"):
        return 1
    try:
        key = call("POST", f"{gateway}/pc/sudo/request", token, request).get("id")
    except (urllib.error.URLError, OSError, ValueError):
        return 1
    if not key:
        return 1
    deadline = now() + WAIT_SECONDS
    while now() < deadline:
        sleep(POLL_SECONDS)
        try:
            status = call("GET", f"{gateway}/pc/sudo/{key}", token).get("status")
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if status == "approved":
            return 0
        if status in ("denied", "expired"):
            return 1
    return 1


def main() -> None:
    config = read_env_file(os.environ.get("KAYA_POWER_ENV", "/etc/kaya-power.env"))
    sys.exit(approve(os.environ, config, describe(os.getppid())))


if __name__ == "__main__":
    main()
