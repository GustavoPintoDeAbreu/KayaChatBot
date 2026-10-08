"""The PC's end of /homelaboff and /homelabrc: an authenticated LAN trigger.

The gateway on the Pi is the only allowed client (its IP plus KAYA_RELAY_TOKEN,
read from /etc/kaya-power.env). It never powers anything off itself: it only
starts kaya-shutdown-manual.service, reports what is running, or writes the
cancel file the shutdown script checks. For /homelabrc it starts or stops a
Claude Remote Control session as POWER_USER, in a fixed directory, inside that
user's systemd manager (never as root). Stdlib only; runs as root under the
system python3.
"""
from __future__ import annotations

import fcntl
import hmac
import json
import os
import pwd
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple
from urllib.parse import parse_qs, urlsplit

UNIT = "kaya-shutdown-manual.service"
Runner = Callable[[List[str], float], subprocess.CompletedProcess]
RC_DIRS = {"home": "", "desktop": "Desktop"}
RC_URL = re.compile(r"https://claude\.ai/code/\S+")
RC_WAIT_SECONDS = 20.0


def read_env_file(path: str) -> Dict[str, str]:
    """KEY=VALUE lines; '#' comments and blank lines skipped; surrounding quotes
    stripped; missing file -> {}."""
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


@dataclass
class ListenerConfig:
    token: str
    allowed_ips: Set[str]
    host: str = "0.0.0.0"
    port: int = 8099
    script: str = "/usr/local/sbin/kaya-shutdown.sh"
    state_dir: str = "/run/kaya-power"
    user: str = "gustavo"
    claude: str = ""

    @classmethod
    def load(cls, env_file: str, environ: Mapping[str, str]) -> "ListenerConfig":
        merged = {**read_env_file(env_file), **dict(environ)}
        return cls(
            token=merged.get("KAYA_RELAY_TOKEN", ""),
            allowed_ips={ip.strip() for ip in merged.get("POWER_ALLOWED_IPS", "192.168.1.238").split(",") if ip.strip()},
            host=merged.get("POWER_LISTEN_HOST", "0.0.0.0"),
            port=int(merged.get("POWER_LISTEN_PORT", "8099")),
            script=merged.get("POWER_SHUTDOWN_SCRIPT", "/usr/local/sbin/kaya-shutdown.sh"),
            state_dir=merged.get("POWER_STATE_DIR", "/run/kaya-power"),
            user=merged.get("POWER_USER", "gustavo"),
            claude=merged.get("POWER_RC_CLAUDE", ""),
        )


def _run(argv: List[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


class PowerControl:
    def __init__(self, config: ListenerConfig, run: Runner = _run) -> None:
        self.config = config
        self.run = run

    def phase(self) -> str:
        # is-active exits 0 only for active/reloading, but a Type=oneshot without
        # RemainAfterExit sits in "activating" for its whole ExecStart — the whole
        # wait. show -p ActiveState names that, so it counts as running too.
        state = self.run(["systemctl", "show", "-p", "ActiveState", "--value", UNIT], 10)
        active = state.stdout.strip()
        if active not in ("active", "activating", "deactivating"):
            return "idle"
        try:
            phase = Path(self.config.state_dir, "phase").read_text(encoding="utf-8").strip()
        except OSError:
            return "waiting"
        return phase if phase in ("waiting", "stopping") else "waiting"

    def status(self) -> Tuple[int, Dict[str, Any]]:
        try:
            result = self.run([self.config.script, "--report"], 25)
        except subprocess.TimeoutExpired:
            return 504, {"error": "report timed out"}
        busy: List[str] = []
        notes: List[str] = []
        for line in result.stdout.splitlines():
            kind, _, value = line.partition("\t")
            if kind == "busy" and value:
                busy.append(value)
            elif kind == "note" and value:
                notes.append(value)
        return 200, {"busy": busy, "notes": notes, "manual": self.phase()}

    def start_off(self) -> Tuple[int, Dict[str, Any]]:
        phase = self.phase()
        if phase != "idle":
            return 409, {"started": False, "phase": phase}
        Path(self.config.state_dir, "cancel").unlink(missing_ok=True)
        result = self.run(["systemctl", "start", "--no-block", UNIT], 10)
        if result.returncode == 0:
            return 202, {"started": True}
        return 500, {"started": False, "error": result.stderr.strip()}

    def cancel(self) -> Tuple[int, Dict[str, Any]]:
        if self.phase() == "idle":
            return 409, {"cancelled": False, "reason": "not running"}
        state_dir = Path(self.config.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        with open(state_dir / "lock", "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                try:
                    phase = (state_dir / "phase").read_text(encoding="utf-8").strip()
                except OSError:
                    phase = "waiting"
                if phase == "stopping":
                    return 409, {"cancelled": False, "reason": "too late"}
                (state_dir / "cancel").write_text("cancel", encoding="utf-8")
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        return 200, {"cancelled": True}


def _home(user: str) -> str:
    try:
        return pwd.getpwnam(user).pw_dir
    except KeyError:
        return f"/home/{user}"


class RemoteSessions:
    """One Claude Remote Control session per directory, as a transient unit in
    POWER_USER's systemd manager (linger keeps that manager up). --spawn=session
    makes it a single session that exits when it ends; nothing restarts it."""

    def __init__(self, config: ListenerConfig, run: Runner = _run,
                 sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.time,
                 home: Optional[str] = None) -> None:
        self.config = config
        self.run = run
        self.sleep = sleep
        self.now = now
        self.home = home or _home(config.user)

    def _systemctl(self, *args: str) -> List[str]:
        return ["systemctl", f"--machine={self.config.user}@", "--user", *args]

    @staticmethod
    def unit(name: str) -> str:
        return f"claude-rc-{name}.service"

    def _active(self, name: str) -> bool:
        result = self.run(self._systemctl("is-active", self.unit(name)), 10)
        return result.stdout.strip() in ("active", "activating")

    def _journal(self, name: str, since: float) -> str:
        result = self.run(["journalctl", f"_SYSTEMD_USER_UNIT={self.unit(name)}",
                           f"--since=@{int(since)}", "--no-pager", "-o", "cat"], 10)
        return result.stdout

    def status(self) -> Tuple[int, Dict[str, Any]]:
        return 200, {"running": [name for name in RC_DIRS if self._active(name)]}

    def start(self, name: str) -> Tuple[int, Dict[str, Any]]:
        if name not in RC_DIRS:
            return 400, {"error": f"unknown dir {name!r}"}
        session = f"homelab-{name}"
        if self._active(name):
            return 409, {"started": False, "name": session}
        claude = self.config.claude or os.path.join(self.home, ".local/bin/claude")
        directory = os.path.join(self.home, RC_DIRS[name]) if RC_DIRS[name] else self.home
        path = f"{self.home}/.local/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin"
        since = self.now() - 1
        argv = ["systemd-run", f"--machine={self.config.user}@", "--user",
                f"--unit={self.unit(name)}", "--collect", f"--working-directory={directory}",
                f"--setenv=PATH={path}", claude, "remote-control", "--spawn=session",
                "--name", session]
        result = self.run(argv, 15)
        if result.returncode != 0:
            return 500, {"started": False, "error": (result.stderr or result.stdout).strip()[-300:]}
        deadline = self.now() + RC_WAIT_SECONDS
        while True:
            output = self._journal(name, since)
            match = RC_URL.search(output)
            if match:
                return 200, {"started": True, "name": session, "url": match.group(0)}
            if not self._active(name):
                lines = [line for line in output.splitlines() if line.strip()][-5:]
                return 500, {"started": False, "error": "\n".join(lines) or "exited at once"}
            if self.now() >= deadline:
                return 200, {"started": True, "name": session, "url": None}
            self.sleep(1)

    def stop(self, name: str) -> Tuple[int, Dict[str, Any]]:
        if name not in RC_DIRS:
            return 400, {"error": f"unknown dir {name!r}"}
        if not self._active(name):
            return 404, {"stopped": False}
        result = self.run(self._systemctl("stop", self.unit(name)), 30)
        if result.returncode != 0:
            return 500, {"stopped": False, "error": result.stderr.strip()[-300:]}
        return 200, {"stopped": True}


def make_handler(config: ListenerConfig, control: PowerControl,
                 sessions: Optional[RemoteSessions] = None) -> type:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: Dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if self.client_address[0] not in config.allowed_ips:
                self._reply(403, {"error": "forbidden"})
                return False
            token = self.headers.get("X-Relay-Token", "")
            if not config.token or not hmac.compare_digest(token.encode(), config.token.encode()):
                self._reply(401, {"error": "invalid token"})
                return False
            return True

        def do_GET(self) -> None:
            if not self._authorized():
                return
            if self.path == "/power/status":
                status, body = control.status()
                self._reply(status, body)
            elif self.path == "/rc/status" and sessions:
                status, body = sessions.status()
                self._reply(status, body)
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._authorized():
                return
            if self.path == "/power/off":
                status, body = control.start_off()
                self._reply(status, body)
            elif self.path == "/power/cancel":
                status, body = control.cancel()
                self._reply(status, body)
            elif sessions and urlsplit(self.path).path in ("/rc/start", "/rc/stop"):
                url = urlsplit(self.path)
                name = parse_qs(url.query).get("dir", ["home"])[0]
                action = sessions.start if url.path == "/rc/start" else sessions.stop
                status, body = action(name)
                self._reply(status, body)
            else:
                self._reply(404, {"error": "not found"})

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[kaya-power-listener] " + (fmt % args) + "\n")

    return Handler


def make_server(config: ListenerConfig, control: PowerControl,
                sessions: Optional[RemoteSessions] = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((config.host, config.port), make_handler(config, control, sessions))


def main() -> None:
    config = ListenerConfig.load(os.environ.get("KAYA_POWER_ENV", "/etc/kaya-power.env"), os.environ)
    if not config.token:
        sys.stderr.write("KAYA_RELAY_TOKEN is empty: refusing to serve an unauthenticated listener\n")
        sys.exit(2)
    make_server(config, PowerControl(config), RemoteSessions(config)).serve_forever()


if __name__ == "__main__":
    main()
