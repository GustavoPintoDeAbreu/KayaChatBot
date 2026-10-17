"""The poweroff announcement carrying the early wake kaya-shutdown.sh chose, run against fake tools."""
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tests.power.test_kaya_shutdown_manual import BASH, FLOCK, REPO, _write_fakes

pytestmark = pytest.mark.skipif(not BASH or not FLOCK, reason="needs bash and flock")

SCRIPT = REPO / "deploy" / "power" / "kaya-going-down.sh"

TZ = ZoneInfo("Europe/Lisbon")


def _at(day, hour, minute=0) -> datetime.datetime:
    """October 2026, Europe/Lisbon: Wednesday is the 21st, before the 25th DST change."""
    return datetime.datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def _epoch(moment: datetime.datetime) -> int:
    return int(moment.timestamp())


@pytest.fixture
def rig(tmp_path):
    _write_fakes(tmp_path)
    env_file = tmp_path / "kaya-power.env"
    env_file.write_text("KAYA_RELAY_TOKEN=relay\n")
    env = {
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        "KAYA_POWER_ENV": str(env_file),
        "POWER_USER": "root",
        "POWER_REPO": str(REPO),
        "POWER_PYTHON": sys.executable,
        "POWER_GATEWAY_URL": "http://gw.test:8088",
        "POWER_PC_APP_URL": "http://app.test:7860",
        "POWER_REPLY_WAIT_SECONDS": "0",
        "POWER_NOW_EPOCH": str(_epoch(_at(21, 23, 40))),
        "POWER_EARLY_WAKE_FILE": str(tmp_path / "var" / "early-wake"),
        "FAKE_LOG": str(tmp_path / "fakes.log"),
    }
    return env, tmp_path


def _run(env):
    return subprocess.run([BASH, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)


def _fakes(tmp_path):
    log = tmp_path / "fakes.log"
    if not log.exists():
        return []
    return [line.split("\t", 1) for line in log.read_text().splitlines() if line]


def _rtc(tmp_path):
    """The -t value of the last rtcwake call, or nothing."""
    values = [rest.split(" -t ", 1)[1].split()[0] for name, rest in _fakes(tmp_path)
              if name == "rtcwake" and " -t " in rest]
    return values[-1] if values else ""


def _going_down(tmp_path):
    """The curl lines that hit /pc/going-down, in log order."""
    return [rest for name, rest in _fakes(tmp_path)
            if name == "curl" and "/pc/going-down" in rest]


def test_going_down_carries_the_early_wake(rig):
    env, tmp_path = rig
    (tmp_path / "var").mkdir()
    (tmp_path / "var" / "early-wake").write_text(f"{_epoch(_at(22, 3, 0))}\n")
    result = _run(env)
    assert result.returncode == 0
    assert _rtc(tmp_path) == str(_epoch(_at(22, 3, 0)))
    assert _going_down(tmp_path), "the going-down notice was never sent"
    for line in _going_down(tmp_path):
        assert f'"wake_at": {_epoch(_at(22, 3, 0))}' in line
    # The marker is consumed by the boot timer, not by the announcement.
    assert (tmp_path / "var" / "early-wake").exists()


def test_a_past_early_wake_is_ignored(rig):
    env, tmp_path = rig
    (tmp_path / "var").mkdir()
    (tmp_path / "var" / "early-wake").write_text(f"{_epoch(_at(21, 22, 0))}\n")
    result = _run(env)
    assert result.returncode == 0
    assert _rtc(tmp_path) == str(_epoch(_at(22, 7, 0)))
    assert not any("wake_at" in line for line in _going_down(tmp_path))


def test_without_an_early_file_nothing_changes(rig):
    env, tmp_path = rig
    result = _run(env)
    assert result.returncode == 0
    assert _rtc(tmp_path) == str(_epoch(_at(22, 7, 0)))
    assert not any("wake_at" in line for line in _going_down(tmp_path))
