"""The scheduled shutdown honouring the idea pipeline's wake request, run against fake tools."""
import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tests.power.test_kaya_shutdown_manual import (BASH, FLOCK, SCRIPT, rig,
                                                   _run_script, _fakes, _poweroffs)

pytestmark = pytest.mark.skipif(not BASH or not FLOCK, reason="needs bash and flock")

TZ = ZoneInfo("Europe/Lisbon")


def _at(day, hour, minute=0) -> datetime.datetime:
    """October 2026, Europe/Lisbon: Wednesday is the 21st, before the 25th DST change."""
    return datetime.datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def _epoch(moment: datetime.datetime) -> int:
    return int(moment.timestamp())


@pytest.fixture
def early(rig):
    env, tmp_path = rig
    env["POWER_NOW_EPOCH"] = str(_epoch(_at(21, 23, 30)))
    env["POWER_WAKE_REQUEST_FILE"] = str(tmp_path / "wake-at")
    env["POWER_EARLY_WAKE_FILE"] = str(tmp_path / "var" / "early-wake")
    return env, tmp_path


def _rtc(tmp_path):
    """The -t value of the last rtcwake call, or nothing."""
    values = [rest.split(" -t ", 1)[1].split()[0] for name, rest in _fakes(tmp_path)
              if name == "rtcwake" and " -t " in rest]
    return values[-1] if values else ""


def _going_down(tmp_path):
    """The curl lines that hit /pc/going-down, in log order."""
    return [rest for name, rest in _fakes(tmp_path)
            if name == "curl" and "/pc/going-down" in rest]


def test_scheduled_shutdown_wakes_at_the_requested_time(early):
    env, tmp_path = early
    (tmp_path / "wake-at").write_text(f"{_epoch(_at(22, 3, 0))}\n")
    result = _run_script(env)
    assert result.returncode == 0
    assert len(_poweroffs(tmp_path)) == 1
    assert _rtc(tmp_path) == str(_epoch(_at(22, 3, 0)))
    assert _going_down(tmp_path), "the going-down notice was never sent"
    for line in _going_down(tmp_path):
        assert f'"wake_at": {_epoch(_at(22, 3, 0))}' in line
    assert (tmp_path / "var" / "early-wake").read_text().strip() == str(_epoch(_at(22, 3, 0)))
    flat = [f"{name} {rest}" for name, rest in _fakes(tmp_path)]
    going_down = next(i for i, line in enumerate(flat) if "/pc/going-down" in line)
    rtcwake = next(i for i, line in enumerate(flat) if line.startswith("rtcwake"))
    poweroff = next(i for i, line in enumerate(flat) if line.startswith("systemctl poweroff"))
    assert going_down < rtcwake < poweroff


def test_request_near_the_morning_is_ignored(early):
    env, tmp_path = early
    (tmp_path / "wake-at").write_text(f"{_epoch(_at(22, 6, 30))}\n")
    (tmp_path / "var").mkdir()
    (tmp_path / "var" / "early-wake").write_text("123\n")
    result = _run_script(env)
    assert result.returncode == 0
    assert len(_poweroffs(tmp_path)) == 1
    assert _rtc(tmp_path) == str(_epoch(_at(22, 7, 0)))
    assert not any("wake_at" in line for line in _going_down(tmp_path))
    assert not (tmp_path / "var" / "early-wake").exists()


def test_request_before_the_window_is_ignored(early):
    env, tmp_path = early
    env["POWER_NOW_EPOCH"] = str(_epoch(_at(21, 22, 30)))
    (tmp_path / "wake-at").write_text(f"{_epoch(_at(22, 3, 0))}\n")
    result = _run_script(env)
    assert result.returncode == 0
    assert _rtc(tmp_path) == str(_epoch(_at(22, 7, 0)))
    assert not any("wake_at" in line for line in _going_down(tmp_path))


def test_no_request_is_unchanged(early):
    env, tmp_path = early
    result = _run_script(env)
    assert result.returncode == 0
    assert len(_poweroffs(tmp_path)) == 1
    assert _rtc(tmp_path) == str(_epoch(_at(22, 7, 0)))
    assert not any("wake_at" in line for line in _going_down(tmp_path))
    assert not (tmp_path / "var" / "early-wake").exists()


def test_manual_mode_ignores_the_request(early):
    env, tmp_path = early
    (tmp_path / "wake-at").write_text(f"{_epoch(_at(22, 3, 0))}\n")
    result = _run_script(env, "--manual")
    assert result.returncode == 0
    assert _rtc(tmp_path) == str(_epoch(_at(22, 7, 0)))
    assert not any("wake_at" in line for line in _going_down(tmp_path))


def test_dry_run_reports_the_early_wake(early):
    env, tmp_path = early
    (tmp_path / "wake-at").write_text(f"{_epoch(_at(22, 3, 0))}\n")
    result = _run_script(env, "--dry-run")
    assert result.returncode == 0
    assert "dry run: the PC is idle and would shut down now" in result.stdout
    assert "for the idea pipeline" in result.stdout
    assert _poweroffs(tmp_path) == []


def test_after_early_wake_without_a_file_does_nothing(early):
    env, tmp_path = early
    result = _run_script(env, "--after-early-wake")
    assert result.returncode == 0
    assert "not an early wake" in result.stdout
    assert _poweroffs(tmp_path) == []


def test_after_early_wake_in_the_morning_stays_on(early):
    env, tmp_path = early
    (tmp_path / "var").mkdir()
    (tmp_path / "var" / "early-wake").write_text(f"{_epoch(_at(22, 3, 0))}\n")
    env["POWER_NOW_EPOCH"] = str(_epoch(_at(22, 7, 30)))
    result = _run_script(env, "--after-early-wake")
    assert result.returncode == 0
    assert "the day has started" in result.stdout
    assert _poweroffs(tmp_path) == []
    assert not (tmp_path / "var" / "early-wake").exists()


def test_after_early_wake_powers_off_and_rearms(early):
    env, tmp_path = early
    (tmp_path / "var").mkdir()
    (tmp_path / "var" / "early-wake").write_text(f"{_epoch(_at(22, 3, 0))}\n")
    env["POWER_NOW_EPOCH"] = str(_epoch(_at(22, 3, 10)))
    (tmp_path / "wake-at").write_text(f"{_epoch(_at(22, 5, 0))}\n")
    result = _run_script(env, "--after-early-wake")
    assert result.returncode == 0
    assert len(_poweroffs(tmp_path)) == 1
    assert _rtc(tmp_path) == str(_epoch(_at(22, 5, 0)))
    assert (tmp_path / "var" / "early-wake").read_text().strip() == str(_epoch(_at(22, 5, 0)))


def test_after_early_wake_gives_up_before_the_morning(early):
    env, tmp_path = early
    (tmp_path / "var").mkdir()
    (tmp_path / "var" / "early-wake").write_text(f"{_epoch(_at(22, 3, 0))}\n")
    env["POWER_NOW_EPOCH"] = str(_epoch(_at(22, 3, 10)))
    env["FAKE_PGREP"] = "Runner.Worker"
    env["POWER_STOP_BEFORE_WAKE_MINUTES"] = "100000"
    result = _run_script(env, "--after-early-wake")
    assert result.returncode == 0
    assert "giving up tonight" in result.stdout
    assert _poweroffs(tmp_path) == []


def _audit_starts(tmp_path):
    return [rest for name, rest in _fakes(tmp_path) if name == "systemctl" and "--user start" in rest]


def test_the_security_audit_runs_once_as_the_last_job(early):
    env, tmp_path = early
    result = _run_script(env)
    assert result.returncode == 0
    assert len(_audit_starts(tmp_path)) == 1 and "lab-audit.service" in _audit_starts(tmp_path)[0]
    assert "starting the security audit" in result.stdout
    assert len(_poweroffs(tmp_path)) == 1
    flat = [f"{name} {rest}" for name, rest in _fakes(tmp_path)]
    start = next(i for i, line in enumerate(flat) if "--user start" in line)
    poweroff = next(i for i, line in enumerate(flat) if line.startswith("systemctl poweroff"))
    assert start < poweroff


def test_a_running_audit_holds_the_poweroff(early):
    env, tmp_path = early
    env["FAKE_AUDIT_STATE"] = "activating"                   # a oneshot unit while it runs
    env["POWER_STOP_BEFORE_WAKE_MINUTES"] = "100000"         # past the deadline: one check, then give up
    result = _run_script(env)
    assert "a security audit is running" in result.stdout and "giving up tonight" in result.stdout
    assert _poweroffs(tmp_path) == []


def test_the_audit_cap_lets_the_poweroff_go(early):
    env, tmp_path = early
    env["FAKE_AUDIT_STATE"] = "activating"
    env["POWER_AUDIT_MAX_MINUTES"] = "0"
    result = _run_script(env)
    assert len(_poweroffs(tmp_path)) == 1 and "security audit is running" not in result.stdout


def test_no_audit_unit_means_no_audit(early):
    env, tmp_path = early
    env["POWER_AUDIT_UNIT"] = ""
    result = _run_script(env)
    assert _audit_starts(tmp_path) == [] and len(_poweroffs(tmp_path)) == 1
    assert "security audit" not in result.stdout
