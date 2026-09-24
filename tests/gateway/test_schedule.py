"""Tests for ``src.gateway.schedule``."""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

import pytest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.schedule import PowerSchedule


# The config from plan section 1.
_POWER = {
    "timezone": "Europe/Lisbon",
    "wake_time": "07:00",
    "wol_lead_minutes": 5,
    "shutdown": {
        "sun": "23:00",
        "mon": "23:00",
        "tue": "23:00",
        "wed": "23:00",
        "thu": "23:00",
        "fri": "02:00",
        "sat": "02:00",
    },
}

TZ = ZoneInfo("Europe/Lisbon")


def _sched() -> PowerSchedule:
    return PowerSchedule.from_config(_POWER)


def _naive(year, month, day, hour, minute):
    """Helper: create a naive datetime, interpreted as Europe/Lisbon."""
    return datetime.datetime(year, month, day, hour, minute)


# ── shutdown_oncalendar ──────────────────────────────────────────────────

class TestShutdownOncalendar:
    def test_groups(self):
        result = _sched().shutdown_oncalendar()
        # 02:00 fires on Sat,Sun (fri/sat evenings with pre-noon time → next day)
        # 23:00 fires on Sun,Mon,Tue,Wed,Thu (same-day evenings)
        # Consecutive run Mon..Thu collapses; Sun is separate.
        assert result == ["Sat,Sun 02:00", "Mon..Thu,Sun 23:00"]


class TestWakeOncalendar:
    def test_format(self):
        assert _sched().wake_oncalendar() == "*-*-* 06:55:00"


# ── is_scheduled_on ──────────────────────────────────────────────────────

class TestIsScheduledOn:
    def test_wed_1200(self):
        """Wednesday 2026-09-23 12:00 → on (last event was a wake)."""
        moment = _naive(2026, 9, 23, 12, 0)
        assert _sched().is_scheduled_on(moment) is True

    def test_wed_2330(self):
        """Wednesday 23:30 → off (shutdown at 23:00)."""
        moment = _naive(2026, 9, 23, 23, 30)
        assert _sched().is_scheduled_on(moment) is False

    def test_fri_2330(self):
        """Friday 2026-09-25 23:30 → on (last event was a wake)."""
        moment = _naive(2026, 9, 25, 23, 30)
        assert _sched().is_scheduled_on(moment) is True

    def test_sat_0300(self):
        """Saturday 2026-09-26 03:00 → off (shutdown fired Sat 02:00)."""
        moment = _naive(2026, 9, 26, 3, 0)
        assert _sched().is_scheduled_on(moment) is False

    def test_sun_2330(self):
        """Sunday 2026-09-27 23:30 → off (shutdown at 23:00)."""
        moment = _naive(2026, 9, 27, 23, 30)
        assert _sched().is_scheduled_on(moment) is False


# ── next_shutdown ────────────────────────────────────────────────────────

class TestNextShutdown:
    def test_wed_1200(self):
        """Wed 12:00 → next shutdown Wed 23:00."""
        moment = _naive(2026, 9, 23, 12, 0)
        nxt = _sched().next_shutdown(moment)
        assert nxt == datetime.datetime(2026, 9, 23, 23, 0, tzinfo=TZ)

    def test_fri_2330(self):
        """Fri 23:30 → next shutdown Sat 02:00 (which is actually Sat evening
        for the Fri shutdown, so fires at Sat 02:00)."""
        moment = _naive(2026, 9, 25, 23, 30)
        nxt = _sched().next_shutdown(moment)
        # Friday's shutdown is at 02:00 the next day (Saturday)
        assert nxt == datetime.datetime(2026, 9, 26, 2, 0, tzinfo=TZ)


# ── next_wake ────────────────────────────────────────────────────────────

class TestNextWake:
    def test_wed_1200(self):
        """Wed 12:00 → next wake Thu 07:00."""
        moment = _naive(2026, 9, 23, 12, 0)
        nxt = _sched().next_wake(moment)
        assert nxt == datetime.datetime(2026, 9, 24, 7, 0, tzinfo=TZ)

    def test_sat_0300(self):
        """Sat 03:00 → next wake Sat 07:00."""
        moment = _naive(2026, 9, 26, 3, 0)
        nxt = _sched().next_wake(moment)
        assert nxt == datetime.datetime(2026, 9, 26, 7, 0, tzinfo=TZ)


# ── DST ──────────────────────────────────────────────────────────────────

class TestDST:
    def test_next_shutdown_after_fall_back(self):
        """Saturday 2026-10-24 23:30 → next shutdown Sunday 2026-10-25 02:00.

        Clocks go back at 02:00 on Sunday 2026-10-25.
        """
        moment = datetime.datetime(2026, 10, 24, 23, 30, tzinfo=TZ)
        nxt = _sched().next_shutdown(moment)
        assert nxt == datetime.datetime(2026, 10, 25, 2, 0, tzinfo=TZ)
        assert nxt.tzinfo is not None


# ── naive input ──────────────────────────────────────────────────────────

class TestNaiveInput:
    def test_next_wake_naive(self):
        """A naive datetime is treated as Europe/Lisbon."""
        moment = _naive(2026, 9, 23, 12, 0)
        nxt = _sched().next_wake(moment)
        assert nxt.tzinfo == TZ


# ── CLI smoke ────────────────────────────────────────────────────────────

class TestCLISmoke:
    def test_oncalendar_shutdown(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "src.gateway.schedule",
                "--config", str(Path(__file__).parent.parent.parent / "config.yaml"),
                "oncalendar-shutdown",
            ],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert result.returncode == 0
        lines = result.stdout.strip().splitlines()
        assert lines == ["Sat,Sun 02:00", "Mon..Thu,Sun 23:00"]

    def test_oncalendar_wake(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "src.gateway.schedule",
                "--config", str(Path(__file__).parent.parent.parent / "config.yaml"),
                "oncalendar-wake",
            ],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "*-*-* 06:55:00"
