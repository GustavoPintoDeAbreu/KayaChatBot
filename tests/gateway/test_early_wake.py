"""The gateway's early wake: accept a time the PC chose, send the packet when due."""
import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.early_wake import EarlyWake
from src.gateway.homelab import WAKE_REQUEST_NAME
from src.gateway.journal import Journal
from src.gateway.monitor import PcMonitor, PcState
from src.gateway.schedule import PowerSchedule

POWER = {"timezone": "Europe/Lisbon", "wake_time": "07:00", "wol_lead_minutes": 5,
         "shutdown": {"sun": "23:00", "mon": "23:00", "tue": "23:00", "wed": "23:00",
                      "thu": "23:00", "fri": "02:00", "sat": "02:00"}}

TZ = ZoneInfo("Europe/Lisbon")


def _at(day, hour, minute=0) -> datetime.datetime:
    """October 2026, Europe/Lisbon: Wednesday is the 21st, before the 25th DST change."""
    return datetime.datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def _epoch(moment: datetime.datetime) -> float:
    return moment.timestamp()


class Clock:
    def __init__(self, value: float):
        self.value = value

    def __call__(self) -> float:
        return self.value


@pytest.fixture
def rig(tmp_path):
    journal = Journal(str(tmp_path / "j.sqlite3"), str(tmp_path / "media"))
    clock = Clock(_epoch(_at(21, 23, 30)))
    monitor = PcMonitor("http://pc.test:7860", now=clock, probe=lambda: PcState.OFFLINE)
    schedule = PowerSchedule.from_config(POWER)
    data_dir = str(tmp_path)
    wake = EarlyWake(journal, monitor, schedule, data_dir, now=clock)
    return wake, journal, monitor, clock, data_dir


def test_accept_keeps_an_honoured_wake_across_a_restart(rig):
    wake, journal, _, clock, data_dir = rig
    requested = _epoch(_at(22, 3, 0))
    assert wake.accept(requested) == requested
    again = EarlyWake(journal, PcMonitor("http://pc.test:7860", now=clock,
                                         probe=lambda: PcState.OFFLINE),
                      PowerSchedule.from_config(POWER), data_dir, now=clock)
    assert again.pending() == requested


@pytest.mark.parametrize("bad", [_epoch(_at(22, 6, 30)), _epoch(_at(21, 22, 0)),
                                 True, "abc", None, float("nan")])
def test_accept_rejects_and_clears(rig, bad):
    wake, _, _, _, _ = rig
    assert wake.accept(_epoch(_at(22, 3, 0))) is not None
    assert wake.accept(bad) is None
    assert wake.pending() is None


def test_tick_before_due_does_nothing(rig):
    wake, _, _, clock, data_dir = rig
    requested = _epoch(_at(22, 3, 0))
    wake.accept(requested)
    wake.tick()
    assert not (Path(data_dir) / WAKE_REQUEST_NAME).exists()
    assert wake.pending() == requested


def test_tick_when_due_and_pc_off_requests_a_wake(rig):
    wake, _, monitor, clock, data_dir = rig
    requested = _epoch(_at(22, 3, 0))
    wake.accept(requested)
    monitor.observe(PcState.OFFLINE)
    clock.value = requested + 5
    wake.tick()
    assert (Path(data_dir) / WAKE_REQUEST_NAME).exists()
    assert wake.pending() is None


def test_tick_when_due_and_pc_on_drops_it(rig):
    wake, _, monitor, clock, data_dir = rig
    requested = _epoch(_at(22, 3, 0))
    wake.accept(requested)
    monitor.observe(PcState.ONLINE)
    clock.value = requested + 5
    wake.tick()
    assert not (Path(data_dir) / WAKE_REQUEST_NAME).exists()
    assert wake.pending() is None


def test_tick_long_past_due_is_dropped(rig):
    wake, _, _, clock, data_dir = rig
    requested = _epoch(_at(22, 3, 0))
    wake.accept(requested)
    clock.value = requested + 7200
    wake.tick()
    assert not (Path(data_dir) / WAKE_REQUEST_NAME).exists()
    assert wake.pending() is None
