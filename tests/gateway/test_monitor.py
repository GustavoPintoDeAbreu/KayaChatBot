"""PcMonitor: how the Pi decides the PC is off, restarting, or going down."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.monitor import PcMonitor, PcState


class Clock:
    def __init__(self, start: float = 1000.0):
        self.value = start

    def __call__(self) -> float:
        return self.value


def _monitor(clock: Clock) -> PcMonitor:
    return PcMonitor("http://192.168.1.149:7860", now=clock)


def test_offline_starts_the_outage_clock():
    clock = Clock()
    monitor = _monitor(clock)
    assert monitor.observe(PcState.OFFLINE) is PcState.OFFLINE
    assert monitor.offline_since == 1000.0
    clock.value = 1050.0
    monitor.observe(PcState.OFFLINE)
    assert monitor.offline_since == 1000.0
    assert monitor.unreachable_for() == 50.0


def test_app_down_is_a_deploy_not_an_outage():
    monitor = _monitor(Clock())
    assert monitor.observe(PcState.APP_DOWN) is PcState.APP_DOWN
    assert monitor.offline_since is None
    assert monitor.unreachable_for() == 0.0


def test_online_clears_everything():
    clock = Clock()
    monitor = _monitor(clock)
    monitor.observe(PcState.OFFLINE)
    monitor.announce_going_down()
    assert monitor.observe(PcState.ONLINE) is PcState.ONLINE
    assert monitor.offline_since is None
    assert monitor.observe(PcState.OFFLINE) is PcState.OFFLINE


def test_going_down_holds_until_the_pc_is_back():
    clock = Clock()
    monitor = _monitor(clock)
    monitor.observe(PcState.ONLINE)
    monitor.announce_going_down()
    assert monitor.state is PcState.GOING_DOWN
    assert monitor.offline_since == 1000.0
    for observed in (PcState.APP_DOWN, PcState.OFFLINE):
        assert monitor.observe(observed) is PcState.GOING_DOWN
    assert monitor.observe(PcState.ONLINE) is PcState.ONLINE


def test_check_uses_the_injected_probe():
    monitor = PcMonitor("http://pc:7860", now=Clock(), probe=lambda: PcState.APP_DOWN)
    assert monitor.check() is PcState.APP_DOWN


def test_down_since_covers_every_kind_of_outage():
    clock = Clock()
    monitor = _monitor(clock)
    monitor.observe(PcState.ONLINE)
    assert monitor.down_since is None
    monitor.observe(PcState.APP_DOWN)
    assert monitor.down_since == 1000.0
    clock.value = 1400.0
    monitor.observe(PcState.OFFLINE)
    assert monitor.down_since == 1000.0 and monitor.down_for() == 400.0
    monitor.observe(PcState.ONLINE)
    assert monitor.down_since is None and monitor.down_for() == 0.0


def test_degraded_holds_until_the_pc_answers():
    monitor = _monitor(Clock())
    monitor.observe(PcState.ONLINE)
    monitor.mark_degraded()
    assert monitor.degraded and monitor.down_since == 1000.0
    monitor.observe(PcState.APP_DOWN)
    assert monitor.degraded
    monitor.observe(PcState.ONLINE)
    assert not monitor.degraded
