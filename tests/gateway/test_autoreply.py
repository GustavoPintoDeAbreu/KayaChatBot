"""The fixed line the Pi sends when the bot cannot answer."""
import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.autoreply import FAULT, FAULT_TEXT, SCHEDULED, OfflineResponder, offline_text
from src.gateway.journal import Journal
from src.gateway.monitor import PcMonitor, PcState
from src.gateway.schedule import PowerSchedule, schedule_sentence

LISBON = ZoneInfo("Europe/Lisbon")
POWER = {"timezone": "Europe/Lisbon", "wake_time": "07:00", "wol_lead_minutes": 5,
         "shutdown": {"sun": "23:00", "mon": "23:00", "tue": "23:00", "wed": "23:00",
                      "thu": "23:00", "fri": "02:00", "sat": "02:00"}}
# Dashes as clause separators; a hyphen inside a word ("respondo-te") is Portuguese.
DASHES = (" - ", "\u2013", "\u2014")


def _epoch(*args) -> float:
    return datetime.datetime(*args, tzinfo=LISBON).timestamp()


# Wednesday 2026-09-23: 12:00 is scheduled on, 23:30 is scheduled off.
DAYTIME = _epoch(2026, 9, 23, 12, 0)
NIGHT = _epoch(2026, 9, 23, 23, 30)


class Clock:
    def __init__(self, value: float):
        self.value = value

    def __call__(self) -> float:
        return self.value


@pytest.fixture
def rig(tmp_path):
    def build(start: float):
        clock = Clock(start)
        monitor = PcMonitor("http://pc:7860", now=clock, probe=lambda: PcState.ONLINE)
        monitor.observe(PcState.ONLINE)
        journal = Journal(str(tmp_path / "j.sqlite3"), str(tmp_path / "media"))
        sent = []
        responder = OfflineResponder(
            journal, lambda chat, text, reply_to=None: sent.append((chat, text, reply_to)),
            PowerSchedule.from_config(POWER), monitor, now=clock)
        return clock, monitor, journal, responder, sent
    return build


def _entry(journal, key, *, chat="group@g.us", addressed=True, backlog=False, kind="message"):
    seq = journal.append({"event": kind, "payload": {"id": key}}, dedup_key=key, event_type=kind,
                         chat_id=chat, sender_id="a@c.us", message_id=key, wa_ts=1,
                         received_at=1.0, addressed=addressed, backlog=backlog)
    return journal.get(seq)


def test_schedule_sentence_follows_the_config():
    assert schedule_sentence(PowerSchedule.from_config(POWER)) == \
        "das 07:00 às 23:00, e até às 02:00 às sextas e sábados"


def test_offline_text_variants_have_no_dashes():
    now = datetime.datetime(2026, 9, 23, 3, 0, tzinfo=LISBON)
    hours = schedule_sentence(PowerSchedule.from_config(POWER))
    same_day = offline_text(now.replace(hour=7), now, hours)
    assert "Volto às 07:00" in same_day and "O meu horário é das 07:00" in same_day
    assert "amanhã às 07:00" in offline_text(now.replace(hour=7) + datetime.timedelta(days=1), now)
    assert "no dia 26/09" in offline_text(now.replace(day=26, hour=7), now)
    for text in (same_day, FAULT_TEXT):
        assert not any(dash in text for dash in DASHES)


def test_online_or_briefly_app_down_stays_silent(rig):
    clock, monitor, journal, responder, sent = rig(DAYTIME)
    assert not responder.maybe_reply(_entry(journal, "m1"))
    monitor.observe(PcState.APP_DOWN)
    clock.value += 120
    assert responder.outage() is None
    assert not responder.maybe_reply(_entry(journal, "m2"))
    assert sent == []


def test_app_down_past_the_grace_is_a_fault(rig):
    clock, monitor, journal, responder, sent = rig(DAYTIME)
    monitor.observe(PcState.APP_DOWN)
    clock.value += 301
    assert responder.outage() == FAULT
    assert responder.maybe_reply(_entry(journal, "m1"))
    assert sent == [("group@g.us", FAULT_TEXT, "m1")]


def test_offline_at_night_gives_the_schedule(rig):
    clock, monitor, journal, responder, sent = rig(NIGHT)
    monitor.observe(PcState.OFFLINE)
    clock.value += 30
    assert responder.outage() is None
    clock.value += 90
    assert responder.outage() == SCHEDULED
    assert responder.maybe_reply(_entry(journal, "m1"))
    assert "O meu horário é das 07:00 às 23:00" in sent[0][1] and "amanhã às 07:00" in sent[0][1]


def test_offline_during_the_day_is_a_fault(rig):
    clock, monitor, journal, responder, sent = rig(DAYTIME)
    monitor.observe(PcState.OFFLINE)
    clock.value += 120
    assert responder.outage() == FAULT


def test_going_down_and_degraded_reply_at_once(rig):
    clock, monitor, journal, responder, sent = rig(DAYTIME)
    monitor.announce_going_down()
    assert responder.outage() == SCHEDULED
    monitor.observe(PcState.ONLINE)
    monitor.observe(PcState.APP_DOWN)
    assert responder.outage() is None
    monitor.mark_degraded()
    assert responder.outage() == FAULT


def test_once_per_chat_per_outage(rig):
    clock, monitor, journal, responder, sent = rig(DAYTIME)
    monitor.observe(PcState.APP_DOWN)
    clock.value += 400
    assert responder.maybe_reply(_entry(journal, "m1"))
    assert not responder.maybe_reply(_entry(journal, "m2"))
    assert responder.maybe_reply(_entry(journal, "m3", chat="other@g.us"))
    monitor.observe(PcState.ONLINE)
    monitor.observe(PcState.APP_DOWN)
    clock.value += 400
    assert responder.maybe_reply(_entry(journal, "m4"))
    assert [chat for chat, _, _ in sent] == ["group@g.us", "other@g.us", "group@g.us"]


def test_unaddressed_backlog_and_reactions_never_get_a_line(rig):
    clock, monitor, journal, responder, sent = rig(DAYTIME)
    monitor.mark_degraded()
    assert not responder.maybe_reply(_entry(journal, "m1", addressed=False))
    assert not responder.maybe_reply(_entry(journal, "m2", backlog=True))
    assert not responder.maybe_reply(_entry(journal, "m3", kind="message.reaction"))
    assert sent == []


def test_a_failed_send_records_nothing(tmp_path):
    clock = Clock(DAYTIME)
    monitor = PcMonitor("http://pc:7860", now=clock)
    monitor.mark_degraded()
    journal = Journal(str(tmp_path / "j.sqlite3"), str(tmp_path / "media"))

    def boom(*args):
        raise RuntimeError("waha down")

    responder = OfflineResponder(journal, boom, PowerSchedule.from_config(POWER), monitor, now=clock)
    entry = _entry(journal, "m1")
    assert not responder.maybe_reply(entry)
    assert not journal.autoreply_sent("group@g.us", monitor.down_since)
