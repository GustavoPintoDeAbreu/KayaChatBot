"""Tests for ``src.gateway.autoreply``."""
from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.journal import Journal
from src.gateway.monitor import PcMonitor, PcState
from src.gateway.schedule import PowerSchedule
from src.gateway.autoreply import OfflineResponder, offline_text


# ── fixtures ─────────────────────────────────────────────────────────────

_POWER_BLOCK = {
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


def _make_journal(tmp_path: Path) -> Journal:
    return Journal(str(tmp_path / "journal.db"), str(tmp_path / "media"))


def _make_schedule():
    return PowerSchedule.from_config(_POWER_BLOCK)


def _make_monitor(state: PcState = PcState.OFFLINE) -> PcMonitor:
    monitor = PcMonitor("http://192.168.1.100:7860")
    monitor._state = state
    return monitor


def _make_entry(
    journal: Journal,
    seq: int = 1,
    chat_id: str = "1234@c.us",
    sender_id: "351912345678@c.us" = "351912345678@c.us",
    addressed: bool = True,
    backlog: bool = False,
    message_id: str = "msg001",
    received_at: float = 1000.0,
) -> int:
    return journal.append(
        {"from": sender_id, "body": "hello", "type": "chat"},
        dedup_key=f"dedup-{seq}",
        event_type="message",
        chat_id=chat_id,
        sender_id=sender_id,
        message_id=message_id,
        wa_ts=int(received_at),
        received_at=received_at,
        addressed=addressed,
        backlog=backlog,
    )


# ── offline_text tests ──────────────────────────────────────────────────

class TestOfflineText:
    def test_same_day(self):
        now = datetime.datetime(2026, 10, 3, 14, 0)
        wake = datetime.datetime(2026, 10, 3, 7, 0)
        text = offline_text(wake, now)
        assert "Volto às 07:00" in text
        assert "amanh" not in text
        assert "dia" not in text

    def test_next_day(self):
        now = datetime.datetime(2026, 10, 3, 14, 0)
        wake = datetime.datetime(2026, 10, 4, 7, 0)
        text = offline_text(wake, now)
        assert "amanh" in text
        assert "07:00" in text

    def test_later_date(self):
        now = datetime.datetime(2026, 10, 3, 14, 0)
        wake = datetime.datetime(2026, 10, 13, 7, 0)
        text = offline_text(wake, now)
        assert "dia 13/10" in text

    def test_no_dashes_in_any_variant(self):
        now = datetime.datetime(2026, 10, 3, 14, 0)
        for day_offset in (0, 1, 10):
            wake = datetime.datetime(2026, 10, 3 + day_offset, 7, 0)
            text = offline_text(wake, now)
            assert "–" not in text  # en-dash
            assert "—" not in text  # em-dash


# ── maybe_reply: no reply while reachable ───────────────────────────────

class TestNoReplyWhileReachable:
    def test_no_reply_online(self, tmp_path: Path):
        journal = _make_journal(tmp_path)
        schedule = _make_schedule()
        monitor = _make_monitor(PcState.ONLINE)
        send = MagicMock()
        responder = OfflineResponder(journal, send, schedule, monitor)
        seq = _make_entry(journal)
        result = responder.maybe_reply(journal.get(seq))
        assert result is False
        send.assert_not_called()

    def test_no_reply_app_down(self, tmp_path: Path):
        journal = _make_journal(tmp_path)
        schedule = _make_schedule()
        monitor = _make_monitor(PcState.APP_DOWN)
        send = MagicMock()
        responder = OfflineResponder(journal, send, schedule, monitor)
        seq = _make_entry(journal)
        result = responder.maybe_reply(journal.get(seq))
        assert result is False

    def test_no_reply_offline_under_grace(self, tmp_path: Path):
        journal = _make_journal(tmp_path)
        schedule = _make_schedule()
        monitor = _make_monitor(PcState.OFFLINE)
        monitor._offline_since = time.time() - 30  # 30 s < 90 s grace
        send = MagicMock()
        responder = OfflineResponder(journal, send, schedule, monitor, grace_seconds=90.0)
        seq = _make_entry(journal)
        result = responder.maybe_reply(journal.get(seq))
        assert result is False


# ── maybe_reply: one reply when OFFLINE past grace ──────────────────────

class TestReplyWhenOffline:
    def test_sends_and_records(self, tmp_path: Path):
        journal = _make_journal(tmp_path)
        schedule = _make_schedule()
        now_time = 2000.0
        monitor = _make_monitor(PcState.OFFLINE)
        monitor._offline_since = now_time - 100  # past grace
        send = MagicMock()
        responder = OfflineResponder(
            journal, send, schedule, monitor,
            grace_seconds=90.0,
            now=lambda: now_time,
        )
        seq = _make_entry(journal)
        result = responder.maybe_reply(journal.get(seq))
        assert result is True
        send.assert_called_once()
        args = send.call_args
        assert args[0][0] == "1234@c.us"
        assert "desligado" in args[0][1]
        assert args[0][2] == "msg001"
        assert journal.autoreply_sent("1234@c.us", now_time - 100)

    def test_quotes_message_id(self, tmp_path: Path):
        journal = _make_journal(tmp_path)
        schedule = _make_schedule()
        monitor = _make_monitor(PcState.OFFLINE)
        monitor._offline_since = 1900.0
        send = MagicMock()
        responder = OfflineResponder(journal, send, schedule, monitor)
        seq = _make_entry(journal, message_id="abc123")
        responder.maybe_reply(journal.get(seq))
        assert send.call_args[0][2] == "abc123"


# ── maybe_reply: second addressed message in same chat → no reply ───────

class TestNoSecondReply:
    def test_same_chat_same_period(self, tmp_path: Path):
        journal = _make_journal(tmp_path)
        schedule = _make_schedule()
        monitor = _make_monitor(PcState.OFFLINE)
        monitor._offline_since = 1900.0
        send = MagicMock()
        responder = OfflineResponder(journal, send, schedule, monitor)
        seq1 = _make_entry(journal, seq=1, message_id="m1")
        seq2 = _make_entry(journal, seq=2, message_id="m2", chat_id="1234@c.us")
        responder.maybe_reply(journal.get(seq1))
        result = responder.maybe_reply(journal.get(seq2))
        assert result is False
        assert send.call_count == 1


# ── maybe_reply: different chat → reply ─────────────────────────────────

class TestDifferentChat:
    def test_reply_in_new_chat(self, tmp_path: Path):
        journal = _make_journal(tmp_path)
        schedule = _make_schedule()
        monitor = _make_monitor(PcState.OFFLINE)
        monitor._offline_since = 1900.0
        send = MagicMock()
        responder = OfflineResponder(journal, send, schedule, monitor)
        seq1 = _make_entry(journal, seq=1, chat_id="1234@c.us", message_id="m1")
        seq2 = _make_entry(journal, seq=2, chat_id="5678@c.us", message_id="m2")
        responder.maybe_reply(journal.get(seq1))
        result = responder.maybe_reply(journal.get(seq2))
        assert result is True
        assert send.call_count == 2


# ── maybe_reply: backlog or unaddressed → no reply ──────────────────────

class TestBacklogUnaddressed:
    def test_backlog_no_reply(self, tmp_path: Path):
        journal = _make_journal(tmp_path)
        schedule = _make_schedule()
        monitor = _make_monitor(PcState.OFFLINE)
        send = MagicMock()
        responder = OfflineResponder(journal, send, schedule, monitor)
        seq = _make_entry(journal, addressed=True, backlog=True)
        result = responder.maybe_reply(journal.get(seq))
        assert result is False

    def test_unaddressed_no_reply(self, tmp_path: Path):
        journal = _make_journal(tmp_path)
        schedule = _make_schedule()
        monitor = _make_monitor(PcState.OFFLINE)
        send = MagicMock()
        responder = OfflineResponder(journal, send, schedule, monitor)
        seq = _make_entry(journal, addressed=False)
        result = responder.maybe_reply(journal.get(seq))
        assert result is False


# ── maybe_reply: GOING_DOWN → reply immediately ─────────────────────────

class TestGoingDown:
    def test_going_down_replies_without_grace(self, tmp_path: Path):
        journal = _make_journal(tmp_path)
        schedule = _make_schedule()
        monitor = _make_monitor(PcState.GOING_DOWN)
        monitor._offline_since = 1999.0  # just set
        send = MagicMock()
        responder = OfflineResponder(journal, send, schedule, monitor, grace_seconds=90.0)
        seq = _make_entry(journal)
        result = responder.maybe_reply(journal.get(seq))
        assert result is True


# ── maybe_reply: send_text raising → False and nothing recorded ─────────

class TestSendFailure:
    def test_exception_returns_false(self, tmp_path: Path):
        journal = _make_journal(tmp_path)
        schedule = _make_schedule()
        monitor = _make_monitor(PcState.OFFLINE)
        monitor._offline_since = 1900.0
        send = MagicMock(side_effect=RuntimeError("network"))
        responder = OfflineResponder(journal, send, schedule, monitor)
        seq = _make_entry(journal)
        result = responder.maybe_reply(journal.get(seq))
        assert result is False
        assert not journal.autoreply_sent("1234@c.us", 1900.0)
