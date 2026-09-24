"""Tests for src.chat.relay (parsing, deduplication, backlog tracking)."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.chat.relay import BacklogTracker, RelayEnvelope, RelayState, parse_envelope


# -- RelayState --------------------------------------------------------------


class TestRelayState:
    def test_fresh_state_not_duplicate(self, tmp_path: Path) -> None:
        state = RelayState(str(tmp_path / "state.json"))
        assert state.is_duplicate("any", 1) is False
        assert state.journal_id == ""
        assert state.last_applied_seq == 0

    def test_mark_and_duplicate(self, tmp_path: Path) -> None:
        state = RelayState(str(tmp_path / "state.json"))
        state.mark_applied("j1", 5)
        assert state.is_duplicate("j1", 5) is True
        assert state.is_duplicate("j1", 3) is True
        assert state.is_duplicate("j1", 6) is False
        assert state.journal_id == "j1"
        assert state.last_applied_seq == 5

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        path = str(tmp_path / "state.json")
        RelayState(path).mark_applied("j2", 10)
        fresh = RelayState(path)
        assert fresh.is_duplicate("j2", 10) is True
        assert fresh.journal_id == "j2"
        assert fresh.last_applied_seq == 10

    def test_different_journal_not_duplicate(self, tmp_path: Path) -> None:
        state = RelayState(str(tmp_path / "state.json"))
        state.mark_applied("j1", 5)
        # Different journal_id is NOT a duplicate
        assert state.is_duplicate("j_other", 5) is False

    def test_old_journal_seqs_no_longer_duplicates(self, tmp_path: Path) -> None:
        state = RelayState(str(tmp_path / "state.json"))
        state.mark_applied("j1", 5)
        state.mark_applied("j2", 1)
        # j1 seqs are no longer duplicates
        assert state.is_duplicate("j1", 3) is False
        assert state.is_duplicate("j1", 5) is False
        # j2 seq 1 IS a duplicate
        assert state.is_duplicate("j2", 1) is True
        assert state.is_duplicate("j2", 2) is False

    def test_missing_state_file(self, tmp_path: Path) -> None:
        state = RelayState(str(tmp_path / "nonexistent.json"))
        assert state.is_duplicate("any", 1) is False
        assert state.journal_id == ""
        assert state.last_applied_seq == 0

    def test_corrupt_state_file(self, tmp_path: Path) -> None:
        path = str(tmp_path / "state.json")
        with open(path, "w") as fh:
            fh.write("not json")
        state = RelayState(path)
        assert state.is_duplicate("any", 1) is False

    def test_atomic_write(self, tmp_path: Path) -> None:
        path = str(tmp_path / "state.json")
        state = RelayState(path)
        state.mark_applied("j1", 42)
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")
        data = json.loads(open(path).read())
        assert data["journal_id"] == "j1"
        assert data["last_applied_seq"] == 42


# -- parse_envelope -----------------------------------------------------------


class TestParseEnvelope:
    def test_happy_path(self) -> None:
        body = {
            "kaya_relay": {
                "journal_id": "uuid-1",
                "seq": 17,
                "received_at": 1790000000.5,
                "replayed": True,
                "deferred_reply": False,
                "backlog_remaining": 42,
            },
            "event": {"event": "message", "payload": {}},
        }
        env = parse_envelope(body)
        assert env.journal_id == "uuid-1"
        assert env.seq == 17
        assert env.received_at == 1790000000.5
        assert env.replayed is True
        assert env.deferred_reply is False
        assert env.backlog_remaining == 42
        assert env.event == {"event": "message", "payload": {}}

    def test_defaults_applied(self) -> None:
        body = {
            "kaya_relay": {"journal_id": "uuid-2", "seq": 1},
            "event": {"foo": "bar"},
        }
        env = parse_envelope(body)
        assert env.received_at == 0.0
        assert env.replayed is False
        assert env.deferred_reply is False
        assert env.backlog_remaining == 0

    def test_missing_kaya_relay(self) -> None:
        with pytest.raises(ValueError, match="kaya_relay"):
            parse_envelope({"event": {}})

    def test_missing_event(self) -> None:
        with pytest.raises(ValueError, match="event"):
            parse_envelope({"kaya_relay": {"journal_id": "x", "seq": 1}})

    def test_seq_string_raises(self) -> None:
        body = {
            "kaya_relay": {"journal_id": "x", "seq": "3"},
            "event": {},
        }
        with pytest.raises(ValueError):
            parse_envelope(body)

    def test_seq_zero_raises(self) -> None:
        body = {
            "kaya_relay": {"journal_id": "x", "seq": 0},
            "event": {},
        }
        with pytest.raises(ValueError):
            parse_envelope(body)

    def test_seq_bool_raises(self) -> None:
        body = {
            "kaya_relay": {"journal_id": "x", "seq": True},
            "event": {},
        }
        with pytest.raises(ValueError):
            parse_envelope(body)

    def test_empty_journal_id_raises(self) -> None:
        body = {
            "kaya_relay": {"journal_id": "", "seq": 1},
            "event": {},
        }
        with pytest.raises(ValueError):
            parse_envelope(body)

    def test_body_not_dict_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_envelope("not a dict")  # type: ignore


# -- BacklogTracker -----------------------------------------------------------


class TestBacklogTracker:
    def test_not_draining_initially(self) -> None:
        tracker = BacklogTracker(stale_after_seconds=600.0, now=lambda: 0.0)
        assert tracker.draining() is False

    def test_draining_after_update(self) -> None:
        clock = 0.0
        tracker = BacklogTracker(stale_after_seconds=600.0, now=lambda: clock)
        tracker.update(3)
        assert tracker.draining() is True

    def test_not_draining_after_clear(self) -> None:
        clock = 0.0
        tracker = BacklogTracker(stale_after_seconds=600.0, now=lambda: clock)
        tracker.update(3)
        tracker.update(0)
        assert tracker.draining() is False

    def test_stale_causes_false(self) -> None:
        clock = 0.0
        tracker = BacklogTracker(stale_after_seconds=600.0, now=lambda: clock)
        tracker.update(3)
        # Advance clock past stale_after
        clock = 601.0
        assert tracker.draining() is False

    def test_not_stale_within_window(self) -> None:
        clock = 0.0
        tracker = BacklogTracker(stale_after_seconds=600.0, now=lambda: clock)
        tracker.update(3)
        clock = 599.0
        assert tracker.draining() is True
