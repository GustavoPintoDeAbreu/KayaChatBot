"""Unit tests for the mixed-rule date capture in generate_knowledge_base.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.generate_knowledge_base import (
    chunk_date_range,
    extract_event_date_hint,
    _update_member_dates,
    _apply_date_fields,
)


class TestChunkDateRange:
    def test_min_max(self):
        msgs = [
            {"timestamp": "2026-02-01T09:00"},
            {"timestamp": "2026-01-05T10:00"},
            {"timestamp": "2026-01-20T12:00"},
        ]
        assert chunk_date_range(msgs) == ("2026-01-05T10:00", "2026-02-01T09:00")

    def test_empty(self):
        assert chunk_date_range([]) == (None, None)
        assert chunk_date_range([{"text": "no ts"}]) == (None, None)


class TestExtractEventDateHint:
    def test_relative_pt(self):
        assert extract_event_date_hint("O Gil partiu o dedo recentemente.") == "recentemente"

    def test_iso_date(self):
        assert extract_event_date_hint("Casou em 2026-05-01 com a Mel.") == "2026-05-01"

    def test_english_relative(self):
        assert extract_event_date_hint("Rafa had his son last month.").lower() == "last month"

    def test_none_when_absent(self):
        assert extract_event_date_hint("Nada temporal aqui.") is None
        assert extract_event_date_hint("") is None


class TestUpdateMemberDates:
    def test_widens_range_and_keeps_latest_hint(self):
        info = {}
        _update_member_dates(info, "2026-01-05T10:00", "2026-01-20T10:00", "ontem")
        _update_member_dates(info, "2025-12-01T10:00", "2026-02-01T10:00", "recentemente")
        assert info["source_date_start"] == "2025-12-01T10:00"
        assert info["source_date_end"] == "2026-02-01T10:00"
        assert info["last_updated"] == "2026-02-01T10:00"
        assert info["event_date_hint"] == "recentemente"

    def test_no_hint_preserves_existing(self):
        info = {"event_date_hint": "ontem"}
        _update_member_dates(info, "2026-01-05T10:00", "2026-01-20T10:00", None)
        assert info["event_date_hint"] == "ontem"


class TestApplyDateFields:
    def test_writes_present_fields(self):
        fact = {"id": "member_gil"}
        _apply_date_fields(fact, {
            "source_date_start": "2026-01-01T00:00",
            "source_date_end": "2026-02-01T00:00",
            "last_updated": "2026-02-01T00:00",
            "event_date_hint": "",
        })
        assert fact["source_date_start"] == "2026-01-01T00:00"
        assert fact["last_updated"] == "2026-02-01T00:00"
        assert "event_date_hint" not in fact  # empty value skipped

    def test_none_is_noop(self):
        fact = {"id": "member_gil"}
        _apply_date_fields(fact, None)
        assert fact == {"id": "member_gil"}
