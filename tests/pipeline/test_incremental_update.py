"""
Unit tests for src/data/incremental_update.py

Covers:
  - SHA-256 content hashing
  - Date-based filtering of messages
  - Deduplication via known-hash sets
  - Pipeline metadata load / save round-trips
  - Edge cases: empty input, all-duplicate input, missing timestamp fields
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.incremental_update import (
    compute_message_hash,
    deduplicate_messages,
    filter_new_messages,
    load_pipeline_metadata,
    save_pipeline_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(timestamp: str, sender: str, content: str) -> dict:
    return {"timestamp": timestamp, "sender": sender, "content": content}


# ---------------------------------------------------------------------------
# test_compute_message_hash_*
# ---------------------------------------------------------------------------

class TestComputeMessageHash:
    def test_hash_is_sha256_of_concatenation(self):
        ts, sender, content = "2024-01-01T10:00:00", "Peter", "hello"
        expected = hashlib.sha256(f"{ts}{sender}{content}".encode()).hexdigest()
        assert compute_message_hash(ts, sender, content) == expected

    def test_hash_is_64_hex_chars(self):
        h = compute_message_hash("2024-01-01T12:00:00", "Gil", "sup")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_inputs_same_hash(self):
        args = ("2024-06-15T09:00:00", "David", "test message")
        assert compute_message_hash(*args) == compute_message_hash(*args)

    def test_different_content_different_hash(self):
        h1 = compute_message_hash("2024-01-01", "Peter", "hello")
        h2 = compute_message_hash("2024-01-01", "Peter", "world")
        assert h1 != h2

    def test_different_sender_different_hash(self):
        h1 = compute_message_hash("2024-01-01", "Gil", "hi")
        h2 = compute_message_hash("2024-01-01", "Peter", "hi")
        assert h1 != h2

    def test_different_timestamp_different_hash(self):
        h1 = compute_message_hash("2024-01-01T10:00:00", "Peter", "hi")
        h2 = compute_message_hash("2024-01-01T11:00:00", "Peter", "hi")
        assert h1 != h2

    def test_empty_fields_returns_valid_hash(self):
        h = compute_message_hash("", "", "")
        assert len(h) == 64


# ---------------------------------------------------------------------------
# test_filter_new_messages_*
# ---------------------------------------------------------------------------

class TestFilterNewMessages:
    def _messages(self):
        return [
            _msg("2024-01-01T10:00:00", "Peter", "first"),
            _msg("2024-01-02T10:00:00", "Gil", "second"),
            _msg("2024-01-03T10:00:00", "David", "third"),
        ]

    def test_filter_new_messages_keeps_messages_after_cutoff(self):
        msgs = self._messages()
        result = filter_new_messages(msgs, "2024-01-01T10:00:00")
        # cutoff is "after", so the exact cutoff timestamp is excluded
        texts = [m["content"] for m in result]
        assert "first" not in texts
        assert "second" in texts
        assert "third" in texts

    def test_filter_new_messages_no_cutoff_returns_all(self):
        msgs = self._messages()
        assert filter_new_messages(msgs, None) == msgs

    def test_filter_new_messages_empty_string_cutoff_returns_all(self):
        msgs = self._messages()
        assert filter_new_messages(msgs, "") == msgs

    def test_filter_new_messages_future_cutoff_returns_empty(self):
        msgs = self._messages()
        result = filter_new_messages(msgs, "2025-01-01T00:00:00")
        assert result == []

    def test_filter_new_messages_all_messages_new(self):
        msgs = self._messages()
        result = filter_new_messages(msgs, "2023-12-31T23:59:59")
        assert len(result) == len(msgs)

    def test_filter_new_messages_empty_input(self):
        assert filter_new_messages([], "2024-01-01T10:00:00") == []

    def test_filter_new_messages_skips_messages_without_timestamp(self):
        msgs = [{"sender": "Gil", "content": "no timestamp"}]
        result = filter_new_messages(msgs, "2024-01-01T10:00:00")
        assert result == []

    def test_filter_new_messages_skips_invalid_timestamps(self):
        msgs = [_msg("not-a-date", "Peter", "bad ts")]
        result = filter_new_messages(msgs, "2024-01-01T10:00:00")
        assert result == []


# ---------------------------------------------------------------------------
# test_deduplicate_messages_*
# ---------------------------------------------------------------------------

class TestDeduplicateMessages:
    def _messages(self):
        return [
            _msg("2024-01-01T10:00:00", "Peter", "hello"),
            _msg("2024-01-02T10:00:00", "Gil", "world"),
            _msg("2024-01-03T10:00:00", "David", "foo"),
        ]

    def test_dedup_empty_known_hashes_keeps_all(self):
        msgs = self._messages()
        result, new_hashes = deduplicate_messages(msgs, set())
        assert len(result) == 3
        assert len(new_hashes) == 3

    def test_dedup_removes_exact_duplicates(self):
        msgs = self._messages()
        # Pre-compute the hash of the first message
        first = msgs[0]
        first_hash = compute_message_hash(
            first["timestamp"], first["sender"], first["content"]
        )
        result, _ = deduplicate_messages(msgs, {first_hash})
        contents = [m["content"] for m in result]
        assert "hello" not in contents
        assert len(result) == 2

    def test_dedup_all_already_processed_returns_empty(self):
        msgs = self._messages()
        existing = {
            compute_message_hash(m["timestamp"], m["sender"], m["content"])
            for m in msgs
        }
        result, _ = deduplicate_messages(msgs, existing)
        assert result == []

    def test_dedup_updates_known_hashes(self):
        msgs = self._messages()
        known: set = set()
        _, updated = deduplicate_messages(msgs, known)
        assert len(updated) == 3

    def test_dedup_none_known_hashes_initialises_fresh_set(self):
        msgs = self._messages()
        result, hashes = deduplicate_messages(msgs, None)
        assert len(result) == 3
        assert isinstance(hashes, set)

    def test_dedup_empty_messages_list(self):
        result, hashes = deduplicate_messages([], set())
        assert result == []
        assert hashes == set()

    def test_dedup_only_duplicates_returns_empty(self):
        msg = _msg("2024-01-01T10:00:00", "Peter", "hi")
        h = compute_message_hash(msg["timestamp"], msg["sender"], msg["content"])
        result, _ = deduplicate_messages([msg, msg], {h})
        assert result == []


# ---------------------------------------------------------------------------
# test_pipeline_metadata_*
# ---------------------------------------------------------------------------

class TestPipelineMetadata:
    def test_save_and_load_round_trip(self, tmp_path):
        meta_file = str(tmp_path / "pipeline_metadata.json")
        metadata = {
            "last_processed_date": "2024-06-15T10:00:00",
            "messages_processed": 1234,
            "run_id": "abc123",
        }
        save_pipeline_metadata(metadata, meta_file)
        loaded = load_pipeline_metadata(meta_file)
        assert loaded == metadata

    def test_load_nonexistent_file_returns_empty_dict(self, tmp_path):
        result = load_pipeline_metadata(str(tmp_path / "nonexistent.json"))
        assert result == {}

    def test_save_creates_parent_directories(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "pipeline_metadata.json")
        save_pipeline_metadata({"key": "value"}, nested)
        assert Path(nested).exists()

    def test_save_overwrites_existing_file(self, tmp_path):
        meta_file = str(tmp_path / "pipeline_metadata.json")
        save_pipeline_metadata({"v": 1}, meta_file)
        save_pipeline_metadata({"v": 2}, meta_file)
        loaded = load_pipeline_metadata(meta_file)
        assert loaded["v"] == 2

    def test_metadata_fields_are_preserved(self, tmp_path):
        meta_file = str(tmp_path / "meta.json")
        metadata = {
            "last_processed_date": "2024-03-01T00:00:00",
            "processed_hashes": ["abc", "def"],
            "total_messages": 500,
            "pipeline_version": "1.0",
        }
        save_pipeline_metadata(metadata, meta_file)
        loaded = load_pipeline_metadata(meta_file)
        assert loaded["last_processed_date"] == "2024-03-01T00:00:00"
        assert loaded["processed_hashes"] == ["abc", "def"]
        assert loaded["total_messages"] == 500
        assert loaded["pipeline_version"] == "1.0"
