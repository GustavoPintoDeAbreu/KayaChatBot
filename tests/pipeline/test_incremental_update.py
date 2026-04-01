"""
Tests for the incremental update pipeline (src/data/incremental_update.py).

Validates:
- SHA-256 hash computation (determinism, sensitivity, unicode safety)
- Loading existing messages: last_timestamp detection and hash set population
- Date-based filtering: messages strictly older than last_timestamp are dropped
- Hash-based deduplication: exact duplicates are skipped
- Appending new messages to all_messages_cleaned.jsonl
- Cross-file deduplication within a single run
- pipeline_metadata.json creation and updates
- No-op behaviour when all input messages are already present
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Make src/ importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.incremental_update import (
    compute_message_hash,
    load_existing_messages,
    load_metadata,
    run_incremental_update,
    save_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(timestamp: str, sender: str, text: str, source: str = "whatsapp") -> dict:
    return {"timestamp": timestamp, "sender": sender, "text": text, "source": source}


def _write_jsonl(path: Path, messages: list) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for msg in messages:
            fh.write(json.dumps(msg, ensure_ascii=False) + "\n")


def _write_wpp_txt(path: Path, messages: list) -> None:
    """Serialize *messages* to a minimal WhatsApp TXT export."""
    with open(path, "w", encoding="utf-8") as fh:
        for msg in messages:
            dt = datetime.fromisoformat(msg["timestamp"])
            # WhatsApp format: M/D/YY, H:MM - Sender: text
            ts_str = (
                f"{dt.month}/{dt.day}/{dt.strftime('%y')}, "
                f"{dt.hour}:{dt.strftime('%M')}"
            )
            fh.write(f"{ts_str} - {msg['sender']}: {msg['text']}\n")


# ---------------------------------------------------------------------------
# Unit tests — compute_message_hash
# ---------------------------------------------------------------------------

class TestComputeMessageHash:
    def test_deterministic(self):
        msg = _msg("2023-01-01T10:00:00", "Alice", "Hello")
        assert compute_message_hash(msg) == compute_message_hash(msg)

    def test_different_text_yields_different_hash(self):
        assert compute_message_hash(
            _msg("2023-01-01T10:00:00", "Alice", "Hello")
        ) != compute_message_hash(
            _msg("2023-01-01T10:00:00", "Alice", "World")
        )

    def test_different_sender_yields_different_hash(self):
        assert compute_message_hash(
            _msg("2023-01-01T10:00:00", "Alice", "Hi")
        ) != compute_message_hash(
            _msg("2023-01-01T10:00:00", "Bob", "Hi")
        )

    def test_sha256_value(self):
        msg = _msg("2023-01-01T10:00:00", "Alice", "Hello")
        expected = hashlib.sha256(b"2023-01-01T10:00:00AliceHello").hexdigest()
        assert compute_message_hash(msg) == expected

    def test_unicode_characters_consistent(self):
        """Non-ASCII characters (accents, emoji) produce a stable 64-char hex hash."""
        msg = _msg("2023-01-01T10:00:00", "João", "Olá 👋 café")
        h = compute_message_hash(msg)
        # Calling twice must return the same value
        assert compute_message_hash(msg) == h
        # Must be a valid 64-char lowercase hex string
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Unit tests — load_existing_messages
# ---------------------------------------------------------------------------

class TestLoadExistingMessages:
    def test_returns_empty_when_file_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.data.incremental_update.OUTPUT_CLEANED",
            tmp_path / "nonexistent.jsonl",
        )
        msgs, hashes, ts = load_existing_messages()
        assert msgs == []
        assert hashes == set()
        assert ts is None

    def test_loads_messages_and_last_timestamp(self, tmp_path, monkeypatch):
        cleaned = tmp_path / "all_messages_cleaned.jsonl"
        messages = [
            _msg("2023-01-01T10:00:00", "Alice", "First"),
            _msg("2023-01-02T11:00:00", "Bob", "Second"),
        ]
        _write_jsonl(cleaned, messages)
        monkeypatch.setattr("src.data.incremental_update.OUTPUT_CLEANED", cleaned)

        msgs, hashes, ts = load_existing_messages()
        assert len(msgs) == 2
        assert ts == "2023-01-02T11:00:00"
        assert len(hashes) == 2


# ---------------------------------------------------------------------------
# Integration tests — run_incremental_update
# ---------------------------------------------------------------------------

class TestRunIncrementalUpdate:
    def _patch(self, monkeypatch, tmp_path):
        """Redirect all module-level path constants to tmp_path."""
        monkeypatch.setattr(
            "src.data.incremental_update.OUTPUT_CLEANED",
            tmp_path / "all_messages_cleaned.jsonl",
        )
        monkeypatch.setattr(
            "src.data.incremental_update.OUTPUT_FINETUNE_CHUNKS",
            tmp_path / "finetune_chunks.jsonl",
        )
        monkeypatch.setattr(
            "src.data.incremental_update.METADATA_FILE",
            tmp_path / "pipeline_metadata.json",
        )
        monkeypatch.setattr("src.data.incremental_update.DATA_DIR", tmp_path)

    def test_new_messages_appended(self, tmp_path, monkeypatch):
        """Messages after last_timestamp are appended; the duplicate is skipped."""
        self._patch(monkeypatch, tmp_path)
        cleaned = tmp_path / "all_messages_cleaned.jsonl"

        existing = [_msg("2023-01-01T10:00:00", "Alice", "Existing")]
        _write_jsonl(cleaned, existing)

        # Export contains the existing message (duplicate) plus two new ones
        wpp_messages = [
            _msg("2023-01-01T10:00:00", "Alice", "Existing"),  # duplicate
            _msg("2023-01-03T09:00:00", "Bob", "New one"),
            _msg("2023-01-04T12:00:00", "Alice", "New two"),
        ]
        wpp_file = tmp_path / "new_chat.txt"
        _write_wpp_txt(wpp_file, wpp_messages)

        result = run_incremental_update([wpp_file], rebuild_db=False)

        assert result is True
        with open(cleaned) as fh:
            lines = [line for line in fh if line.strip()]
        assert len(lines) == 3  # 1 existing + 2 new

    def test_exact_duplicates_skipped_across_two_files(self, tmp_path, monkeypatch):
        """The same message appearing in two input files is added only once."""
        self._patch(monkeypatch, tmp_path)
        cleaned = tmp_path / "all_messages_cleaned.jsonl"
        cleaned.write_text("")  # Start from empty dataset

        shared = _msg("2023-02-01T08:00:00", "Carlos", "Shared message")
        unique_a = _msg("2023-02-02T09:00:00", "Alice", "Only in A")
        unique_b = _msg("2023-02-03T10:00:00", "Bob", "Only in B")

        file_a = tmp_path / "chat_a.txt"
        file_b = tmp_path / "chat_b.txt"
        _write_wpp_txt(file_a, [shared, unique_a])
        _write_wpp_txt(file_b, [shared, unique_b])

        result = run_incremental_update([file_a, file_b], rebuild_db=False)

        assert result is True
        with open(cleaned) as fh:
            lines = [line for line in fh if line.strip()]
        # shared once + unique_a + unique_b = 3
        assert len(lines) == 3

    def test_metadata_created_and_updated(self, tmp_path, monkeypatch):
        """pipeline_metadata.json is created with the correct values."""
        self._patch(monkeypatch, tmp_path)
        cleaned = tmp_path / "all_messages_cleaned.jsonl"
        metadata_file = tmp_path / "pipeline_metadata.json"

        existing = [_msg("2023-01-01T10:00:00", "Alice", "Hi")]
        _write_jsonl(cleaned, existing)

        new_msgs = [_msg("2023-01-05T08:00:00", "Bob", "New")]
        wpp_file = tmp_path / "new.txt"
        _write_wpp_txt(wpp_file, new_msgs)

        run_incremental_update([wpp_file], rebuild_db=False)

        with open(metadata_file) as fh:
            meta = json.load(fh)

        assert meta["last_processed_date"] == "2023-01-05T08:00:00"
        assert meta["total_messages"] == 2
        assert len(meta["processing_history"]) == 1
        assert meta["processing_history"][0]["messages_added"] == 1
        assert meta["processing_history"][0]["chunks_created"] == 1

    def test_no_new_messages_dataset_unchanged(self, tmp_path, monkeypatch):
        """If the export has only old/duplicate messages, the cleaned file is unchanged."""
        self._patch(monkeypatch, tmp_path)
        cleaned = tmp_path / "all_messages_cleaned.jsonl"

        existing = [_msg("2023-06-01T10:00:00", "Alice", "Latest")]
        _write_jsonl(cleaned, existing)

        old_msgs = [_msg("2023-01-01T09:00:00", "Bob", "Very old message")]
        wpp_file = tmp_path / "old_export.txt"
        _write_wpp_txt(wpp_file, old_msgs)

        result = run_incremental_update([wpp_file], rebuild_db=False)

        assert result is True
        # Content must be unchanged: still exactly the original 1 message
        with open(cleaned) as fh:
            lines = [line for line in fh if line.strip()]
        assert len(lines) == 1
        msg_back = json.loads(lines[0])
        assert msg_back["text"] == "Latest"
