"""Tests for ``src.gateway.journal``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.journal import Journal


# ── fixtures ─────────────────────────────────────────────────────────────

def _event():
    return {"from": "1234@c.us", "body": "Hello world", "type": "chat"}


@pytest.fixture
def journal(tmp_path: Path):
    j = Journal(str(tmp_path / "journal.db"), str(tmp_path / "media"))
    yield j
    j.close()


# ── append / dedup ───────────────────────────────────────────────────────

class TestAppend:
    def test_returns_increasing_seqs(self, journal):
        seq1 = journal.append(_event(), dedup_key="k1", event_type="message",
                              chat_id="c1", sender_id="s1", message_id="m1",
                              wa_ts=1000, received_at=1.0,
                              addressed=False, backlog=False)
        seq2 = journal.append(_event(), dedup_key="k2", event_type="message",
                              chat_id="c1", sender_id="s1", message_id="m2",
                              wa_ts=1001, received_at=2.0,
                              addressed=False, backlog=False)
        assert seq1 is not None and seq2 is not None
        assert seq2 == seq1 + 1

    def test_dedup_returns_none(self, journal):
        seq1 = journal.append(_event(), dedup_key="same", event_type="message",
                              chat_id="c1", sender_id="s1", message_id="m1",
                              wa_ts=1000, received_at=1.0,
                              addressed=False, backlog=False)
        seq2 = journal.append(_event(), dedup_key="same", event_type="message",
                              chat_id="c1", sender_id="s1", message_id="m2",
                              wa_ts=1001, received_at=2.0,
                              addressed=False, backlog=False)
        assert seq1 is not None
        assert seq2 is None


# ── journal_id ───────────────────────────────────────────────────────────

class TestJournalId:
    def test_stable_across_reopen(self, tmp_path: Path):
        j1 = Journal(str(tmp_path / "journal.db"), str(tmp_path / "media"))
        id1 = j1.journal_id
        j1.close()
        j2 = Journal(str(tmp_path / "journal.db"), str(tmp_path / "media"))
        id2 = j2.journal_id
        j2.close()
        assert id1 == id2
        assert len(id1) == 32
        assert all(c in "0123456789abcdef" for c in id1)


# ── next_pending / mark_delivered ────────────────────────────────────────

class TestPending:
    def test_fifo(self, journal):
        journal.append(_event(), dedup_key="p1", event_type="message",
                       chat_id="c1", sender_id="s1", message_id="m1",
                       wa_ts=1000, received_at=1.0,
                       addressed=False, backlog=False)
        journal.append(_event(), dedup_key="p2", event_type="message",
                       chat_id="c1", sender_id="s1", message_id="m2",
                       wa_ts=1001, received_at=2.0,
                       addressed=False, backlog=False)
        first = journal.next_pending()
        assert first.seq == 1

    def test_mark_delivered_removes_from_pending(self, journal):
        seq = journal.append(_event(), dedup_key="d1", event_type="message",
                             chat_id="c1", sender_id="s1", message_id="m1",
                             wa_ts=1000, received_at=1.0,
                             addressed=False, backlog=False)
        journal.mark_delivered(seq, at=5.0)
        assert journal.next_pending() is None
        assert journal.pending_count() == 0

    def test_pending_count(self, journal):
        journal.append(_event(), dedup_key="c1", event_type="message",
                       chat_id="c1", sender_id="s1", message_id="m1",
                       wa_ts=1000, received_at=1.0,
                       addressed=False, backlog=False)
        journal.append(_event(), dedup_key="c2", event_type="message",
                       chat_id="c1", sender_id="s1", message_id="m2",
                       wa_ts=1001, received_at=2.0,
                       addressed=False, backlog=False)
        assert journal.pending_count() == 2


# ── attach_media ─────────────────────────────────────────────────────────

class TestAttachMedia:
    def test_writes_bytes_and_sanitises_name(self, journal):
        seq = journal.append(_event(), dedup_key="m1", event_type="message",
                             chat_id="c1", sender_id="s1", message_id="m1",
                             wa_ts=1000, received_at=1.0,
                             addressed=False, backlog=False)
        path = journal.attach_media(seq, b"hello media",
                                    "../weird name?.ogg", "audio/ogg")
        assert path is not None
        assert Path(path).exists()
        assert Path(path).read_bytes() == b"hello media"
        evt = journal.get(seq)
        assert evt is not None
        assert evt.media_status == "stored"
        assert evt.media_mime == "audio/ogg"


# ── newer_addressed_from ─────────────────────────────────────────────────

class TestNewerAddressedFrom:
    def test_true_for_later_addressd_non_backlog(self, journal):
        journal.append(_event(), dedup_key="a1", event_type="message",
                       chat_id="c1", sender_id="s1", message_id="m1",
                       wa_ts=1000, received_at=1.0,
                       addressed=False, backlog=False)
        journal.append(_event(), dedup_key="a2", event_type="message",
                       chat_id="c1", sender_id="s1", message_id="m2",
                       wa_ts=1001, received_at=2.0,
                       addressed=True, backlog=False)
        assert journal.newer_addressed_from(1, "c1", "s1") is True

    def test_false_when_no_addressed_row(self, journal):
        journal.append(_event(), dedup_key="b1", event_type="message",
                       chat_id="c1", sender_id="s1", message_id="m1",
                       wa_ts=1000, received_at=1.0,
                       addressed=False, backlog=False)
        assert journal.newer_addressed_from(1, "c1", "s1") is False


# ── autoreply_sent / record_autoreply ────────────────────────────────────

class TestAutoreply:
    def test_per_chat_offline_since(self, journal):
        chat_id = "1234@c.us"
        offline = 100.0
        assert journal.autoreply_sent(chat_id, offline) is False
        journal.record_autoreply(chat_id, offline, at=101.0)
        assert journal.autoreply_sent(chat_id, offline) is True


# ── purge_delivered ──────────────────────────────────────────────────────

class TestPurgeDelivered:
    def test_deletes_old_delivered_and_media(self, journal):
        seq1 = journal.append(_event(), dedup_key="purge1", event_type="message",
                              chat_id="c1", sender_id="s1", message_id="m1",
                              wa_ts=1000, received_at=1.0,
                              addressed=False, backlog=False)
        journal.mark_delivered(seq1, at=10.0)
        journal.attach_media(seq1, b"data", "test.ogg", "audio/ogg")

        journal.append(_event(), dedup_key="keep1", event_type="message",
                       chat_id="c1", sender_id="s1", message_id="m2",
                       wa_ts=1001, received_at=2.0,
                       addressed=False, backlog=False)

        journal.record_autoreply("1234@c.us", offline_since=5.0, at=10.0)

        result = journal.purge_delivered(older_than=20.0)
        assert result["events"] == 1
        assert result["media_dirs"] == 1
        assert journal.pending_count() == 1
        assert journal.get(seq1) is None


# ── stats ────────────────────────────────────────────────────────────────

class TestStats:
    def test_counts(self, journal):
        journal.append(_event(), dedup_key="s1", event_type="message",
                       chat_id="c1", sender_id="s1", message_id="m1",
                       wa_ts=1000, received_at=1.0,
                       addressed=False, backlog=False)
        journal.append(_event(), dedup_key="s2", event_type="message",
                       chat_id="c1", sender_id="s1", message_id="m2",
                       wa_ts=1001, received_at=2.0,
                       addressed=False, backlog=False)
        stats = journal.stats()
        assert stats["events"] == 2
        assert stats["pending"] == 2
        assert stats["db_bytes"] > 0
        assert stats["media_bytes"] >= 0
        assert stats["oldest_pending_received_at"] == 1.0


# ── record_attempt ───────────────────────────────────────────────────────

class TestRecordAttempt:
    def test_increments(self, journal):
        seq = journal.append(_event(), dedup_key="at1", event_type="message",
                             chat_id="c1", sender_id="s1", message_id="m1",
                             wa_ts=1000, received_at=1.0,
                             addressed=False, backlog=False)
        journal.record_attempt(seq, "timeout")
        journal.record_attempt(seq, "retry fail")
        evt = journal.get(seq)
        assert evt is not None
        assert evt.attempts == 2
        assert evt.last_error == "retry fail"


# ── mark_auto_replied ────────────────────────────────────────────────────

class TestMarkAutoReplied:
    def test_sets_timestamp(self, journal):
        seq = journal.append(_event(), dedup_key="ar1", event_type="message",
                             chat_id="c1", sender_id="s1", message_id="m1",
                             wa_ts=1000, received_at=1.0,
                             addressed=False, backlog=False)
        journal.mark_auto_replied(seq, at=99.0)
        evt = journal.get(seq)
        assert evt is not None


# ── get_meta / set_meta ──────────────────────────────────────────────────

class TestMeta:
    def test_get_default(self, journal):
        assert journal.get_meta("missing", "def") == "def"
        journal.set_meta("key1", "val1")
        assert journal.get_meta("key1") == "val1"


# ── mark_media_failed ────────────────────────────────────────────────────

class TestMediaFailed:
    def test_sets_status(self, journal):
        seq = journal.append(_event(), dedup_key="mf1", event_type="message",
                             chat_id="c1", sender_id="s1", message_id="m1",
                             wa_ts=1000, received_at=1.0,
                             addressed=False, backlog=False)
        journal.mark_media_failed(seq, "disk full")
        evt = journal.get(seq)
        assert evt is not None
        assert evt.media_status == "failed"


def test_attach_media_never_escapes_the_seq_dir(tmp_path):
    from src.gateway.journal import Journal

    journal = Journal(str(tmp_path / "j.sqlite3"), str(tmp_path / "media"))
    seq = journal.append({"event": "message"}, dedup_key="dots", event_type="message",
                         chat_id="c", sender_id="s", message_id="m", wa_ts=1,
                         received_at=1.0, addressed=False, backlog=False)
    for hostile in ("..", ".", "../../etc/passwd", ""):
        stored = Path(journal.attach_media(seq, b"x", hostile, "text/plain"))
        assert stored.parent == (tmp_path / "media" / str(seq)).resolve()
