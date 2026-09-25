"""SQLite journal for the Pi gateway.

Every WhatsApp event received by the gateway is written to a local SQLite
database.  A separate worker reads pending events, forwards them to the GPU
PC, and updates their state.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


@dataclass
class JournalEvent:
    """A row from the events table, with ``body_json`` deserialised."""

    seq: int
    event_type: str
    chat_id: str
    sender_id: str
    message_id: str
    wa_ts: Optional[int]
    received_at: float
    event: Dict[str, Any]
    media_path: str
    media_mime: str
    media_status: str
    addressed: bool
    backlog: bool
    attempts: int
    last_error: str


_EVENT_COLUMNS = """seq, event_type, chat_id, sender_id, message_id, wa_ts, received_at,
    body_json, media_path, media_mime, media_status, addressed, backlog, attempts, last_error"""


def _to_event(row: tuple) -> JournalEvent:
    """Build a ``JournalEvent`` from a row selected with ``_EVENT_COLUMNS``."""
    return JournalEvent(
        seq=row[0], event_type=row[1], chat_id=row[2], sender_id=row[3],
        message_id=row[4], wa_ts=row[5], received_at=row[6], event=json.loads(row[7]),
        media_path=row[8], media_mime=row[9], media_status=row[10],
        addressed=bool(row[11]), backlog=bool(row[12]), attempts=row[13],
        last_error=row[14],
    )


class Journal:
    """Thread-safe SQLite journal for gateway events."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS events (
          seq INTEGER PRIMARY KEY AUTOINCREMENT,
          dedup_key TEXT NOT NULL UNIQUE,
          event_type TEXT NOT NULL,
          chat_id TEXT NOT NULL DEFAULT '',
          sender_id TEXT NOT NULL DEFAULT '',
          message_id TEXT NOT NULL DEFAULT '',
          wa_ts INTEGER,
          received_at REAL NOT NULL,
          body_json TEXT NOT NULL,
          media_path TEXT NOT NULL DEFAULT '',
          media_mime TEXT NOT NULL DEFAULT '',
          media_status TEXT NOT NULL DEFAULT 'none',
          addressed INTEGER NOT NULL DEFAULT 0,
          backlog INTEGER NOT NULL DEFAULT 0,
          auto_replied_at REAL,
          state TEXT NOT NULL DEFAULT 'pending',
          attempts INTEGER NOT NULL DEFAULT 0,
          last_error TEXT NOT NULL DEFAULT '',
          delivered_at REAL
        );
        CREATE INDEX IF NOT EXISTS events_pending ON events(state, seq);
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS autoreplies (
          chat_id TEXT NOT NULL,
          offline_since REAL NOT NULL,
          sent_at REAL NOT NULL,
          PRIMARY KEY (chat_id, offline_since)
        );
    """

    def __init__(self, db_path: str, media_dir: str) -> None:
        """Open (or create) the journal at *db_path* with media in *media_dir*."""
        self._db_path = str(db_path)
        self._media_dir = str(media_dir)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self._media_dir).mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    @property
    def journal_id(self) -> str:
        """A stable uuid4 hex identifier created on first open."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'journal_id'"
            ).fetchone()
            if row is None:
                new_id = uuid.uuid4().hex
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('journal_id', ?)",
                    (new_id,),
                )
                self._conn.commit()
                return new_id
            return row[0]

    def append(
        self,
        event: Dict[str, Any],
        *,
        dedup_key: str,
        event_type: str,
        chat_id: str,
        sender_id: str,
        message_id: str,
        wa_ts: Optional[int],
        received_at: float,
        addressed: bool,
        backlog: bool,
    ) -> Optional[int]:
        """Insert an event.  Returns *seq* or ``None`` when *dedup_key* exists."""
        with self._lock:
            try:
                cur = self._conn.execute(
                    """INSERT INTO events
                        (dedup_key, event_type, chat_id, sender_id, message_id,
                         wa_ts, received_at, body_json, addressed, backlog)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        dedup_key,
                        event_type,
                        chat_id,
                        sender_id,
                        message_id,
                        wa_ts,
                        received_at,
                        json.dumps(event, ensure_ascii=False),
                        int(addressed),
                        int(backlog),
                    ),
                )
                self._conn.commit()
                return cur.lastrowid
            except sqlite3.IntegrityError:
                return None

    def attach_media(self, seq: int, data: bytes, filename: str, mime: str) -> str:
        """Write media bytes to ``<media_dir>/<seq>/<safe_filename>`` atomically.

        Returns the absolute path.
        """
        with self._lock:
            seq_dir = Path(self._media_dir) / str(seq)
            seq_dir.mkdir(parents=True, exist_ok=True)
            safe_name = _safe_filename(filename)
            tmp_path = seq_dir / f"{seq}.tmp"
            tmp_path.write_bytes(data)
            final_path = seq_dir / safe_name
            os.replace(str(tmp_path), str(final_path))
            abs_path = str(final_path.resolve())
            self._conn.execute(
                """UPDATE events
                   SET media_path = ?, media_mime = ?, media_status = 'stored'
                   WHERE seq = ?""",
                (abs_path, mime, seq),
            )
            self._conn.commit()
            return abs_path

    def mark_media_failed(self, seq: int, error: str) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE events
                   SET media_status = 'failed', last_error = ?
                   WHERE seq = ?""",
                (error, seq),
            )
            self._conn.commit()

    def get(self, seq: int) -> Optional[JournalEvent]:
        """Return the ``JournalEvent`` for *seq*, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_EVENT_COLUMNS} FROM events WHERE seq = ?", (seq,)
            ).fetchone()
        return _to_event(row) if row is not None else None

    def next_pending(self) -> Optional[JournalEvent]:
        """Return the lowest-seq pending event, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_EVENT_COLUMNS} FROM events WHERE state = 'pending' "
                "ORDER BY seq ASC LIMIT 1"
            ).fetchone()
        return _to_event(row) if row is not None else None

    def pending_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE state = 'pending'"
            ).fetchone()
            return row[0]

    def record_attempt(self, seq: int, error: str) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE events
                   SET attempts = attempts + 1, last_error = ?
                   WHERE seq = ?""",
                (error, seq),
            )
            self._conn.commit()

    def mark_delivered(self, seq: int, at: float) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE events
                   SET state = 'delivered', delivered_at = ?
                   WHERE seq = ?""",
                (at, seq),
            )
            self._conn.commit()

    def newer_addressed_from(self, seq: int, chat_id: str, sender_id: str) -> bool:
        """Return ``True`` if a row with a greater seq has the same chat_id and
        sender_id, addressed=1 and backlog=0."""
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*) FROM events
                   WHERE seq > ? AND chat_id = ? AND sender_id = ?
                   AND addressed = 1 AND backlog = 0""",
                (seq, chat_id, sender_id),
            ).fetchone()
            return row[0] > 0

    def mark_auto_replied(self, seq: int, at: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE events SET auto_replied_at = ? WHERE seq = ?",
                (at, seq),
            )
            self._conn.commit()

    def autoreply_sent(self, chat_id: str, offline_since: float) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM autoreplies WHERE chat_id = ? AND offline_since = ?",
                (chat_id, offline_since),
            ).fetchone()
            return row[0] > 0

    def record_autoreply(self, chat_id: str, offline_since: float, at: float) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO autoreplies (chat_id, offline_since, sent_at)
                   VALUES (?, ?, ?)""",
                (chat_id, offline_since, at),
            )
            self._conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row is not None else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO meta (key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, value),
            )
            self._conn.commit()

    def purge_delivered(self, older_than: float) -> Dict[str, int]:
        """Delete delivered rows with ``delivered_at < older_than``.

        Removes their media directories and autoreplies with ``sent_at < older_than``.
        Returns ``{"events": n, "media_dirs": m}``.
        """
        with self._lock:
            # Gather media paths before deleting
            rows = self._conn.execute(
                "SELECT seq, media_path FROM events WHERE state = 'delivered' AND delivered_at < ?",
                (older_than,),
            ).fetchall()
            seqs = [row[0] for row in rows]
            media_paths = [row[1] for row in rows if row[1]]
            if seqs:
                placeholders = ",".join("?" * len(seqs))
                self._conn.execute(
                    f"DELETE FROM events WHERE seq IN ({placeholders})", seqs
                )

            self._conn.execute(
                "DELETE FROM autoreplies WHERE sent_at < ?", (older_than,)
            )
            self._conn.commit()

        removed_dirs = 0
        for media_path in media_paths:
            seq_dir = Path(media_path).parent
            if seq_dir.is_dir():
                shutil.rmtree(seq_dir, ignore_errors=True)
                removed_dirs += 1
        return {"events": len(seqs), "media_dirs": removed_dirs}

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            pending = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE state = 'pending'"
            ).fetchone()[0]
            oldest_row = self._conn.execute(
                "SELECT received_at FROM events WHERE state = 'pending' ORDER BY seq ASC LIMIT 1"
            ).fetchone()
            oldest_pending = oldest_row[0] if oldest_row else None

        db_bytes = 0
        for suffix in ("", "-wal", "-shm"):
            db_file = Path(self._db_path + suffix)
            if db_file.exists():
                db_bytes += db_file.stat().st_size

        media_bytes = sum(entry.stat().st_size for entry in Path(self._media_dir).rglob("*")
                          if entry.is_file())

        return {
            "events": total,
            "pending": pending,
            "db_bytes": db_bytes,
            "media_bytes": media_bytes,
            "oldest_pending_received_at": oldest_pending,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _safe_filename(name: str) -> str:
    """Keep only ``[A-Za-z0-9._-]`` and no leading dots; fall back to ``media``.

    Leading dots are stripped so that ``..`` (or a hidden name) can never become
    a path component of its own.
    """
    safe = _SAFE_FILENAME.sub("_", name or "").lstrip(".")
    return safe or "media"
