"""Durable append-only log of messages the bot has seen, per scope.

The bot only *replies* when addressed (a DM, or an @mention in a group), but the
group's ordinary chatter is the most valuable thing it could remember. So every
message it observes is logged here, before the reply gate — the ingester then
folds them into the vector store on its own schedule.

One file per scope (``src/chat/scope.py``), which keeps a DM's messages
physically separate from the group's on disk, not merely filtered at query time.

Lines are JSON objects:
    {"id", "chat_id", "sender", "text", "timestamp", "scope"}

``id`` is derived from the chat and the WhatsApp message id, so re-logging the
same message — which WAHA does freely when it replays history after a reconnect —
produces the same id and the ingester upserts rather than duplicates.
"""
from __future__ import annotations

import hashlib
import threading
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Serialises `append`. Module level, not per instance: two MessageLog objects can
# be built over the same directory (the adapter makes one, anything else that
# logs makes another), and a per-instance lock would not stop them interleaving.
#
# It covers the dedupe check as well as the write, because `uid in _seen`
# followed by `_seen.add(uid)` is a check-then-act that two threads can both
# pass — and since WAHA delivers every message twice, two threads handling one
# message id is the normal case here rather than a rare one.
#
# Being straight about what this does NOT fix: neither a torn write nor a double
# insert could be reproduced without it (300 threads, 40KB records). `O_APPEND`
# writes are atomic on Linux, and the GIL makes the check-then-act window very
# hard to hit. This is cheap atomicity for a function that should obviously have
# it, not a repair for the corruption found below — see `_ensure_newline`.
_append_lock = threading.Lock()


def message_uid(chat_id: str, message_id: str) -> str:
    """Stable id for one message. Same inputs always give the same id."""
    raw = f"{chat_id}\x00{message_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:80] or "unknown"


def _ensure_newline(path: Path) -> None:
    """Terminate a truncated last line before appending after it.

    One line of the live group log was found holding 159 characters of one record
    followed by the whole of the next:

        ..."timestamp": 1787737868, "{"id": "2c914ed2e7...", "chat_id": ...

    The first record was cut off mid-write with no trailing newline — a process
    killed during a flush is the likeliest cause — and the next append then
    continued that same line. `read` skips anything it cannot parse, so BOTH
    messages were lost: the truncated one, and the perfectly good one that landed
    behind it.

    That second loss is the avoidable one. Starting a fresh line costs one byte
    read per append and means a partial write can only ever damage its own
    record.

    Never raises: this is a guard, and it must not be the reason a message goes
    unlogged.
    """
    try:
        if not path.exists() or path.stat().st_size == 0:
            return
        with open(path, "rb+") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
                logger.warning("%s did not end in a newline — a previous write was "
                               "cut short; starting a fresh line", path.name)
    except OSError as exc:  # noqa: BLE001
        logger.debug("could not check the tail of %s: %s", path, exc)


class MessageLog:
    """Append-only per-scope JSONL log of observed messages."""

    def __init__(self, base_dir: str = "data/live_messages"):
        self.base_dir = Path(base_dir)
        if not self.base_dir.is_absolute():
            self.base_dir = Path(__file__).parent.parent.parent / base_dir
        # Remember ids written this process so a WAHA backlog replay does not
        # rewrite thousands of lines we already have.
        self._seen: set = set()

    def path_for(self, scope: str) -> Path:
        return self.base_dir / f"{_safe(scope)}.jsonl"

    def scopes(self) -> List[str]:
        """The real scope strings present on disk.

        Deliberately NOT the filenames: a scope like ``dm:ec904910`` is stored as
        ``dm_ec904910.jsonl`` because ``:`` is awkward in a filename. Returning the
        stem would make the ingester stamp chunks with ``dm_ec904910`` while
        retrieval computes ``dm:ec904910`` — the chunks would be written and then
        never match a query. So read the scope back out of the records.
        """
        if not self.base_dir.exists():
            return []
        found: List[str] = []
        for path in sorted(self.base_dir.glob("*.jsonl")):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        scope = json.loads(line).get("scope")
                        if scope and scope not in found:
                            found.append(scope)
                        break
            except (OSError, json.JSONDecodeError):
                continue
        return found

    def append(
        self,
        *,
        chat_id: str,
        message_id: str,
        sender: str,
        text: str,
        timestamp: Optional[int],
        scope: str,
        reply_to_id: str = "",
        reply_to_text: str = "",
        sender_id: str = "",
        sender_phone: str = "",
    ) -> bool:
        """Log one message. Returns False if it was already logged this process.

        ``reply_to_*`` record what a reply was answering. Roughly a third of the
        group's messages are four words or fewer and most of those are replies,
        so without the edge the corpus is full of fragments like "Ao contrário de
        outros" — unreadable to a human and an invitation to invent for anything
        extracting facts from it. Only a short excerpt is stored: the parent is
        already in this log as its own record.

        ``sender_id``/``sender_phone`` are who actually sent it. Only the
        resolved display name used to be kept, and a display name is chosen by
        the person and shared with other people: one member was mapped to
        another's name for weeks, and every message he sent was logged, retrieved
        and attributed to the wrong man with no way to tell afterwards. Two
        members answering to the same first name had the same problem. The number
        is the identity; the name is a lookup that can be corrected later, but
        only if the number was written down at the time.
        """
        if not text or not text.strip():
            return False
        uid = message_uid(chat_id, message_id or text[:64])
        with _append_lock:
            if uid in self._seen:
                return False
            self._seen.add(uid)
            return self._write(uid, chat_id, sender, text, timestamp, scope,
                               reply_to_id, reply_to_text, sender_id, sender_phone)

    def _write(self, uid, chat_id, sender, text, timestamp, scope,
               reply_to_id, reply_to_text, sender_id, sender_phone) -> bool:
        """Serialise and append one record. Callers must hold ``_append_lock``."""

        record = {
            "id": uid,
            "chat_id": chat_id,
            "sender": sender,
            "text": text,
            "timestamp": int(timestamp or 0),
            "scope": scope,
        }
        if sender_id:
            record["sender_id"] = sender_id
        if sender_phone:
            record["sender_phone"] = sender_phone
        if reply_to_id or reply_to_text:
            record["reply_to_id"] = reply_to_id
            record["reply_to_text"] = reply_to_text
        try:
            path = self.path_for(scope)
            path.parent.mkdir(parents=True, exist_ok=True)
            _ensure_newline(path)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except OSError as exc:
            logger.warning("Could not log message to %s: %s", self.base_dir, exc)
            return False

    def read(self, scope: str, after_ts: int = 0) -> Iterator[Dict[str, Any]]:
        """Yield logged messages for a scope newer than ``after_ts``, in order."""
        path = self.path_for(scope)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if int(rec.get("timestamp") or 0) > after_ts:
                        yield rec
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)
