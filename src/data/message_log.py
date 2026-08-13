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
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


def message_uid(chat_id: str, message_id: str) -> str:
    """Stable id for one message. Same inputs always give the same id."""
    raw = f"{chat_id}\x00{message_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:80] or "unknown"


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
        if uid in self._seen:
            return False
        self._seen.add(uid)

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
