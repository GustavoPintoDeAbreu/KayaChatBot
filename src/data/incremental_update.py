"""
Incremental pipeline update utilities.

Provides helpers for filtering, deduplicating, and tracking processed
messages so that re-runs of the data pipeline only handle new content.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def compute_message_hash(timestamp: str, sender: str, content: str) -> str:
    """Return the SHA-256 hex digest of 'timestamp + sender + content'."""
    raw = f"{timestamp}{sender}{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def filter_new_messages(
    messages: List[Dict[str, Any]],
    last_processed_date: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Return only messages whose timestamp is strictly after *last_processed_date*.

    Parameters
    ----------
    messages:
        List of message dicts, each expected to have a ``"timestamp"`` key
        with an ISO-8601 datetime string.
    last_processed_date:
        ISO-8601 datetime string of the most recently processed message.
        If ``None`` or empty, all messages are returned unchanged.

    Returns
    -------
    Filtered list of messages.
    """
    if not last_processed_date:
        return list(messages)

    cutoff = datetime.fromisoformat(last_processed_date)
    new_messages = []
    for msg in messages:
        ts = msg.get("timestamp", "")
        if not ts:
            continue
        try:
            msg_dt = datetime.fromisoformat(ts)
        except ValueError:
            continue
        if msg_dt > cutoff:
            new_messages.append(msg)

    return new_messages


def deduplicate_messages(
    messages: List[Dict[str, Any]],
    known_hashes: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Remove messages whose hash already exists in *known_hashes*.

    Parameters
    ----------
    messages:
        List of message dicts with ``"timestamp"``, ``"sender"``, and
        ``"content"`` keys.
    known_hashes:
        Set of SHA-256 hex strings representing already-processed messages.
        Mutated in-place to include hashes of messages that pass through.
        If ``None``, a fresh set is created.

    Returns
    -------
    (unique_messages, updated_hashes) tuple.
    """
    if known_hashes is None:
        known_hashes = set()

    unique = []
    for msg in messages:
        h = compute_message_hash(
            msg.get("timestamp", ""),
            msg.get("sender", ""),
            msg.get("content", ""),
        )
        if h not in known_hashes:
            known_hashes.add(h)
            unique.append(msg)

    return unique, known_hashes


def load_pipeline_metadata(metadata_file: str) -> Dict[str, Any]:
    """
    Load pipeline metadata from a JSON file.

    Returns an empty dict if the file does not exist.
    """
    path = Path(metadata_file)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_pipeline_metadata(metadata: Dict[str, Any], metadata_file: str) -> None:
    """Persist pipeline metadata to a JSON file, creating parent dirs as needed."""
    path = Path(metadata_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
