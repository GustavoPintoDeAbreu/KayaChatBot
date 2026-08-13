"""Memory scoping: which chat's history a retrieval is allowed to see.

The bot talks in the Kaya group AND one-to-one with individual members. Before
live ingestion existed this did not matter, because the vector store held nothing
but the historical group export — one flat collection, no filtering, no way to
leak. Ingesting live conversation is exactly what would break that: a DM would
land in the same collection and could surface as "memory" in the group.

The rule is deliberately asymmetric, because shared group memory is the bot's
whole purpose while a DM is private:

    scope              written by                     readable from
    shared             the historical export + the    EVERYWHERE
                       Kaya group itself
    dm:<chat>          a one-to-one conversation      only that DM
    group:<chat>       any other group                only that group

So a DM can recall anything the group ever said (intended), but the group can
never recall what someone told the bot privately (required).

Short-term history is already isolated by ``KeyedSessionMemory`` (one file per
chat); this is the long-term equivalent for the vector store.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional

SHARED = "shared"
DM_PREFIX = "dm:"
GROUP_PREFIX = "group:"

_GROUP_SUFFIX = "@g.us"


def _short(chat_id: str) -> str:
    """Stable, non-reversible-ish short key for a chat id.

    Hashed rather than raw so a phone number is not written into every chunk's
    metadata — the store already holds message text, there is no reason to also
    index it by personal number.
    """
    return hashlib.sha256(chat_id.encode("utf-8")).hexdigest()[:16]


def scope_for_chat(chat_id: Optional[str], shared_chats: Iterable[str] = ()) -> str:
    """The scope new content from ``chat_id`` should be written under.

    ``shared_chats`` lists the chat ids whose content is group-wide memory (the
    Kaya group). Everything else is private to its own chat.
    """
    if not chat_id:
        # No chat context (CLI, benchmarks, the web UI) reads and writes shared.
        return SHARED
    normalized = chat_id.strip()
    if normalized in set(shared_chats or ()):
        return SHARED
    prefix = GROUP_PREFIX if normalized.endswith(_GROUP_SUFFIX) else DM_PREFIX
    return f"{prefix}{_short(normalized)}"


def readable_scopes(current_scope: str) -> List[str]:
    """Scopes a chat in ``current_scope`` is allowed to retrieve from."""
    if current_scope == SHARED:
        return [SHARED]
    return [SHARED, current_scope]


def scope_filter(current_scope: str) -> Dict[str, Any]:
    """A Chroma ``where`` clause restricting retrieval to readable scopes."""
    allowed = readable_scopes(current_scope)
    if len(allowed) == 1:
        return {"scope": allowed[0]}
    return {"scope": {"$in": allowed}}


def is_readable(chunk_scope: Optional[str], current_scope: str) -> bool:
    """Whether a chunk carrying ``chunk_scope`` may be shown in ``current_scope``.

    A chunk with no scope is treated as ``shared``: everything predating scoping
    is the historical group export. The backfill in
    ``scripts/backfill_chunk_scope.py`` makes that explicit in the store.
    """
    return (chunk_scope or SHARED) in readable_scopes(current_scope)


def unregistered_groups(log_dir: Any, shared_chats: Iterable[str] = ()) -> List[str]:
    """Groups the bot has talked in whose history is NOT group-wide memory.

    ``shared_chats`` comes from a gitignored file read once at import, and
    nothing else ever mentions it. When the Kaya group was missing from it, the
    group's own live history was being written to a private scope and image
    editing was refused in the very room it exists for — and the only symptom
    anybody saw was the bot answering "Só faço imagens no grupo, não por aqui",
    in the group.

    A group in this list is not necessarily a mistake: the bot can sit in a chat
    whose history should stay private. So this reports, and the caller warns; it
    never changes behaviour. It is what turns a silent misconfiguration into a
    visible one.
    """
    import json
    from pathlib import Path

    directory = Path(log_dir)
    if not directory.exists():
        return []

    seen = set()
    for path in sorted(directory.glob("*.jsonl")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    chat_id = (json.loads(line) or {}).get("chat_id", "")
                    if isinstance(chat_id, str) and chat_id.endswith(_GROUP_SUFFIX):
                        seen.add(chat_id)
        except Exception:  # noqa: BLE001 — a startup hint is never worth a crash
            continue

    return sorted(c for c in seen if scope_for_chat(c, shared_chats) != SHARED)


def parse_iso(value: Optional[str]) -> Optional[str]:
    """Normalise a timestamp for lexicographic comparison, or None if unusable.

    ISO-8601 strings sort chronologically as text, which is what the recency
    cutoff relies on, so this only needs to reject obviously bad values.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    return text if re.match(r"^\d{4}-\d{2}-\d{2}", text) else None
