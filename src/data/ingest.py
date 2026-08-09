"""Incremental ingestion: fold newly-seen messages into the vector store.

Replaces the "re-extract and rebuild everything" cycle. Two properties matter:

**Idempotent.** Every chunk id is derived from the messages it contains, so
running the same ingest twice upserts the same rows instead of duplicating them.
That is what makes a crashed run, or WAHA replaying its backlog, safe to repeat.

**Scoped.** Chunks are written with the scope of the chat they came from
(``src/chat/scope.py``), so a DM is retrievable only inside that DM while the
group's history stays readable everywhere.

Runs on a watermark per scope, kept in ``data/ingest_state.json``: catch-up on
boot handles "the bot was down and missed things", and a periodic pass keeps
memory fresh while it runs. Never runs at message time — embedding on the GPU
while answering would compete with the reply.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.message_log import MessageLog

logger = logging.getLogger(__name__)

COLLECTION = "kaya_conversations"
DEFAULT_STATE = "data/ingest_state.json"


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts or 0), tz=timezone.utc).replace(tzinfo=None).isoformat()


def chunk_uid(scope: str, message_ids: List[str]) -> str:
    """Id derived from the messages in the chunk — same content, same id.

    This is what makes ingestion idempotent: re-chunking the same messages
    produces the same id, so the write is an upsert rather than a duplicate.
    """
    raw = (scope + "|" + ",".join(message_ids)).encode("utf-8")
    return "live_" + hashlib.sha256(raw).hexdigest()[:24]


class IngestState:
    """Per-scope watermark: the newest message timestamp already ingested."""

    def __init__(self, path: str = DEFAULT_STATE):
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = Path(__file__).parent.parent.parent / path
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def watermark(self, scope: str) -> int:
        return int((self._data.get("scopes", {}) or {}).get(scope, {}).get("last_ts", 0))

    def set_watermark(self, scope: str, ts: int, ingested: int) -> None:
        scopes = self._data.setdefault("scopes", {})
        entry = scopes.setdefault(scope, {})
        entry["last_ts"] = int(ts)
        entry["last_run"] = datetime.now(timezone.utc).isoformat()
        entry["total_ingested"] = int(entry.get("total_ingested", 0)) + int(ingested)
        self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning("Could not save ingest state to %s: %s", self.path, exc)


def build_chunks(
    messages: List[Dict[str, Any]],
    scope: str,
    max_messages: int = 16,
    max_chars: int = 1800,
) -> List[Dict[str, Any]]:
    """Group consecutive messages into retrievable chunks.

    Mirrors the shape ``build_vector_db.py`` produces for the historical export —
    same metadata keys — so both kinds of chunk are retrieved and formatted
    identically. Adds ``scope`` and ``source: live``.
    """
    chunks: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    chars = 0

    def flush() -> None:
        nonlocal current, chars
        if not current:
            return
        lines, participants, ids = [], [], []
        for m in current:
            sender = (m.get("sender") or "Alguém").strip()
            lines.append(f"{sender}: {m.get('text','').strip()}")
            if sender not in participants:
                participants.append(sender)
            ids.append(m["id"])
        start, end = current[0].get("timestamp", 0), current[-1].get("timestamp", 0)
        chunks.append({
            "id": chunk_uid(scope, ids),
            "text": "\n".join(lines),
            "metadata": {
                "participants": ",".join(participants),
                "mentioned": "",
                "message_count": len(current),
                "token_count": int(chars / 4),
                "timestamp_start": _iso(start),
                "timestamp_end": _iso(end),
                "scope": scope,
                "source": "live",
            },
        })
        current, chars = [], 0

    for msg in messages:
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        current.append(msg)
        chars += len(text)
        if len(current) >= max_messages or chars >= max_chars:
            flush()
    flush()
    return chunks


class Ingester:
    """Reads the message log and upserts new chunks into the vector store."""

    def __init__(self, config: Dict[str, Any], encoder=None, collection=None):
        self.config = config
        rag = config.get("rag", {}) or {}
        self.db_path = rag.get("db_path", "./data/rag_db")
        self.embedding_model = rag.get("embedding_model", "BAAI/bge-m3")
        self.log = MessageLog(
            (config.get("whatsapp", {}) or {}).get("message_log_dir", "data/live_messages")
        )
        self.state = IngestState(
            (config.get("whatsapp", {}) or {}).get("ingest_state_file", DEFAULT_STATE)
        )
        self._encoder = encoder
        self._collection = collection

    # ── lazy resources ───────────────────────────────────────────────────────
    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model, trust_remote_code=True)
        return self._encoder

    @property
    def collection(self):
        if self._collection is None:
            import chromadb

            path = Path(self.db_path)
            if not path.is_absolute():
                path = Path(__file__).parent.parent.parent / self.db_path
            client = chromadb.PersistentClient(path=str(path))
            self._collection = client.get_or_create_collection(
                name=COLLECTION, metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    # ── the work ─────────────────────────────────────────────────────────────
    def ingest_scope(self, scope: str) -> Dict[str, Any]:
        """Ingest everything logged for one scope since its watermark."""
        since = self.state.watermark(scope)
        messages = list(self.log.read(scope, after_ts=since))
        if not messages:
            return {"scope": scope, "messages": 0, "chunks": 0, "since": since}

        chunks = build_chunks(messages, scope)
        if chunks:
            texts = [c["text"] for c in chunks]
            embeddings = self.encoder.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            ).tolist()
            # upsert, not add: re-running must not duplicate.
            self.collection.upsert(
                ids=[c["id"] for c in chunks],
                documents=texts,
                metadatas=[c["metadata"] for c in chunks],
                embeddings=embeddings,
            )

        newest = max(int(m.get("timestamp") or 0) for m in messages)
        self.state.set_watermark(scope, newest, len(chunks))
        return {
            "scope": scope, "messages": len(messages), "chunks": len(chunks),
            "since": since, "watermark": newest,
        }

    def ingest_all(self) -> List[Dict[str, Any]]:
        """Ingest every scope that has logged messages."""
        results = []
        for scope in self.log.scopes():
            try:
                results.append(self.ingest_scope(scope))
            except Exception as exc:  # noqa: BLE001 — one bad scope must not stop the rest
                logger.warning("Ingest failed for scope %s: %s", scope, exc)
                results.append({"scope": scope, "error": str(exc)})
        return results


def run_ingest(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convenience entry point used by the boot catch-up and the periodic pass."""
    t0 = time.time()
    results = Ingester(config).ingest_all()
    total_chunks = sum(r.get("chunks", 0) for r in results)
    total_msgs = sum(r.get("messages", 0) for r in results)
    if total_msgs:
        print(
            f"✓ Ingested {total_msgs} message(s) into {total_chunks} chunk(s) "
            f"across {len(results)} scope(s) in {time.time() - t0:.1f}s"
        )
    return results
