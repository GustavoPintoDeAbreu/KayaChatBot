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
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.data.message_log import MessageLog, message_uid

logger = logging.getLogger(__name__)

COLLECTION = "kaya_conversations"
# How far ahead of now a message timestamp may be before it is treated as bogus.
# Generous: clock skew between phones is real, deliberate time travel is not.
_FUTURE_TOLERANCE_S = 3600

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

    def chat_watermark(self, scope: str, chat_id: str) -> int:
        """A chat's own watermark inside its scope; the scope's until it has one.

        The Community's sub-groups share the ``shared`` scope but are chunked
        apart, so each needs its own mark: a quiet chat's unsettled tail must
        not hold back, or re-chunk, a busy one. A chat that has none yet starts
        from the scope's, which is exactly where the single watermark left it.
        """
        entry = (self._data.get("scopes", {}) or {}).get(scope, {}) or {}
        chats = entry.get("chats", {}) or {}
        if chat_id in chats:
            return int(chats[chat_id])
        return int(entry.get("last_ts", 0))

    def set_chat_watermarks(self, scope: str, marks: Dict[str, int], ingested: int) -> None:
        """Store each chat's mark; the scope's own is the lowest of them."""
        scopes = self._data.setdefault("scopes", {})
        entry = scopes.setdefault(scope, {})
        chats = entry.setdefault("chats", {})
        chats.update({chat_id: int(ts) for chat_id, ts in marks.items()})
        entry["last_ts"] = min(int(ts) for ts in chats.values()) if chats else int(entry.get("last_ts", 0))
        entry["last_run"] = datetime.now(timezone.utc).isoformat()
        entry["total_ingested"] = int(entry.get("total_ingested", 0)) + int(ingested)
        self._save()

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


_IMAGE_BLOCK = re.compile(r"\[Imagem:\s*(.*?)\]\s*", re.DOTALL)


def strip_failed_descriptions(text: str) -> str:
    """Drop ``[Imagem: …]`` blocks where the describer said it saw nothing.

    ``vision.describe_bytes`` now refuses to return those, but the message log
    already holds the ones written before that guard existed — and this log is
    what becomes long-term memory. Left alone, "[Imagem: Por favor, fornece o
    vídeo ou a imagem]" is indexed as something a member said.

    A caption alongside a failed description is real and survives; a message that
    was nothing but the failed block returns empty and is skipped by the caller.
    """
    if "[Imagem:" not in (text or ""):
        return text
    from src.chat.vision import looks_like_a_refusal

    return _IMAGE_BLOCK.sub(
        lambda m: "" if looks_like_a_refusal(m.group(1)) else m.group(0), text
    ).strip()


def build_chunks(
    messages: List[Dict[str, Any]],
    scope: str,
    max_messages: int = 16,
    max_chars: int = 1800,
    settle_seconds: int = 0,
    now: Optional[int] = None,
    keep_bot_quote: Optional[Callable[[Dict[str, Any]], bool]] = None,
    channel_of: Optional[Callable[[str], str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Group consecutive messages into retrievable chunks.

    Mirrors the shape ``build_vector_db.py`` produces for the historical export —
    same metadata keys — so both kinds of chunk are retrieved and formatted
    identically. Adds ``scope`` and ``source: live``.

    Returns ``(chunks, consumed_through_ts)``. The second value is the timestamp
    the caller may advance its watermark to, and it is the point of
    ``settle_seconds``: this used to flush an unconditional partial chunk at the
    end of every run. At a two-hour cadence that was harmless, since a partial
    chunk was nearly full anyway. At fifteen minutes it would shatter the index
    into one- and two-message chunks — and those are what ``min_similarity`` and
    ``top_k`` then have to rank, so retrieval quality would quietly fall while
    every timing number stayed green.

    So a trailing partial chunk is only emitted once its newest message has sat
    still for ``settle_seconds``. Otherwise those messages are left unconsumed and
    the watermark stops short of them, and the next run picks them up and builds a
    full chunk. ``settle_seconds=0`` keeps the old flush-everything behaviour, for
    the one-shot CLI path where nothing will come later.

    ``keep_bot_quote`` decides whether a reply to the BOT carries the bot's
    words into memory. The bot's messages are never in the log themselves, so
    the quote is the only way they get in, and on 2026-09-28 that was how its
    roasts were becoming "what the group said". Refused, the line keeps the
    member's side and says only that it answered the bot.

    Every chunk records the ``chat_id`` it came from and, through
    ``channel_of``, the channel's name. The caller hands this one chat at a
    time (``Ingester.ingest_scope``): the Community's sub-groups share a scope,
    and a chunk interleaving the trip channel with general is a chunk about
    neither.
    """
    chunks: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    chars = 0
    consumed_through = 0

    def flush() -> None:
        nonlocal current, chars, consumed_through
        if not current:
            return
        lines, participants, ids = [], [], []
        # The log's own id is a hash of (chat_id, message_id) while reply_to_id
        # is the raw WhatsApp id, so the parent has to be hashed the same way
        # before the two can be compared.
        chunk_ids = {m["id"] for m in current}
        for m in current:
            sender = (m.get("sender") or "Alguém").strip()
            text = strip_failed_descriptions((m.get("text") or "").strip())
            # What a reply was answering, inlined as a prefix. Skipped when the
            # parent is already a line in this same chunk, so a back-and-forth
            # does not get every message stated twice.
            quoted = (m.get("reply_to_text") or "").strip()
            parent_uid = (
                message_uid(m.get("chat_id", ""), m["reply_to_id"])
                if m.get("reply_to_id") else ""
            )
            if (quoted and m.get("reply_to_bot") and keep_bot_quote is not None
                    and not keep_bot_quote(m)):
                lines.append(f"{sender} (a responder ao bot): {text}")
            elif quoted and parent_uid not in chunk_ids:
                lines.append(f'{sender} (a responder a "{quoted}"): {text}')
            else:
                lines.append(f"{sender}: {text}")
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
                "chat_id": str(current[0].get("chat_id") or ""),
                "channel": (channel_of(str(current[0].get("chat_id") or "")) or "")
                if channel_of else "",
            },
        })
        consumed_through = max(consumed_through, int(end or 0))
        current, chars = [], 0

    for msg in messages:
        text = strip_failed_descriptions((msg.get("text") or "").strip())
        if not text:
            continue
        current.append(msg)
        chars += len(text)
        if len(current) >= max_messages or chars >= max_chars:
            flush()

    # The tail: emit it only once it has stopped growing, otherwise leave it for
    # the next run so it can become a full chunk instead of a fragment.
    if current:
        newest = max(int(m.get("timestamp") or 0) for m in current)
        settled = settle_seconds <= 0 or (
            (now if now is not None else int(time.time())) - newest >= settle_seconds
        )
        if settled:
            flush()
    return chunks, consumed_through


class Ingester:
    """Reads the message log and upserts new chunks into the vector store."""

    def __init__(self, config: Dict[str, Any], encoder=None, collection=None):
        self.config = config
        rag = config.get("rag", {}) or {}
        self.db_path = rag.get("db_path", "./data/rag_db")
        self.embedding_model = rag.get("embedding_model", "BAAI/bge-m3")
        self.embedding_device = rag.get("embedding_device") or None
        self.log = MessageLog(
            (config.get("whatsapp", {}) or {}).get("message_log_dir", "data/live_messages")
        )
        self.state = IngestState(
            (config.get("whatsapp", {}) or {}).get("ingest_state_file", DEFAULT_STATE)
        )
        self._encoder = encoder
        self._collection = collection
        icfg = ((config.get("whatsapp", {}) or {}).get("ingest", {}) or {})
        self.settle_seconds = int(float(icfg.get("settle_minutes", 0)) * 60)

    # ── lazy resources ───────────────────────────────────────────────────────
    @property
    def encoder(self):
        if self._encoder is None:
            # The serving process already holds this exact model: the retriever
            # loads BAAI/bge-m3 once into a singleton. Building a second one costs
            # ~2.2GB of the same card the model answers from, and it was being
            # rebuilt on EVERY cycle — twelve cold loads a day at the old cadence,
            # ninety-six at the new one. Borrow the loaded one when there is one.
            try:
                from src.chat.retriever import peek_retriever

                shared = peek_retriever()
                if shared is not None and getattr(shared, "encoder", None) is not None:
                    self._encoder = shared.encoder
                    return self._encoder
            except Exception as exc:  # noqa: BLE001 — fall back to our own
                logger.debug("could not borrow the retriever's encoder: %s", exc)

            from sentence_transformers import SentenceTransformer

            logger.info("loading a private embedder (%s)", self.embedding_model)
            self._encoder = SentenceTransformer(self.embedding_model, trust_remote_code=True,
                                                device=self.embedding_device)
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

    def _bot_quote_filter(self) -> Callable[[Dict[str, Any]], bool]:
        """Keep a quote of the bot only when its ledger entry allows it.

        One read of each chat's ledger per pass, not one per message.
        """
        from src.chat.reply_review import ReplyLedger, keep_quote, reply_key

        ledger = ReplyLedger((self.config.get("whatsapp", {}) or {}).get(
            "replies_dir", "data/whatsapp_replies"))
        cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

        def keep(message: Dict[str, Any]) -> bool:
            chat_id = message.get("chat_id", "")
            if chat_id not in cache:
                cache[chat_id] = ledger.entries(chat_id)
            return keep_quote(cache[chat_id].get(reply_key(message.get("reply_to_text", ""))))

        return keep

    # ── the work ─────────────────────────────────────────────────────────────
    def ingest_scope(self, scope: str) -> Dict[str, Any]:
        """Ingest everything logged for one scope since its watermarks, chat by chat."""
        since = self.state.watermark(scope)
        logged = list(self.log.read(scope, after_ts=since))
        by_chat: Dict[str, List[Dict[str, Any]]] = {}
        for message in logged:
            chat_id = str(message.get("chat_id") or "")
            if int(message.get("timestamp") or 0) > self.state.chat_watermark(scope, chat_id):
                by_chat.setdefault(chat_id, []).append(message)
        if not by_chat:
            return {"scope": scope, "messages": 0, "chunks": 0, "since": since}

        keep_bot_quote = self._bot_quote_filter()
        channel_of = self._channel_of()
        chunks: List[Dict[str, Any]] = []
        marks: Dict[str, int] = {}
        for chat_id, messages in by_chat.items():
            built, consumed_through = build_chunks(
                messages, scope, settle_seconds=self.settle_seconds,
                keep_bot_quote=keep_bot_quote, channel_of=channel_of)
            chunks.extend(built)
            marks[chat_id] = self._advance(scope, chat_id, messages, consumed_through)

        if chunks:
            texts = [c["text"] for c in chunks]
            # Serialised against generation. This used to run unsynchronised on
            # the same card the model serves from — safe only because it ran
            # every two hours and was therefore unlikely to collide. At fifteen
            # minutes "unlikely" stops being an argument.
            from src.chat.gpu_lock import gpu_section

            with gpu_section(self.config):
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

        self.state.set_chat_watermarks(scope, marks, len(chunks))
        return {
            "scope": scope, "messages": sum(len(m) for m in by_chat.values()),
            "chunks": len(chunks), "since": since, "watermark": self.state.watermark(scope),
        }

    def rechunk_since(self, scope: str, since: str) -> Dict[str, Any]:
        """Rebuild a scope's live chunks from ``since`` (``YYYY-MM-DD``, UTC), once.

        For chunks written before per-chat chunking, when the Community's
        sub-groups were interleaved in one stream (``whatsapp.ingest.rechunk_since``).
        Deletes the live chunks that end on or after ``since`` and rewinds the
        scope to the earliest message they covered, so the next pass rebuilds
        them one chat at a time with their channel. Recorded in the state file,
        so it runs once per value; it runs inside the server process because
        ChromaDB must not be written by two processes at once.
        """
        done = (self.state._data.get("rechunked", {}) or {}).get(scope)
        if done == since:
            return {"scope": scope, "rechunked": 0, "skipped": "already done"}
        found = self.collection.get(where={"$and": [{"source": "live"}, {"scope": scope}]},
                                    include=["metadatas"])
        doomed = [(chunk_id, metadata) for chunk_id, metadata
                  in zip(found.get("ids") or [], found.get("metadatas") or [])
                  if str((metadata or {}).get("timestamp_end") or "") >= since]
        if doomed:
            earliest = min(str(metadata.get("timestamp_start") or since) for _, metadata in doomed)
            rewind = int(datetime.fromisoformat(earliest).replace(tzinfo=timezone.utc).timestamp()) - 1
            ids = [chunk_id for chunk_id, _ in doomed]
            for start in range(0, len(ids), 500):
                self.collection.delete(ids=ids[start:start + 500])
            entry = self._data_scope(scope)
            entry["last_ts"] = min(int(entry.get("last_ts", rewind)), rewind)
            entry["chats"] = {}
        self.state._data.setdefault("rechunked", {})[scope] = since
        self.state._save()
        logger.info("rechunk %s since %s: %d live chunk(s) removed", scope, since, len(doomed))
        return {"scope": scope, "rechunked": len(doomed)}

    def _data_scope(self, scope: str) -> Dict[str, Any]:
        return self.state._data.setdefault("scopes", {}).setdefault(scope, {})

    def _advance(self, scope: str, chat_id: str, messages: List[Dict[str, Any]],
                 consumed_through: int) -> int:
        """Where one chat's watermark may move to after this pass."""
        since = self.state.chat_watermark(scope, chat_id)
        # Clamp to now. The watermark is a high-water mark, so a single message
        # with a clock-skewed or bogus future timestamp pins it ahead of real
        # time and every later message is silently skipped — memory stops
        # updating with no error anywhere. A simulator run hit exactly this: one
        # event dated in the future froze `shared` ingestion, and a fact planted
        # afterwards was logged but never became retrievable.
        horizon = int(time.time()) + _FUTURE_TOLERANCE_S
        timestamps = [int(m.get("timestamp") or 0) for m in messages]
        newest = max((t for t in timestamps if t <= horizon), default=0)
        skipped = [t for t in timestamps if t > horizon]
        if skipped:
            logger.warning(
                "%s/%s: %d message(s) dated in the future (max %s); watermark held at %s",
                scope, chat_id, len(skipped), max(skipped), newest)
        # Never advance past messages that were deliberately left unconsumed for
        # the next run — doing so is how they would be lost forever, since read()
        # only ever returns what is strictly newer than the watermark.
        if consumed_through:
            newest = min(newest, consumed_through)
        else:
            newest = since
        # Nothing legitimately newer: leave the watermark alone rather than
        # moving it backwards.
        return max(newest, since)

    def _channel_of(self) -> Callable[[str], str]:
        """A chat id's channel name, read once per pass."""
        from src.chat import channels

        names = channels.from_config(self.config)
        cache: Dict[str, str] = {}

        def channel_of(chat_id: str) -> str:
            if chat_id not in cache:
                try:
                    cache[chat_id] = names.label(chat_id) if chat_id.endswith("@g.us") else ""
                except Exception:  # noqa: BLE001 — a missing name is not a lost chunk
                    cache[chat_id] = ""
            return cache[chat_id]

        return channel_of

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
    ingester = Ingester(config)
    since = ((config.get("whatsapp", {}) or {}).get("ingest", {}) or {}).get("rechunk_since")
    if since:
        try:
            ingester.rechunk_since("shared", str(since))
        except Exception as exc:  # noqa: BLE001 — the old chunks stay, which is the old behaviour
            logger.warning("rechunk since %s failed: %s", since, exc)
    results = ingester.ingest_all()
    total_chunks = sum(r.get("chunks", 0) for r in results)
    total_msgs = sum(r.get("messages", 0) for r in results)
    if total_msgs:
        print(
            f"✓ Ingested {total_msgs} message(s) into {total_chunks} chunk(s) "
            f"across {len(results)} scope(s) in {time.time() - t0:.1f}s"
        )
    return results
