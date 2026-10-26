"""Media nobody asked about waits while GPU1 is lent, and is filled in later.

A photo, sticker, voice note or document turns into text before it is logged
(``[Imagem: …]``, a transcript, ``[Documento: …]``), and that used to need Kaya's
model, or Whisper on her card, for every one of them, addressed or not. Each one
therefore threw a borrower off GPU1 (an idea build, a coding session) and
restarted the idle clock that lends the card out again. Since 2026-10-26 only a
reply takes the card back.

While GPU1 is lent (``inference_backend.gpu1_lent``), a message that will not be
answered is logged with a placeholder instead (``[Imagem: (ainda por ver)]``),
its media copied to ``data/enrichment/media/`` and the work queued in
``data/enrichment/pending.json``. ``Enricher`` fills it in later:

- right before a reply in that chat (``before_reply``): its voice notes, and the
  message being quoted, so the answer hears them;
- right after a reply (``kick``), while the model is warm anyway;
- whenever a poll finds GPU1 free again (``poll_seconds``).

A fill rewrites the log through ``MessageLog.amend`` and the session window
through ``KeyedSessionMemory.replace_line``. The ingester holds a chat's chunks
at its oldest pending message (``barriers``) for up to ``ingest_hold_hours``;
a fill after that becomes a small chunk of its own (``late_fills``).

Stdlib only: the adapter imports this, and the Pi gateway imports the adapter
(``tests/gateway/test_torch_free.py``).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

PLACEHOLDERS = {
    "image": "[Imagem: (ainda por ver)]",
    "sticker": "[Sticker: (ainda por ver)]",
    "audio": "[Áudio: (ainda por ouvir)]",
}
PENDING_RE = re.compile(r"\[(?:Imagem|Sticker|Áudio|Documento):[^\]]*\(ainda por (?:ver|ouvir|ler)\)\]")

DEFAULTS = {
    "defer_when_gpu1_lent": True,
    "dir": "data/enrichment",
    "poll_seconds": 120,
    "lock_timeout_seconds": 5,
    "max_attempts": 3,
    "drain_before_reply_kinds": ["audio"],
    "ingest_hold_hours": 12,
    "drop_after_days": 7,
}

Filler = Callable[[Dict[str, Any], Optional[bytes], Dict[str, Any]], Optional[str]]


def settings(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = ((config.get("whatsapp", {}) or {}).get("enrichment", {}) or {})
    return {**DEFAULTS, **raw}


def document_placeholder(filename: str) -> str:
    name = re.sub(r"[\[\]]", "", (filename or "documento").strip()) or "documento"
    return f"[Documento: {name} (ainda por ler)]"


def strip_pending(text: str) -> str:
    """Drop placeholders that were never filled, so they are never embedded."""
    if "(ainda por " not in (text or ""):
        return text
    return re.sub(r"[ \t]*\n[ \t]*\n+", "\n", PENDING_RE.sub("", text)).strip()


def _fill_block(text: str, block: str, filled: Optional[str]) -> str:
    """``text`` with ``block`` replaced by ``filled``, or removed when it failed."""
    if block not in text:
        return text
    if filled:
        return text.replace(block, filled, 1)
    return re.sub(r"\n{2,}", "\n", text.replace(block, "", 1)).strip()


def queue_dir(config: Dict[str, Any]) -> Path:
    """``whatsapp.enrichment.dir``; a relative one is under the repo, like the message log.
    ``KAYA_ENRICHMENT_DIR`` wins (the tests, the simulator)."""
    directory = Path(os.environ.get("KAYA_ENRICHMENT_DIR") or settings(config)["dir"])
    return directory if directory.is_absolute() else Path(__file__).resolve().parents[2] / directory


def _queue_path(config: Dict[str, Any]) -> Path:
    return queue_dir(config) / "pending.json"


def barriers(config: Dict[str, Any], scope: str, now: Optional[float] = None) -> Dict[str, int]:
    """``{chat_id: oldest pending timestamp}`` for one scope: ingest stops short of it.

    An entry older than ``ingest_hold_hours`` no longer holds anything back, so a
    queue that cannot drain (GPU1 lent all day) does not freeze long-term memory.
    """
    data = _read(_queue_path(config))
    hold = float(settings(config)["ingest_hold_hours"]) * 3600
    moment = time.time() if now is None else now
    found: Dict[str, int] = {}
    for entry in data.get("entries", {}).values():
        if entry.get("scope") != scope or moment - float(entry.get("queued_at") or 0) > hold:
            continue
        chat_id, ts = str(entry.get("chat_id") or ""), int(entry.get("timestamp") or 0)
        found[chat_id] = min(found.get(chat_id, ts), ts)
    return found


def _read(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


class EnrichmentQueue:
    """The pending entries and their media, on disk, so a restart keeps them."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.path = self.dir / "pending.json"
        self.media = self.dir / "media"
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Any]:
        data = _read(self.path)
        data.setdefault("entries", {})
        data.setdefault("late_fills", [])
        return data

    def _save(self, data: Dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, entry: Dict[str, Any], payloads: Dict[int, bytes]) -> None:
        with self._lock:
            data = self._load()
            if entry["uid"] in data["entries"]:
                return
            for index, payload in payloads.items():
                if not payload:
                    continue
                self.media.mkdir(parents=True, exist_ok=True)
                path = self.media / f"{entry['uid']}-{index}"
                path.write_bytes(payload)
                entry["parts"][index]["media_path"] = str(path)
            data["entries"][entry["uid"]] = entry
            self._save(data)

    def get(self, uid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._load()["entries"].get(uid)

    def entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            found = list(self._load()["entries"].values())
        return sorted(found, key=lambda entry: (int(entry.get("timestamp") or 0), entry["uid"]))

    def bump(self, uid: str, error: str) -> int:
        with self._lock:
            data = self._load()
            entry = data["entries"].get(uid)
            if entry is None:
                return 0
            entry["attempts"] = int(entry.get("attempts") or 0) + 1
            entry["last_error"] = error[:300]
            self._save(data)
            return entry["attempts"]

    def remove(self, uid: str, late_fill: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            data = self._load()
            entry = data["entries"].pop(uid, None)
            if late_fill:
                data["late_fills"].append(late_fill)
            self._save(data)
        for part in (entry or {}).get("parts", []):
            if part.get("media_path"):
                Path(part["media_path"]).unlink(missing_ok=True)

    def late_fills(self, scope: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [fill for fill in self._load()["late_fills"] if fill.get("scope") == scope]

    def clear_late_fills(self, ids: List[str]) -> None:
        with self._lock:
            data = self._load()
            data["late_fills"] = [fill for fill in data["late_fills"] if fill.get("id") not in ids]
            self._save(data)

    def __len__(self) -> int:
        with self._lock:
            return len(self._load()["entries"])


class Enricher:
    """Fills in queued media once the model may be called without taking GPU1.

    ``fillers`` map a part's kind (image, sticker, audio, document) to a callable
    ``(part, payload, entry) -> text or None``; ``can_run`` says whether the model may
    be called now without evicting anything (``ensure_model_loaded(urgent=False)``)
    and ``prepare_urgent`` takes the card back for a reply.
    """

    def __init__(self, config: Dict[str, Any], queue: EnrichmentQueue, fillers: Dict[str, Filler],
                 message_log: Any, session_store: Any, can_run: Callable[[], bool],
                 prepare_urgent: Callable[[], Any], now: Callable[[], float] = time.time):
        self.config = config
        self.settings = settings(config)
        self.queue = queue
        self.fillers = fillers
        self.message_log = message_log
        self.session_store = session_store
        self.can_run = can_run
        self.prepare_urgent = prepare_urgent
        self.now = now
        self._filled: Dict[str, Optional[str]] = {}
        self._inflight: Dict[str, threading.Event] = {}
        self._guard = threading.Lock()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings["defer_when_gpu1_lent"])

    def should_defer(self) -> bool:
        """True while enrichment nobody asked for has to wait for the card."""
        if not self.enabled:
            return False
        try:
            return not self.can_run()
        except Exception as exc:  # noqa: BLE001 — unsure means wait, never evict
            logger.warning("could not tell whether GPU1 is lent: %s", exc)
            return True

    def defer(self, *, uid: str, message_id: str, chat_id: str, scope: str,
              timestamp: Optional[int], log: Dict[str, Any], text: str,
              session_line: str, parts: List[Dict[str, Any]],
              payloads: Dict[int, bytes]) -> None:
        """Queue a logged message whose placeholders are filled in later."""
        entry = {"uid": uid, "message_id": message_id, "chat_id": chat_id, "scope": scope,
                 "timestamp": int(timestamp or 0), "queued_at": self.now(), "attempts": 0,
                 "last_error": "", "log": log, "text": text, "session_line": session_line,
                 "parts": parts}
        self.queue.add(entry, payloads)
        logger.info("deferred %s in %s until GPU1 is free",
                    ", ".join(part["kind"] for part in parts), chat_id)

    def _payload(self, part: Dict[str, Any]) -> Optional[bytes]:
        path = part.get("media_path")
        if not path:
            return None
        try:
            return Path(path).read_bytes()
        except OSError:
            return None

    def process(self, uid: str, lock_timeout: Optional[float] = None) -> bool:
        """Fill one entry in; True once it is done (filled, failed for good, or gone)."""
        with self._guard:
            event = self._inflight.get(uid)
            mine = event is None
            if mine:
                event = self._inflight[uid] = threading.Event()
        if not mine:
            event.wait(timeout=120)
            return self.queue.get(uid) is None
        try:
            return self._process(uid, lock_timeout)
        finally:
            with self._guard:
                self._inflight.pop(uid, None)
            event.set()

    def _process(self, uid: str, lock_timeout: Optional[float]) -> bool:
        from src.chat.gpu_lock import GpuBusyError, gpu_section

        entry = self.queue.get(uid)
        if entry is None:
            return True
        too_old = self.now() - float(entry.get("queued_at") or 0) > \
            float(self.settings["drop_after_days"]) * 86400
        results: List[Optional[str]] = []
        if not too_old:
            try:
                with gpu_section(self.config, timeout=lock_timeout):
                    for part in entry["parts"]:
                        filler = self.fillers.get(part["kind"])
                        results.append(filler(part, self._payload(part), entry) if filler else None)
            except GpuBusyError:
                return False
            except Exception as exc:  # noqa: BLE001 — retried, then given up on
                attempts = self.queue.bump(uid, f"{type(exc).__name__}: {exc}")
                logger.warning("enrichment of %s failed (%d): %s", uid, attempts, exc)
                if attempts < int(self.settings["max_attempts"]):
                    return False
                results = []
        text, line = entry["text"], entry.get("session_line") or ""
        for index, part in enumerate(entry["parts"]):
            filled = results[index] if index < len(results) else None
            text = _fill_block(text, part["block"], filled)
            if line:
                line = _fill_block(line, part["block"], filled)
        self.message_log.amend(chat_id=entry["chat_id"], message_id=entry["message_id"],
                               text=text, timestamp=entry["timestamp"], scope=entry["scope"],
                               **entry.get("log", {}))
        new_line: Optional[str] = line
        if line and not line.partition(":")[2].strip():
            new_line = None
        if entry.get("session_line"):
            self.session_store.replace_line(entry["chat_id"], entry["session_line"], new_line)
            with self._guard:
                self._filled[entry["session_line"]] = new_line
                while len(self._filled) > 500:
                    self._filled.pop(next(iter(self._filled)))
        late = None
        if text and self.now() - float(entry.get("queued_at") or 0) > \
                float(self.settings["ingest_hold_hours"]) * 3600:
            late = {"id": uid, "chat_id": entry["chat_id"], "scope": entry["scope"],
                    "timestamp": entry["timestamp"], "text": text,
                    "sender": (entry.get("log") or {}).get("sender", "")}
        self.queue.remove(uid, late_fill=late)
        logger.info("filled in %s (%s)", uid, "ok" if any(results) else "nothing usable")
        return True

    def before_reply(self, chat_id: str, reply_to_id: str = "") -> int:
        """Fill in what a reply in ``chat_id`` must hear first; returns how many.

        Its voice notes (``drain_before_reply_kinds``) and the message it quotes.
        The card is taken back first, as for the reply itself: Whisper has no
        room next to a borrower.
        """
        kinds = set(self.settings["drain_before_reply_kinds"] or [])
        due = [entry for entry in self.queue.entries()
               if entry.get("chat_id") == chat_id
               and ((reply_to_id and entry.get("message_id") == reply_to_id)
                    or any(part["kind"] in kinds for part in entry["parts"]))]
        if not due:
            return 0
        self.prepare_urgent()
        done = 0
        for entry in due:
            if self.process(entry["uid"]):
                done += 1
        return done

    def resolve_line(self, line: str) -> Optional[str]:
        """A session line as it reads now: filled in, dropped (None), or unchanged."""
        with self._guard:
            return self._filled.get(line, line)

    def kick(self) -> None:
        self._wake.set()

    def drain(self) -> int:
        """Fill in everything that can be, oldest first, while GPU1 stays free."""
        done = 0
        for entry in self.queue.entries():
            if self.should_defer():
                break
            if self.process(entry["uid"], lock_timeout=float(self.settings["lock_timeout_seconds"])):
                done += 1
        return done

    def start(self) -> None:
        """The background worker: drains on a kick, and every ``poll_seconds``."""
        if self._thread is not None or not self.enabled:
            return

        def loop() -> None:
            while True:
                self._wake.wait(timeout=float(self.settings["poll_seconds"]))
                self._wake.clear()
                if not len(self.queue):
                    continue
                try:
                    self.drain()
                except Exception as exc:  # noqa: BLE001 — the worker must not die
                    logger.warning("enrichment pass failed: %s", exc)

        self._thread = threading.Thread(target=loop, daemon=True, name="enrichment")
        self._thread.start()
        if len(self.queue):
            self.kick()
