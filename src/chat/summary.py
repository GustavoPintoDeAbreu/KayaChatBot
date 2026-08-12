"""A rolling summary of what has scrolled out of a chat's verbatim window.

The bot holds the last ``whatsapp.history_turns`` lines of a conversation
exactly, and everything older has to be *found* again by semantic search. That
works well for a fact stated plainly ("o código do alarme é 4417") and badly for
the shape of a long exchange: who agreed to what, in what order, and what was
decided in the end. Search returns whichever chunk is nearest the question, not
the thread.

So each chat keeps a short running summary of the turns that have already fallen
off. It is regenerated in the background as the conversation grows, appended
above the verbatim block, and never leaves the machine.

Three properties matter more than the summary's prose:

* **It never blocks a reply.** Generation happens on a worker thread after the
  answer has been sent. The worker takes the GPU lock with a short timeout and
  gives up on ``GpuBusyError`` rather than waiting — ``whatsapp_server._process``
  DROPS an inbound message when the lock is contended, so a summary that sat on
  the lock would cost someone their turn. A skipped update is retried after the
  next message; there is nothing to lose by being late.
* **It stays in its own chat.** One file per chat id, exactly like
  ``KeyedSessionMemory``, and it is never written into the shared vector store.
  Scope isolation therefore needs no new rules to hold.
* **It updates incrementally.** The prompt revises the existing summary using
  only the lines that have newly fallen out, so the cost of a summary does not
  grow with the age of the conversation.
"""
from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chat.memory import _safe_key

# European Portuguese, because that is what the group speaks and what the summary
# will be read back into. Deliberately asks for decisions and attribution rather
# than atmosphere: the point is to keep the thread of an exchange, which is
# exactly what semantic search over individual chunks loses.
_SYSTEM = (
    "Resumes conversas de um grupo de amigos para memória de um bot. "
    "Escreves em português europeu, em texto corrido, no máximo {max_words} palavras. "
    "Guarda o que interessa a longo prazo: decisões tomadas, planos, quem disse ou "
    "combinou o quê, factos sobre as pessoas, e assuntos por resolver. "
    "Ignora conversa fiada, cumprimentos e piadas sem consequência. "
    "Não inventes nada que não esteja nas mensagens. Não comentes a tarefa, "
    "responde apenas com o resumo."
)

_UPDATE = (
    "Resumo até agora:\n{previous}\n\n"
    "Novas mensagens que saíram da janela recente:\n{new_lines}\n\n"
    "Reescreve o resumo incorporando as novas mensagens. Mantém o que continua "
    "relevante, deixa cair o que já não interessa."
)

_FIRST = (
    "Mensagens que saíram da janela recente:\n{new_lines}\n\n"
    "Escreve o resumo."
)


class ChatSummaryStore:
    """One JSON file per chat, holding its rolling summary and a position marker.

    ``lines_seen`` is the total number of lines the chat's session store had at
    the last update. It is what makes the trigger cheap: the caller compares it
    against the current count instead of re-reading the conversation.
    """

    def __init__(self, base_dir: str = "data/whatsapp_summaries"):
        self.base_dir = Path(base_dir)
        if not self.base_dir.is_absolute():
            self.base_dir = Path(__file__).parent.parent.parent / base_dir
        self._lock = threading.Lock()

    def _path(self, chat_id: str) -> Path:
        return self.base_dir / f"{_safe_key(chat_id)}.json"

    def load(self, chat_id: str) -> Dict[str, Any]:
        path = self._path(chat_id)
        if not path.exists():
            return {"summary": "", "lines_seen": 0, "updated": ""}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a corrupt summary must not break a reply
            return {"summary": "", "lines_seen": 0, "updated": ""}

    def summary_for(self, chat_id: str) -> str:
        return (self.load(chat_id).get("summary") or "").strip()

    def save(self, chat_id: str, summary: str, lines_seen: int) -> None:
        path = self._path(chat_id)
        payload = {
            "summary": summary.strip(),
            "lines_seen": int(lines_seen),
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(path)


class SummaryWriter:
    """Background worker that keeps chat summaries up to date.

    One thread, one queue, deduplicated by chat id: a chat already waiting for an
    update is not queued twice. Everything here is best-effort — every failure
    path leaves the previous summary in place and returns.
    """

    def __init__(self, config: Dict[str, Any], backend: Any,
                 store: Optional[ChatSummaryStore] = None):
        self.config = config
        self.backend = backend
        cfg = ((config.get("chat", {}) or {}).get("summary", {}) or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.every_lines = int(cfg.get("every_lines", 30))
        self.max_words = int(cfg.get("max_words", 150))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 220))
        self.lock_timeout = float(cfg.get("lock_timeout_seconds", 5))
        wcfg = config.get("whatsapp", {}) or {}
        self.store = store or ChatSummaryStore(
            wcfg.get("summaries_dir", "data/whatsapp_summaries"))
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._pending: set = set()
        self._guard = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ── trigger ──────────────────────────────────────────────────────────────
    def maybe_update(self, chat_id: str, history: List[str]) -> bool:
        """Queue an update if enough lines have accumulated. Never raises.

        Returns True when something was queued, which is what the tests assert
        on — the generation itself is asynchronous and deliberately unobservable
        from the caller's turn.
        """
        if not self.enabled or not chat_id or not history:
            return False
        state = self.store.load(chat_id)
        seen = int(state.get("lines_seen") or 0)
        if len(history) - seen < self.every_lines:
            return False
        with self._guard:
            if chat_id in self._pending:
                return False
            self._pending.add(chat_id)
        self._queue.put((chat_id, list(history), state))
        self._ensure_worker()
        return True

    def _ensure_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._guard:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="kaya-summary", daemon=True)
            self._thread.start()

    # ── worker ───────────────────────────────────────────────────────────────
    def _run(self) -> None:
        while True:
            try:
                chat_id, history, state = self._queue.get(timeout=300)
            except queue.Empty:
                return
            try:
                self._update_one(chat_id, history, state)
            except Exception as exc:  # noqa: BLE001 — a summary is never worth a crash
                print(f"⚠️  summary update failed for {chat_id}: {exc}")
            finally:
                with self._guard:
                    self._pending.discard(chat_id)
                self._queue.task_done()

    def _update_one(self, chat_id: str, history: List[str],
                    state: Dict[str, Any]) -> None:
        from src.chat.gpu_lock import GpuBusyError, gpu_section

        seen = int(state.get("lines_seen") or 0)
        new_lines = history[seen:]
        if not new_lines:
            return
        previous = (state.get("summary") or "").strip()
        prompt = (_UPDATE.format(previous=previous, new_lines="\n".join(new_lines))
                  if previous else _FIRST.format(new_lines="\n".join(new_lines)))
        messages = [
            {"role": "system", "content": _SYSTEM.format(max_words=self.max_words)},
            {"role": "user", "content": prompt},
        ]
        try:
            # Short timeout, and skip rather than wait: holding this lock would
            # make the bridge drop somebody's message.
            with gpu_section(self.config, timeout=self.lock_timeout):
                raw = self.backend.generate(
                    messages,
                    max_new_tokens=self.max_new_tokens,
                    sampling={"temperature": 0.3, "top_p": 0.9, "top_k": 0,
                              "repetition_penalty": 1.05},
                )
        except GpuBusyError:
            # lines_seen is untouched, so the next message re-queues this chat.
            print(f"⏳ summary for {chat_id} deferred — GPU busy")
            return
        summary = (raw or "").strip()
        if not summary:
            return
        self.store.save(chat_id, summary, len(history))
