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

from src.chat import reply_review
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
    "As linhas do Kaya Bot são o próprio bot a falar: o que ele diz sobre alguém "
    "nunca é um facto sobre essa pessoa, a não ser que um membro o confirme, e as "
    "marcadas como piada nunca entram no resumo como factos. "
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


# How many trailing lines identify where the last summary stopped. One line
# collides too easily in a chat full of "Fds" and "Ahahah"; three effectively
# never do.
_MARKER_LINES = 3


def new_lines_since(history: List[str], state: Dict[str, Any]) -> List[str]:
    """The lines written since the last summary, found by content not by count.

    A count cannot work here. The session window is capped, so ``len(history)``
    stops growing while the conversation does not, and ``lines_seen`` ratchets up
    to meet it — the live group sat at lines_seen=90 against a 100-line window
    for three weeks, which made ``len(history) - lines_seen >= 30`` permanently
    unsatisfiable and silently killed the rolling summary.

    So the marker is the last few lines that were summarised. Everything after
    them is new. If they are not in the window at all the conversation has moved
    on entirely and all of it is new, which is both correct and what heals a
    state file written by the old, broken counter.
    """
    marker = [line for line in (state.get("marker") or []) if line]
    if not marker:
        return list(history)
    width = len(marker)
    for start in range(len(history) - width, -1, -1):
        if history[start:start + width] == marker:
            return history[start + width:]
    return list(history)


class ChatSummaryStore:
    """One JSON file per chat, holding its rolling summary and a position marker.

    ``marker`` is the last few lines covered by the summary; ``new_lines_since``
    locates them to work out what is new. ``lines_seen`` is kept for readability
    when someone opens the file, and is no longer read by anything.
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
            return {"summary": "", "lines_seen": 0, "marker": [], "updated": ""}
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a corrupt summary must not break a reply
            return {"summary": "", "lines_seen": 0, "marker": [], "updated": ""}
        state.setdefault("marker", [])
        return state

    def summary_for(self, chat_id: str) -> str:
        return (self.load(chat_id).get("summary") or "").strip()

    def save(self, chat_id: str, summary: str, history: List[str]) -> None:
        path = self._path(chat_id)
        payload = {
            "summary": summary.strip(),
            "lines_seen": len(history),
            "marker": [line for line in history[-_MARKER_LINES:] if line],
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
                 store: Optional[ChatSummaryStore] = None,
                 ledger: Optional[reply_review.ReplyLedger] = None):
        self.config = config
        self.backend = backend
        cfg = ((config.get("chat", {}) or {}).get("summary", {}) or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.every_lines = int(cfg.get("every_lines", 30))
        self.max_words = int(cfg.get("max_words", 150))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 220))
        self.lock_timeout = float(cfg.get("lock_timeout_seconds", 5))
        # The bot's own replies, and what has been decided about them. Without a
        # ledger its lines reach the summary unfiltered, as they always did.
        self.ledger = ledger
        self.review_enabled = bool(cfg.get("review_replies", True))
        self.review_limit = int(cfg.get("review_max_replies", 20))
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
        if len(new_lines_since(history, state)) < self.every_lines:
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

        new_lines = new_lines_since(history, state)
        if not new_lines:
            return
        previous = (state.get("summary") or "").strip()
        if not self._review(chat_id, new_lines):
            print(f"⏳ summary for {chat_id} deferred — GPU busy")
            return
        # Filtered for the prompt only. The marker below is taken from the
        # unfiltered history, or a dropped line at the end would never be found
        # again and every later update would re-read the whole window.
        prompt_lines = self._prompt_lines(chat_id, new_lines)
        if not prompt_lines:
            if previous:
                self.store.save(chat_id, previous, history)
            return
        prompt = (_UPDATE.format(previous=previous, new_lines="\n".join(prompt_lines))
                  if previous else _FIRST.format(new_lines="\n".join(prompt_lines)))
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
            # The marker is untouched, so the next message re-queues this chat.
            print(f"⏳ summary for {chat_id} deferred — GPU busy")
            return
        summary = (raw or "").strip()
        if not summary:
            return
        self.store.save(chat_id, summary, history)

    # ── the bot's own lines ──────────────────────────────────────────────────
    def _prompt_lines(self, chat_id: str, lines: List[str]) -> List[str]:
        """The lines the summary writer sees: bad replies out, jokes labelled."""
        if self.ledger is None:
            return list(lines)
        entries = self.ledger.entries(chat_id)
        kept = []
        for line in lines:
            entry = (entries.get(reply_review.reply_key(line))
                     if line.startswith(reply_review.BOT_PREFIX) else None)
            shown = reply_review.summary_line(line, entry)
            if shown is not None:
                kept.append(shown)
        return kept

    def _review(self, chat_id: str, lines: List[str]) -> bool:
        """Judge the bot's unjudged replies on the local model. False = GPU busy.

        Its own lock acquisition, not a share of the summary's: each generation
        holds the lock only as long as it runs, and a reply waiting for it is
        dropped rather than queued. Any other failure returns True, and the
        summary goes ahead with the verdicts it already has.
        """
        if self.ledger is None or not self.review_enabled:
            return True
        from src.chat.gpu_lock import GpuBusyError, gpu_section

        targets = reply_review.review_targets(lines, self.ledger, chat_id,
                                              self.review_limit)
        if not targets:
            return True
        messages = [
            {"role": "system", "content": reply_review.REVIEW_SYSTEM},
            {"role": "user",
             "content": reply_review.build_review_prompt(lines, targets)},
        ]
        try:
            with gpu_section(self.config, timeout=self.lock_timeout):
                raw = self.backend.generate(
                    messages,
                    max_new_tokens=40 * len(targets),
                    sampling={"temperature": 0.0, "top_p": 1.0, "top_k": 0,
                              "repetition_penalty": 1.0},
                )
        except GpuBusyError:
            return False
        except Exception as exc:  # noqa: BLE001 — a review is never worth a summary
            print(f"⚠️  reply review failed for {chat_id}: {exc}")
            return True
        verdicts = reply_review.parse_review(raw, len(targets))
        for number, (verdict, reason) in verdicts.items():
            self.ledger.mark(chat_id, lines[targets[number - 1]], verdict, reason,
                             source="review")
        set_aside = sum(1 for verdict, _ in verdicts.values()
                        if verdict in reply_review.BAD_VERDICTS)
        print(f"🧾 reviewed {len(verdicts)}/{len(targets)} bot replies in {chat_id}, "
              f"{set_aside} set aside")
        return True
