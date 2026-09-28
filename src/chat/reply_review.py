"""What the bot said, and whether it was any good: a ledger of its own replies.

The rolling summary reads every line of a chat, the bot's included, and on
2026-09-28 that is how a roast became a fact. At 17:08 the bot mocked Rafa's
"three failed startups"; Rafa answered "I finished them". At 17:25 the summary
was rewritten with *"O Rafa, empreendedor que lançou três startups falhadas"*,
and at 17:26, asked what the "jogo do titz" was — something it had no record of
at all — the bot answered with that sentence almost word for word. The group's
verdict was "alucinação de contexto", and it was exactly that: its own joke,
laundered through the summary, handed back to it as background.

Dropping the bot's lines from the summary would lose the answers that are worth
keeping. So each reply is recorded here when it is sent, with the mode that
produced it, and later judged:

* **by its mode**, deterministically: a roast or a banter line is a joke, and a
  joke is never a fact about anyone (``is_joke``);
* **by a bug report**: every bot line in a ``/bug``'s recent turns is ``mau``;
* **by the local model** (``review``), in the summary's own background pass: it
  sees each reply between what prompted it and how the group reacted, and says
  whether it answered what was asked (``ok``), went somewhere nobody asked
  (``fora``), or said something the group then contested (``inventado``).

The verdicts are the record, and ``summary_line`` / ``keep_quote`` are how they
are used: a bad reply never reaches the summary or long-term memory; a joke
reaches them labelled as one.

Stdlib only: ``whatsapp_adapter`` imports this, and the Pi gateway imports the
adapter (``tests/gateway/test_torch_free.py``).
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.chat.memory import _safe_key
from src.chat.response_utils import truncate_history_line

BOT_PREFIX = "Kaya Bot: "

OK = "ok"
FORA = "fora"
INVENTADO = "inventado"
MAU = "mau"
VERDICTS = (OK, FORA, INVENTADO, MAU)
BAD_VERDICTS = (FORA, INVENTADO, MAU)

# Modes whose replies are jokes by construction. Strings rather than router
# constants so this module stays importable without the router.
JOKE_MODES = ("roast", "banter")

# Enough words to tell two replies apart, few enough that a quote WhatsApp
# truncated (``whatsapp.quoted_max_words``, 30) still carries the whole key.
_KEY_WORDS = 12
_EDGE_PUNCTUATION = ".,;:!?¡¿\"'«»()[]*_~“”‘’"


def reply_key(text: str) -> str:
    """The first words of a reply, normalised, so a truncated quote finds it."""
    body = (text or "").strip()
    if body.startswith(BOT_PREFIX):
        body = body[len(BOT_PREFIX):]
    words = []
    for word in body.lower().split():
        word = word.strip(_EDGE_PUNCTUATION)
        if word and word not in ("…", "..."):
            words.append(word)
        if len(words) >= _KEY_WORDS:
            break
    return " ".join(words)


def is_joke(entry: Optional[Dict[str, Any]]) -> bool:
    return bool(entry) and entry.get("mode") in JOKE_MODES


def is_bad(entry: Optional[Dict[str, Any]]) -> bool:
    return bool(entry) and entry.get("verdict") in BAD_VERDICTS


class ReplyLedger:
    """One JSON file per chat: reply key -> ``{mode, ts, verdict, reason, source}``.

    Capped at ``max_entries`` per chat, oldest dropped first. Every method
    swallows its own failures — a ledger is never worth a reply.
    """

    def __init__(self, base_dir: str = "data/whatsapp_replies", max_entries: int = 500):
        self.base_dir = Path(base_dir)
        if not self.base_dir.is_absolute():
            self.base_dir = Path(__file__).parent.parent.parent / base_dir
        self.max_entries = max_entries
        self._lock = threading.Lock()

    def _path(self, chat_id: str) -> Path:
        return self.base_dir / f"{_safe_key(chat_id)}.json"

    def _load(self, chat_id: str) -> Dict[str, Dict[str, Any]]:
        path = self._path(chat_id)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a corrupt ledger reads as empty
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, chat_id: str, entries: Dict[str, Dict[str, Any]]) -> None:
        if len(entries) > self.max_entries:
            newest = sorted(entries.items(), key=lambda item: item[1].get("ts", ""))
            entries = dict(newest[-self.max_entries:])
        path = self._path(chat_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    def entries(self, chat_id: str) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return self._load(chat_id)

    def lookup(self, chat_id: str, text: str) -> Optional[Dict[str, Any]]:
        key = reply_key(text)
        if not key or not chat_id:
            return None
        with self._lock:
            return self._load(chat_id).get(key)

    def record(self, chat_id: str, text: str, mode: str = "") -> None:
        """A reply was just sent. Never raises."""
        key = reply_key(text)
        if not key or not chat_id:
            return
        try:
            with self._lock:
                entries = self._load(chat_id)
                entry = entries.get(key, {})
                entry.update({"mode": mode or entry.get("mode", ""),
                              "ts": datetime.now(timezone.utc).isoformat()})
                entry.setdefault("verdict", None)
                entries[key] = entry
                self._save(chat_id, entries)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  could not record a reply: {exc}")

    def mark(self, chat_id: str, text: str, verdict: str, reason: str = "",
             source: str = "") -> None:
        """Judge one reply. Never raises. A reply not recorded yet is created."""
        key = reply_key(text)
        if not key or not chat_id or verdict not in VERDICTS:
            return
        try:
            with self._lock:
                entries = self._load(chat_id)
                entry = entries.get(key) or {
                    "mode": "", "ts": datetime.now(timezone.utc).isoformat()}
                entry.update({"verdict": verdict, "reason": reason.strip()[:200],
                              "source": source})
                entries[key] = entry
                self._save(chat_id, entries)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  could not record a verdict: {exc}")

    def mark_bug_report(self, chat_id: str, recent_turns: List[str]) -> int:
        """Every bot line a ``/bug`` was filed against is ``mau``. Returns how many."""
        marked = 0
        for line in recent_turns or []:
            if line.startswith(BOT_PREFIX):
                self.mark(chat_id, line, MAU, "bug report", source="bug")
                marked += 1
        return marked


def summary_line(line: str, entry: Optional[Dict[str, Any]]) -> Optional[str]:
    """How one history line reaches the summary writer; None drops it."""
    if not line.startswith(BOT_PREFIX):
        return line
    if is_bad(entry):
        return None
    if is_joke(entry):
        return f"Kaya Bot (piada, não é facto): {line[len(BOT_PREFIX):]}"
    return line


def keep_quote(entry: Optional[Dict[str, Any]]) -> bool:
    """Whether a member's reply to the bot may carry the bot's words into memory.

    Only a reply the ledger knows, that was not a joke and was not judged bad. A
    quote it cannot find is dropped: it is the bot's text, not the group's, and
    the group's side of the exchange is kept either way.
    """
    return bool(entry) and not is_joke(entry) and not is_bad(entry)


# ── the local review ─────────────────────────────────────────────────────────
REVIEW_SYSTEM = (
    "Avalias respostas de um bot (Kaya Bot) num grupo de WhatsApp de amigos. "
    "Cada resposta numerada vem entre as mensagens que a antecederam e as que "
    "vieram a seguir. Para cada uma escreve uma linha, exactamente no formato "
    "'N: VEREDICTO - motivo curto', em que VEREDICTO é um de:\n"
    "OK - responde ao que lhe disseram ou pediram. Piadas, insultos e roasts são "
    "OK quando são sobre o que foi pedido.\n"
    "FORA - não responde ao que foi perguntado: muda de assunto, ignora a "
    "pergunta, responde a uma pergunta com um insulto no lugar da resposta, ou "
    "vai falar de alguém que não fazia parte da conversa.\n"
    "INVENTADO - afirma algo sobre alguém ou alguma coisa que as mensagens "
    "seguintes contestam, corrigem ou dizem que é alucinação.\n"
    "Uma linha por resposta, nada mais."
)

_VERDICT_LINE = re.compile(
    r"^\s*(?:resposta\s*)?\[?(\d+)\]?\s*[:.)\-]?\s*\**\s*(OK|FORA|INVENTADO)\b\**\s*[-–—:]?\s*(.*)$",
    re.IGNORECASE,
)


def review_targets(lines: List[str], ledger: Optional[ReplyLedger], chat_id: str,
                   limit: int = 20) -> List[int]:
    """Indices of bot lines that have no verdict yet, oldest first."""
    entries = ledger.entries(chat_id) if ledger is not None else {}
    targets = []
    for index, line in enumerate(lines):
        if not line.startswith(BOT_PREFIX):
            continue
        entry = entries.get(reply_key(line))
        if entry and entry.get("verdict"):
            continue
        targets.append(index)
    return targets[:limit]


def build_review_prompt(lines: List[str], targets: List[int], before: int = 3,
                        after: int = 3, max_words: int = 40) -> str:
    blocks = []
    for number, index in enumerate(targets, start=1):
        context = [truncate_history_line(line, max_words)
                   for line in lines[max(0, index - before):index]]
        reaction = [truncate_history_line(line, max_words)
                    for line in lines[index + 1:index + 1 + after]]
        blocks.append(
            f"[{number}]\nAntes:\n" + ("\n".join(context) or "(nada)") +
            f"\nResposta {number}: {truncate_history_line(lines[index], 80)}\n"
            "Depois:\n" + ("\n".join(reaction) or "(nada)")
        )
    return "\n\n".join(blocks)


def parse_review(raw: str, count: int) -> Dict[int, Tuple[str, str]]:
    """``{number: (verdict, reason)}`` for every well-formed line in range."""
    parsed: Dict[int, Tuple[str, str]] = {}
    for line in (raw or "").splitlines():
        match = _VERDICT_LINE.match(line)
        if not match:
            continue
        number = int(match.group(1))
        if 1 <= number <= count and number not in parsed:
            parsed[number] = (match.group(2).lower(), match.group(3).strip())
    return parsed
