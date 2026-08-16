"""Counting, done by counting.

Semantic retrieval returns the ``top_k`` chunks nearest a question. That is the
right tool for "quando foi o jantar?" and structurally the wrong one for "quantas
vezes é que cada um disse X" — the answer is not in any one chunk, it is a
property of every message there has ever been.

Asked exactly that, the bot wrote a per-member table off the back of 6674
characters of retrieved context. Against the log it had been undercounting by
8x, with the ranking inverted: the top user, at 198, was reported as third with
3. Frederico said "não deves ter contado todos" and was right; the bot answered
"pode ser que me tenha escapado alguns" and carried on.

So the numbers are produced here, by reading the log, and the model is handed the
finished table to phrase. It never does the arithmetic.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent

# The term the user wants counted, pulled out of the question. Ordered longest
# context first: an explicitly quoted term always wins over a guessed one.
_QUOTED = re.compile(r"[\"“”'`«»]([^\"“”'`«»]{2,40})[\"“”'`«»]")
_SAID = re.compile(
    r"(?:disse(?:ram|mos)?|dizem|diz|falou|falaram|usou|usaram|escreveu|escreveram"
    r"|said|says|used|wrote)\s+(?:a\s+palavra\s+|o\s+termo\s+|the\s+word\s+)?"
    r"([\wÀ-ÿ'-]{2,40})",
    re.IGNORECASE,
)


# Words that follow a "disse"/"said" verb without being the thing to count.
# "quem é que diz mais isso?" would otherwise be answered as a count of "mais",
# and a plausible-looking number for the wrong word is the failure being fixed.
_NOT_A_TERM = {
    "mais", "menos", "muito", "muitas", "isso", "isto", "aquilo", "que", "quando",
    "onde", "como", "porque", "nada", "tudo", "sempre", "nunca", "alguma", "algo",
    "uma", "um", "o", "a", "os", "as", "essa", "esse", "esta", "este", "lá", "cá",
    "more", "less", "that", "this", "it", "what", "when", "anything", "nothing",
    "something", "always", "never", "the", "a", "an",
}


def _fold(text: str) -> str:
    """Lowercase and strip accents, so "nao" finds "não"."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def extract_term(question: str) -> str:
    """The word or phrase to count, or "" when the question does not name one.

    A quoted term wins outright. Otherwise the word after a "disse"/"said" verb
    is taken, which is how these questions are actually phrased. Returning ""
    means the caller must ask what to count rather than invent an answer.
    """
    quoted = _QUOTED.search(question or "")
    if quoted:
        return quoted.group(1).strip()
    said = _SAID.search(question or "")
    if said:
        candidate = said.group(1).strip()
        if _fold(candidate) not in _NOT_A_TERM:
            return candidate
    return ""


def load_resolver(config: Dict[str, Any]) -> Any:
    """A ``SenderResolver`` from config, or None if it cannot be built.

    Without it "Gil João" and "Gil" are two people — the raw log has both — and a
    per-member table splits one member across two rows.
    """
    try:
        from src.data.identity_resolver import SenderResolver

        data_cfg = config.get("data", {}) or {}
        members = data_cfg.get("group_members_file") or "data/group_members.json"
        path = Path(members)
        if not path.is_absolute():
            path = BASE_DIR / members
        return SenderResolver(str(path), data_cfg.get("sender_aliases"))
    except Exception as exc:  # noqa: BLE001 — unfolded names beat no count at all
        logger.warning("could not load the sender resolver: %s", exc)
        return None


def archive_path(config: Dict[str, Any]) -> Optional[Path]:
    """The pre-bot export, which is most of the history there is.

    The group existed for years before the bot did. A count that silently starts
    at the bot's install date is exactly the "isso deve ter sido só na semana
    passada" answer that got the fabricated table caught.
    """
    data_cfg = config.get("data", {}) or {}
    raw = data_cfg.get("cleaned_messages_file") or "data/all_messages_cleaned.jsonl"
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / raw
    return path if path.exists() else None


def _iter_log_files(scope: str, base_dir: Optional[Path] = None) -> List[Path]:
    """The log files this scope is allowed to count over.

    Scope isolation is the whole point: a DM must not be able to tally the
    group's history, so it counts only its own file.
    """
    directory = Path(base_dir) if base_dir else BASE_DIR / "data" / "live_messages"
    if not directory.is_dir():
        return []
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", scope or "shared")[:80]
    path = directory / f"{safe}.jsonl"
    return [path] if path.exists() else []


def count_term(
    term: str,
    scope: str = "shared",
    base_dir: Optional[Path] = None,
    resolver: Any = None,
    archive: Optional[Path] = None,
) -> Tuple[List[Tuple[str, int]], int]:
    """Count ``term`` per sender. Returns ``(rows, total)``, rows sorted desc.

    ``resolver`` is a ``SenderResolver``; without it "Gil João" and "Gil" are two
    people, which is how the raw log has them. ``archive`` adds the pre-bot
    history (``data/all_messages_cleaned.jsonl``) — the group existed for years
    before the bot did, and a count that silently starts at the bot's install date
    is the "isso deve ter sido só na semana passada" answer.
    """
    needle = _fold(term).strip()
    if not needle:
        return [], 0

    per_sender: Counter = Counter()
    seen_ids: set = set()

    def absorb(rows: Iterable[Dict[str, Any]], dedupe: bool) -> None:
        for row in rows:
            if dedupe:
                uid = row.get("id")
                if uid:
                    if uid in seen_ids:
                        continue
                    seen_ids.add(uid)
            hits = _fold(str(row.get("text") or "")).count(needle)
            if not hits:
                continue
            sender = str(row.get("sender") or "").strip()
            if resolver is not None and sender:
                sender = resolver.resolve(sender)
            per_sender[sender or "Alguém"] += hits

    for path in _iter_log_files(scope, base_dir):
        absorb(_read_jsonl(path), dedupe=True)
    if archive and Path(archive).exists():
        # The export has no message ids, so it cannot be deduped against the live
        # log. It is only read for the shared scope, where the two do not overlap:
        # the export ends where the bot's own logging begins.
        absorb(_read_jsonl(Path(archive)), dedupe=False)

    rows = sorted(per_sender.items(), key=lambda item: (-item[1], item[0]))
    return rows, sum(per_sender.values())


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        logger.warning("could not read %s: %s", path, exc)


def format_table(term: str, rows: List[Tuple[str, int]], total: int) -> str:
    """The counted table, as a context block for the model to phrase.

    Handed to the model as fact, with an explicit instruction not to adjust it.
    The failure being fixed is not that the model refused to answer; it is that
    it produced numbers that felt right.
    """
    if not rows:
        return (f'Contagem exacta de "{term}" no histórico: ninguém usou essa '
                f"expressão. Este número foi contado, não estimado.")
    lines = "\n".join(f"- {name}: {count}" for name, count in rows)
    return (
        f'Contagem exacta de "{term}" em todo o histórico do grupo '
        f"(contado mensagem a mensagem, não estimado):\n{lines}\n"
        f"Total: {total}.\n"
        "Usa estes números tal como estão. Não os arredondes, não os alteres e "
        "não acrescentes membros que não estejam na lista. Se te disserem que "
        "estão errados, diz que foram contados a partir do histórico completo."
    )
