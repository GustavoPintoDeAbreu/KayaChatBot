"""Recaps of one topic: "o que é que decidimos sobre o almoço de natal?".

Top-k retrieval hands the model the few chunks nearest the question, which is the
wrong tool for summarising a back-and-forth spread over days: options get
suggested, dropped and brought back, and the decision is usually late in the
thread. Like ``tally.py``, this module reads the scope-bound log instead, selects
the messages on the topic, and hands the model a dated, attributed transcript.

Bot replies are not in the message log (the adapter logs only what people sent),
and the rolling summary is deliberately not used: it is model-written, and the
bot's own claims were laundered into group facts through it once (PR #95). A
recap is built only from what people actually wrote.
"""
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Pattern, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

from src.chat.response_utils import truncate_history_line
from src.chat.scope import SHARED, is_readable, readable_scopes
from src.chat.tally import _fold
from src.data.ingest import strip_failed_descriptions
from src.data.message_log import MessageLog

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "max_new_tokens": 450,
    "semantic_top_k": 8,
    "neighbour_rows": 3,
    "neighbour_gap_minutes": 30,
    "common_ratio": 0.02,
    "common_floor": 25,
    "max_chars": 40000,
    "max_archive_chunks": 3,
    "max_keywords": 6,
}

_STOPWORDS = frozenset({
    # Portuguese function words
    "que", "qual", "quais", "quem", "como", "onde", "quando", "porque", "para", "pra",
    "pelo", "pela", "com", "sem", "sobre", "dos", "das", "nos", "nas", "num", "numa",
    "uma", "uns", "umas", "este", "esta", "esse", "essa", "isso", "isto", "aquilo",
    "aquele", "aquela", "ele", "ela", "eles", "elas", "nós", "nos", "vocês", "voces",
    "meu", "minha", "teu", "tua", "seu", "sua", "nosso", "nossa", "mais", "menos",
    "muito", "tudo", "todos", "todas", "cada", "foi", "era", "ser", "estar", "tem",
    "ter", "vai", "vamos", "ainda", "afinal", "entao", "agora", "aqui",
    # the request itself, not the topic
    "resume", "resumo", "resumir", "resumes", "faz", "faca", "fazer", "diz", "dizer",
    "decidimos", "decidiram", "decidido", "decidida", "decisao", "dissemos", "disseram",
    "disse", "falamos", "falaram", "falou", "conversa", "conversas", "conversamos",
    "ponto", "situacao", "grupo", "chat", "kaya", "bot", "pessoal", "malta",
    # English
    "the", "and", "what", "who", "where", "when", "about", "did", "was", "were",
    "we", "our", "this", "that", "with", "for", "summary", "summarise", "summarize",
    "recap", "decide", "decided", "said", "say", "talked", "conversation",
})

RECAP_INSTRUCTION = (
    'Faz o ponto de situação de "{topic}" usando só as mensagens do histórico que '
    "te foram dadas acima. Usa três partes, cada uma com o título em negrito do "
    "WhatsApp: *Decidido*, *Em aberto* e *Quem disse o quê* (em inglês, se a "
    "pergunta for em inglês: *Decided*, *Still open*, *Who said what*). Em *Quem "
    'disse o quê* escreve uma linha por pessoa, a começar por "- Nome:", com a '
    "posição ou proposta mais recente dessa pessoa; se mudou de ideias, diz de quê "
    "para quê. Uma opção que foi posta de parte ou abandonada não está em aberto: "
    "diz que caiu e, se a razão estiver nas mensagens, porquê. Junta as opções "
    "repetidas numa só. Só atribuis uma posição a alguém se houver uma mensagem "
    'dessa pessoa que a mostre. Se nada foi decidido, diz "nada decidido ainda". '
    "Ignora as mensagens que não são sobre o tema. Não inventes opções, datas, "
    "preços nem pessoas."
)
NO_TOPIC_INSTRUCTION = (
    "Não foi possível perceber de que tema queres o resumo. Pergunta qual é o "
    "tema numa frase, em vez de fazeres um resumo."
)
NO_MATCH_INSTRUCTION = (
    'Não há mensagens sobre "{topic}" no histórico desta conversa. Diz isso numa '
    "frase e não inventes um resumo."
)
UNREADABLE_INSTRUCTION = (
    "Não foi possível ler o histórico agora. Diz que não conseguiste fazer o "
    "resumo, em vez de inventares um."
)


def recap_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """`chat.recap` from config merged over DEFAULTS."""
    section = ((config.get("chat") or {}).get("recap") or {})
    return {**DEFAULTS, **section}


def topic_keywords(topic: str, max_keywords: int = 6) -> List[str]:
    """Folded content words of the topic, in order, deduplicated."""
    keywords: List[str] = []
    for token in re.findall(r"[a-z0-9]+", _fold(topic or "")):
        if len(token) >= 3 and token not in _STOPWORDS and token not in keywords:
            keywords.append(token)
    return keywords[:max_keywords]


def load_rows(log_dir: Any, scope: str) -> List[Dict[str, Any]]:
    """Every logged message this scope may read, oldest first.

    ``readable_scopes`` is the rule retrieval follows (a DM reads shared plus
    itself), unlike ``tally``, which reads only the scope's own file: a recap
    joins a semantic half that already crosses into shared with this log half,
    and both halves must see the same messages.
    """
    log = MessageLog(str(log_dir))
    seen: Set[str] = set()
    rows: List[Dict[str, Any]] = []
    for readable in readable_scopes(scope or SHARED):
        for row in log.read(readable):
            row_id = row.get("id")
            if row_id is not None:
                if row_id in seen:
                    continue
                seen.add(row_id)
            text = strip_failed_descriptions((row.get("text") or "").strip())
            if not text:
                continue
            rows.append({**row, "text": text})
    rows.sort(key=lambda row: int(row.get("timestamp") or 0))
    return rows


def keyword_pattern(keyword: str) -> Pattern:
    """A prefix match at a word start, on folded text: "restaurante" finds "restaurantes"."""
    return re.compile(rf"\b{re.escape(keyword)}")


def select_rows(
    rows: List[Dict[str, Any]],
    keywords: Sequence[str],
    windows: Sequence[Tuple[int, int]] = (),
    neighbour_rows: int = 3,
    neighbour_gap_seconds: int = 1800,
    common_ratio: float = 0.02,
    common_floor: int = 25,
) -> Tuple[List[Dict[str, Any]], Set[int]]:
    """The rows on the topic, oldest first, and the indexes (into the returned
    list) of the seed rows.

    A keyword found in more than ``max(common_floor, common_ratio × rows)``
    messages is common: it seeds only together with a second keyword, so
    "almoço" alone does not pull in every lunch. A rare keyword seeds on its own.
    """
    folded = [_fold(row.get("text") or "") for row in rows]
    patterns = {keyword: keyword_pattern(keyword) for keyword in keywords}
    matches = [{keyword for keyword, pattern in patterns.items() if pattern.search(text)}
               for text in folded]
    threshold = max(common_floor, common_ratio * len(rows))
    common = {keyword for keyword in keywords
              if sum(1 for found in matches if keyword in found) > threshold}
    needed = min(2, len(keywords))

    seed_indexes: Set[int] = set()
    for index, found in enumerate(matches):
        if not found:
            continue
        if (found - common) or len(found) >= needed:
            seed_indexes.add(index)

    selected: Set[int] = set(seed_indexes)
    by_chat: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        by_chat.setdefault(row.get("chat_id") or "", []).append(index)
    for indexes in by_chat.values():
        position_of = {row_index: position for position, row_index in enumerate(indexes)}
        for seed in seed_indexes:
            if seed not in position_of:
                continue
            position = position_of[seed]
            seed_ts = int(rows[seed].get("timestamp") or 0)
            for near in indexes[max(0, position - neighbour_rows):position + neighbour_rows + 1]:
                if abs(int(rows[near].get("timestamp") or 0) - seed_ts) <= neighbour_gap_seconds:
                    selected.add(near)

    for index, row in enumerate(rows):
        stamp = int(row.get("timestamp") or 0)
        if any(start <= stamp <= end for start, end in windows):
            selected.add(index)

    ordered = sorted(selected)
    seeds = {position for position, index in enumerate(ordered) if index in seed_indexes}
    return [rows[index] for index in ordered], seeds


def _parse_utc(value: Any) -> datetime:
    """The ingester writes chunk times as UTC-naive ISO strings (``ingest._iso``)."""
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def chunk_windows(chunks: Sequence[Dict[str, Any]],
                  scope: str) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Live hits become time windows into the log; older hits are kept as text.

    The scope check is load-bearing: the windows are applied to log rows with no
    other scope check behind them.
    """
    windows: List[Tuple[int, int]] = []
    archive: List[str] = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        if not is_readable(metadata.get("scope"), scope):
            continue
        if metadata.get("source") == "live":
            try:
                start = _parse_utc(chunk.get("timestamp_start") or metadata.get("timestamp_start"))
                end = _parse_utc(chunk.get("timestamp_end") or metadata.get("timestamp_end"))
            except (ValueError, TypeError):
                continue
            windows.append((int(start.timestamp()), int(end.timestamp())))
        else:
            text = chunk.get("text") or ""
            if text:
                archive.append(text)
    return windows, archive


def format_line(row: Dict[str, Any], resolver: Any, tz: Any) -> str:
    """One transcript line, in local time, with the canonical sender name."""
    sender = (row.get("sender") or "Alguém").strip()
    if resolver is not None:
        sender = resolver.resolve(sender)
    when = datetime.fromtimestamp(int(row.get("timestamp") or 0), tz).strftime("%d/%m %H:%M")
    quoted = (row.get("reply_to_text") or "").strip()
    if quoted:
        return f'[{when}] {sender} (a responder a "{truncate_history_line(quoted, 12)}"): {row["text"]}'
    return f"[{when}] {sender}: {row['text']}"


def fit_budget(lines: List[str], seeds: Set[int], max_chars: int) -> Tuple[List[str], int]:
    """Drop lines until the total fits: neighbours/window rows oldest-first, then
    seeds oldest-first. Returns (kept lines in original order, dropped count).

    A plan is decided late in a thread, so the oldest material is what goes first.
    """
    kept = set(range(len(lines)))
    total = sum(len(line) + 1 for line in lines)
    order = ([index for index in range(len(lines)) if index not in seeds]
             + [index for index in range(len(lines)) if index in seeds])
    dropped = 0
    for index in order:
        if total <= max_chars:
            break
        kept.discard(index)
        total -= len(lines[index]) + 1
        dropped += 1
    return [lines[index] for index in sorted(kept)], dropped


def protect_speaker_lines(text: str, speaker: str) -> str:
    """Prefix "- " to any line that starts with "<speaker>:".

    ``clean_response`` cuts a reply at the first line that starts with the
    asker's name, taking it for a hallucinated new turn. A recap has one line per
    person under *Quem disse o quê*, so the asker's own line would cut the rest.
    """
    if not speaker:
        return text
    lines = []
    for line in (text or "").split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(f"{speaker}:"):
            line = f"{line[:len(line) - len(stripped)]}- {stripped}"
        lines.append(line)
    return "\n".join(lines)


def gather(
    config: Dict[str, Any],
    topic: str,
    scope: Optional[str],
    retriever: Any = None,
    resolver: Any = None,
    log_dir: Any = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """(context_block, instruction, stats) for one recap turn. Never raises
    from the retriever; a log read error propagates to the caller."""
    cfg = recap_config(config)
    scope = scope or SHARED
    log_dir = log_dir or (config.get("whatsapp") or {}).get("message_log_dir", "data/live_messages")
    tz = ZoneInfo(((config.get("power") or {}).get("timezone")) or "Europe/Lisbon")

    topic = " ".join((topic or "").split())
    stats: Dict[str, Any] = {
        "recap_topic": topic, "recap_keywords": [], "recap_rows": 0, "recap_seed_rows": 0,
        "recap_semantic_chunks": 0, "recap_archive_chunks": 0, "recap_dropped_rows": 0,
    }
    if not topic:
        return "", NO_TOPIC_INSTRUCTION, stats

    keywords = topic_keywords(topic, int(cfg["max_keywords"]))
    stats["recap_keywords"] = keywords

    windows: List[Tuple[int, int]] = []
    archive: List[str] = []
    if retriever is not None:
        try:
            chunks = retriever.retrieve(topic, top_k=int(cfg["semantic_top_k"]), scope=scope)
        except Exception as exc:  # noqa: BLE001 — the log half still answers
            print(f"⚠️  recap retrieval failed: {exc}")
            chunks = []
        windows, archive = chunk_windows(chunks or [], scope)
        archive = archive[:int(cfg["max_archive_chunks"])]
        stats["recap_semantic_chunks"] = len(windows)
        stats["recap_archive_chunks"] = len(archive)

    rows = load_rows(log_dir, scope)
    selected, seeds = select_rows(rows, keywords, windows, int(cfg["neighbour_rows"]),
                                  int(cfg["neighbour_gap_minutes"]) * 60,
                                  float(cfg["common_ratio"]), int(cfg["common_floor"]))
    lines = [format_line(row, resolver, tz) for row in selected]
    budget = int(cfg["max_chars"]) - sum(len(text) + 1 for text in archive)
    kept, dropped = fit_budget(lines, seeds, max(0, budget))
    stats["recap_rows"] = len(kept)
    stats["recap_seed_rows"] = len(seeds)
    stats["recap_dropped_rows"] = dropped

    if not kept and not archive:
        return "", NO_MATCH_INSTRUCTION.format(topic=topic), stats

    block = (f'Mensagens do histórico sobre "{topic}" ({len(kept)} mensagens, as mais '
             f"antigas primeiro; nem todas são sobre o tema):\n" + "\n".join(kept))
    if dropped:
        block += f"\n({dropped} mensagens mais antigas sobre o tema não couberam aqui.)"
    if archive:
        block += "\n\nConversas antigas que podem ser sobre o tema:\n" + "\n---\n".join(archive)
    return block, RECAP_INSTRUCTION.format(topic=topic), stats
