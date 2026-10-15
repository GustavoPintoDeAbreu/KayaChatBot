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
    "max_new_tokens": 700,
    "semantic_top_k": 8,
    "neighbour_rows": 3,
    "neighbour_gap_minutes": 30,
    "common_ratio": 0.02,
    "common_floor": 25,
    "max_chars": 40000,
    "max_archive_chunks": 3,
    "max_keywords": 6,
    "chat_days": 14,
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
    "discutimos", "discutido", "discutida", "discutiram", "discutir", "discussao", "falado",
    "falada", "falar", "combinado", "combinamos", "combinaram", "ficamos", "ficou", "ficaram",
    "quer", "querem", "queres", "acha", "acham", "achas", "propos", "propuseram", "sugeriu",
    "sugeriram", "mencionou", "mencionaram", "sido", "disto", "disso", "nisto", "nisso",
    "podes", "pode", "podem", "podias", "consegues", "quero", "queria", "gostava", "favor",
    "sff", "pls", "please", "can", "you", "could",
    "tldr", "resumao", "assunto", "assuntos", "tema", "temas", "passou", "passado", "passada",
    "ultimo", "ultima", "ultimos", "ultimas", "desde", "ate", "aconteceu",
    # periods: a time word is a window, not a keyword
    "hoje", "ontem", "anteontem", "antes", "semana", "fim", "noite", "dia", "dias",
    "today", "yesterday", "week", "weekend", "night", "last",
    # English
    "the", "and", "what", "who", "where", "when", "about", "did", "was", "were",
    "we", "our", "this", "that", "with", "for", "summary", "summarise", "summarize",
    "recap", "decide", "decided", "said", "say", "talked", "conversation",
    "discussed", "discuss", "mentioned", "everyone", "each", "want", "wants", "land", "landed",
})

RECAP_INSTRUCTION = (
    'Responde a «{request}» sobre "{topic}" usando só as mensagens do histórico '
    "que te foram dadas acima. Se a pergunta é específica (o que uma pessoa disse, "
    "quem propôs uma coisa, só o que se decidiu), responde só a isso, em poucas "
    "frases e sem secções. Se é um pedido geral de resumo, usa três partes, cada "
    "uma com o título em negrito do WhatsApp, com um só asterisco de cada lado: "
    "*Decidido*, *Em aberto* e *Quem disse o quê*, todos na língua da resposta, "
    "nunca misturados (em inglês: *Decided*, *Still open*, *Who said what*). Em "
    '*Quem disse o quê* escreve uma linha por pessoa, a começar por "- Nome:", com '
    "a posição ou proposta mais recente dessa pessoa; se mudou de ideias, diz de "
    "quê para quê. Uma opção que foi posta de parte ou abandonada não está em "
    "aberto: diz que caiu e, se a razão estiver nas mensagens, porquê. Junta as "
    "opções repetidas numa só. Só atribuis uma posição a alguém se houver uma "
    'mensagem dessa pessoa que a mostre. Se nada foi decidido, diz "nada decidido '
    'ainda". Ignora as mensagens que não são sobre o tema. Não inventes opções, '
    "datas, preços nem pessoas."
)
CHAT_INSTRUCTION = (
    "Responde a «{request}» usando só as mensagens desta conversa{period} dadas "
    "acima. Se a pergunta é específica (só o que se decidiu, o que uma pessoa "
    "disse, quem propôs uma coisa), responde só a isso, em poucas frases e sem "
    "secções. Se é um pedido geral de resumo: primeiro os assuntos principais, um "
    "a um, cada um com uma ou duas frases; depois, se houver, o que ficou decidido "
    "e o que está em aberto, e quem defendeu o quê nos pontos que importam. Usa "
    "títulos em negrito do WhatsApp, com um só asterisco de cada lado, todos na "
    "língua da resposta. Só atribuis uma posição a alguém se houver uma mensagem "
    "dessa pessoa que a mostre. Não inventes nada."
)
NO_MESSAGES_INSTRUCTION = (
    "Não há mensagens nesta conversa{period} para resumir. Diz isso numa frase e "
    "não inventes um resumo."
)
NO_MATCH_INSTRUCTION = (
    'Não há mensagens sobre "{topic}" no histórico desta conversa. Diz isso numa '
    "frase e não inventes um resumo."
)
UNREADABLE_INSTRUCTION = (
    "Não foi possível ler o histórico agora. Diz que não conseguiste fazer o "
    "resumo, em vez de inventares um."
)


# Verbs of asking for a recap, in any conjugation ("discutiu", "decidiram",
# "resumes", "conversámos"): listing every form missed some.
_REQUEST_STEMS = ("discut", "decid", "resum", "convers", "combin", "mencion", "suger", "propus")


def recap_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """`chat.recap` from config merged over DEFAULTS."""
    section = ((config.get("chat") or {}).get("recap") or {})
    return {**DEFAULTS, **section}


def topic_keywords(topic: str, max_keywords: int = 6) -> List[str]:
    """Folded content words of the topic, in order, deduplicated."""
    keywords: List[str] = []
    for token in re.findall(r"[a-z0-9]+", _fold(topic or "")):
        if (len(token) >= 3 and token not in _STOPWORDS and not token.startswith(_REQUEST_STEMS)
                and token not in keywords):
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


def period_window(text: str, tz: Any, now: Optional[datetime] = None
                  ) -> Optional[Tuple[int, int, str]]:
    """The period a request names ("hoje", "ontem", "esta semana"...), as epoch
    bounds and a label, or None. Same table the retriever uses for "ontem"."""
    from src.chat.retriever import _recency_window

    local_now = (now or datetime.now(tz)).astimezone(tz).replace(tzinfo=None)
    found = _recency_window(text or "", local_now)
    if found is None:
        return None
    start, end, label = found
    return (int(datetime.fromisoformat(start).replace(tzinfo=tz).timestamp()),
            int(datetime.fromisoformat(end).replace(tzinfo=tz).timestamp()), label)


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


_MARKDOWN_BOLD = re.compile(r"\*{2,}([^*\n]+?)\*{2,}")


def whatsapp_bold(text: str) -> str:
    """``**Decidido**`` (markdown) becomes ``*Decidido*`` (WhatsApp).

    Asked for WhatsApp bold, the model still wrote markdown on the real log
    (2026-10-15), which WhatsApp shows as literal asterisks, and once escaped
    them as well (``**\\*Decidido\\***``).
    """
    return _MARKDOWN_BOLD.sub(r"*\1*", (text or "").replace("\\*", "*"))


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
    request: str = "",
    chat_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """(context_block, instruction, stats) for one recap turn. Never raises
    from the retriever; a log read error propagates to the caller.

    Two kinds of request. One that names a subject ("o que decidimos sobre o
    almoço de natal?", "quem propôs o Fuso?") reads the messages on it. One that
    names none ("o que é que se discutiu?", "resume o que decidimos", "tl;dr")
    reads this chat's recent conversation: the router would otherwise pass the
    question itself as the topic, whose words are all request words, and find
    nothing. A period in either ("hoje", "ontem", "esta semana") bounds the rows.
    """
    cfg = recap_config(config)
    scope = scope or SHARED
    log_dir = log_dir or (config.get("whatsapp") or {}).get("message_log_dir", "data/live_messages")
    tz = ZoneInfo(((config.get("power") or {}).get("timezone")) or "Europe/Lisbon")

    topic = " ".join((topic or "").split())
    request = " ".join((request or "").split())
    period = period_window(f"{request} {topic}", tz, now)
    stats: Dict[str, Any] = {
        "recap_topic": topic, "recap_keywords": [], "recap_rows": 0, "recap_seed_rows": 0,
        "recap_semantic_chunks": 0, "recap_archive_chunks": 0, "recap_dropped_rows": 0,
        "recap_mode": "topic", "recap_period": period[2] if period else "", "recap_this_chat": False,
    }
    keywords = topic_keywords(topic, int(cfg["max_keywords"]))
    if request and not topic_keywords(request, int(cfg["max_keywords"])):
        keywords = []
    stats["recap_keywords"] = keywords
    rows = load_rows(log_dir, scope)
    if period:
        rows = [row for row in rows if period[0] <= int(row.get("timestamp") or 0) <= period[1]]

    if not keywords:
        return _chat_recap(cfg, rows, chat_id, period, resolver, tz, stats, now, request)

    windows: List[Tuple[int, int]] = []
    archive: List[str] = []
    if retriever is not None and not period:
        try:
            chunks = retriever.retrieve(topic, top_k=int(cfg["semantic_top_k"]), scope=scope)
        except Exception as exc:  # noqa: BLE001 — the log half still answers
            print(f"⚠️  recap retrieval failed: {exc}")
            chunks = []
        windows, archive = chunk_windows(chunks or [], scope)
        archive = archive[:int(cfg["max_archive_chunks"])]
        stats["recap_semantic_chunks"] = len(windows)
        stats["recap_archive_chunks"] = len(archive)

    def select(candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Set[int]]:
        return select_rows(candidates, keywords, windows, int(cfg["neighbour_rows"]),
                           int(cfg["neighbour_gap_minutes"]) * 60,
                           float(cfg["common_ratio"]), int(cfg["common_floor"]))

    # The discussion in the chat being asked comes first: "o que é que o Gil disse
    # sobre o restaurante?" in the lunch chat is about that chat's restaurants, not
    # every restaurant the main group mentioned in 90 days. Only a chat that never
    # discussed the topic reads the rest of its scope.
    selected, seeds = select([row for row in rows if row.get("chat_id") == chat_id]) if chat_id else ([], set())
    stats["recap_this_chat"] = bool(seeds)
    if not seeds:
        selected, seeds = select(rows)
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
    return block, RECAP_INSTRUCTION.format(request=request or topic, topic=topic), stats


def _chat_recap(cfg: Dict[str, Any], rows: List[Dict[str, Any]], chat_id: Optional[str],
                period: Optional[Tuple[int, int, str]], resolver: Any, tz: Any,
                stats: Dict[str, Any], now: Optional[datetime],
                request: str) -> Tuple[str, str, Dict[str, Any]]:
    """A recap of this chat's recent conversation: its own messages only (never
    another chat in the same scope), the named period or the last ``chat_days``,
    newest kept when it does not all fit."""
    stats["recap_mode"] = "chat"
    period_text = f" ({period[2]})" if period else ""
    if chat_id:
        rows = [row for row in rows if row.get("chat_id") == chat_id]
    if not period:
        current = (now or datetime.now(tz)).timestamp()
        since = current - float(cfg["chat_days"]) * 86400
        rows = [row for row in rows if int(row.get("timestamp") or 0) >= since]
    lines = [format_line(row, resolver, tz) for row in rows]
    kept, dropped = fit_budget(lines, set(), int(cfg["max_chars"]))
    stats["recap_rows"] = len(kept)
    stats["recap_dropped_rows"] = dropped
    if not kept:
        return "", NO_MESSAGES_INSTRUCTION.format(period=period_text), stats
    block = (f"Mensagens desta conversa{period_text} ({len(kept)} mensagens, as mais "
             f"antigas primeiro):\n" + "\n".join(kept))
    if dropped:
        block += f"\n({dropped} mensagens mais antigas não couberam aqui.)"
    return block, CHAT_INSTRUCTION.format(request=request or "resume a conversa", period=period_text), stats
