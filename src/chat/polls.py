"""Build a WhatsApp poll from what a chat has been discussing.

When the group asks the bot to turn the options it has been talking about into a
poll, the bot reads the whole log of THAT chat, has the local model work out
which options are still in play, and posts a native multiple-choice poll.

This module is deliberately torch-free: it reads the message log, shapes the
prompt, parses the model's answer and records what was sent. The model call
itself is injected by the caller (the engine holds the backend and the GPU
lock).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.chat.response_utils import truncate_history_line
from src.chat.tally import _fold
from src.data.message_log import MessageLog

BASE_DIR = Path(__file__).resolve().parents[2]

STATUS_OK = "ok"
STATUS_NO_MESSAGES = "no_messages"
STATUS_ERROR = "error"
STATUS_DISABLED = "disabled"
STATUS_NO_CHAT = "no_chat"

DEFAULT_QUESTION = "Qual preferem?"
NO_CHAT_LINE = "Só faço sondagens no WhatsApp."

SYSTEM_PROMPT = (
    "Lês uma conversa de WhatsApp de um grupo de amigos e preparas uma sondagem. "
    "Não inventas nada: cada opção tem de ter sido proposta por alguém na conversa."
)

RULES = (
    "Uma opção sai quando foi recusada ou descartada (fechado, caro demais, sem mesa, "
    "alguém vetou e ninguém insistiu) ou quando foi trocada por outra. Uma opção que "
    "voltou a ser proposta depois de recusada está em jogo.\n"
    "Junta nomes diferentes do mesmo sítio numa só opção, com o nome mais completo que aparecer.\n"
    "Ignora tudo o que não for sobre este tema.\n"
    "Se o pedido disser quais são as opções, usa exatamente essas."
)

FINAL_INSTRUCTION = (
    "Responde só neste formato, uma linha por item, sem mais nada:\n"
    "PERGUNTA: <a pergunta da sondagem, curta>\n"
    "OPÇÃO: <nome>\n"
    "FORA: <nome> | <motivo em poucas palavras>\n"
    "Põe as linhas OPÇÃO pela ordem de apoio, a mais apoiada primeiro."
)

MAP_INSTRUCTION = (
    "Este é só um excerto da conversa. Diz como estavam as opções sobre este tema "
    "no FIM deste excerto, uma linha por opção:\n"
    "EM JOGO: <nome>\n"
    "FORA: <nome> | <motivo em poucas palavras>\n"
    "Se o excerto não fala deste tema, responde só NADA."
)


def settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """The poll settings, with defaults for anything missing."""
    section = (config.get("chat", {}) or {}).get("poll", {}) or {}
    return {
        "enabled": bool(section.get("enabled", False)),
        "store": section.get("store", "data/polls.jsonl"),
        "lookback_days": int(section.get("lookback_days", 90)),
        "max_messages": int(section.get("max_messages", 2000)),
        "line_max_words": int(section.get("line_max_words", 80)),
        "chunk_words": int(section.get("chunk_words", 5000)),
        "max_chunks": int(section.get("max_chunks", 6)),
        "max_options": int(section.get("max_options", 12)),
        "max_option_chars": int(section.get("max_option_chars", 100)),
        "max_question_chars": int(section.get("max_question_chars", 200)),
        "max_dropped_listed": int(section.get("max_dropped_listed", 5)),
        "max_new_tokens": int(section.get("max_new_tokens", 400)),
        "temperature": float(section.get("temperature", 0.2)),
    }


def log_dir(config: Dict[str, Any]) -> str:
    return (config.get("whatsapp") or {}).get("message_log_dir", "data/live_messages")


def store_path(config: Dict[str, Any]) -> Path:
    """Where sent polls are recorded, resolved against the repo root."""
    path = Path(settings(config)["store"])
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def load_chat_messages(base_dir: str, scope: str, chat_id: str,
                       since_ts: int = 0, max_messages: int = 2000) -> List[Dict[str, Any]]:
    """The log of one chat, newest ``max_messages`` in chronological order.

    Community sub-groups all append to the same scope file, so the log is
    filtered by ``chat_id`` as well as by scope.
    """
    log = MessageLog(base_dir)
    seen: set = set()
    records: List[Dict[str, Any]] = []
    for rec in log.read(scope, after_ts=since_ts):
        if rec.get("chat_id") != chat_id:
            continue
        if (rec.get("scope") or scope) != scope:
            continue
        rid = rec.get("id")
        if rid is not None and rid in seen:
            continue
        if rid is not None:
            seen.add(rid)
        records.append(rec)
    records.sort(key=lambda rec: int(rec.get("timestamp") or 0))
    return records[-max_messages:]


def format_lines(records: Sequence[Dict[str, Any]], max_words: int) -> List[str]:
    """One line per message, oldest first, shaped for the model to read."""
    lines: List[str] = []
    for rec in records:
        ts = int(rec.get("timestamp") or 0)
        when = datetime.fromtimestamp(ts).strftime("%d/%m %H:%M") if ts else "??/?? ??:??"
        sender = rec.get("sender") or "Alguém"
        text = " ".join(str(rec.get("text") or "").split())
        reply_to = " ".join(str(rec.get("reply_to_text") or "").split())
        if reply_to:
            text = f'(a responder a: "{reply_to}") {text}'
        lines.append(truncate_history_line(f"[{when}] {sender}: {text}", max_words))
    return lines


def chunk_lines(lines: Sequence[str], max_words: int, max_chunks: int) -> List[List[str]]:
    """Greedy chronological chunks, keeping only the newest ``max_chunks``."""
    chunks: List[List[str]] = []
    current: List[str] = []
    count = 0
    for line in lines:
        words = len(line.split())
        if current and count + words > max_words:
            chunks.append(current)
            current = []
            count = 0
        current.append(line)
        count += words
    if current:
        chunks.append(current)
    return chunks[-max_chunks:]


def _single_prompt(topic: str, request: str, lines: Sequence[str]) -> str:
    return (
        f"Tema da sondagem: {topic}\n"
        f"Pedido: {request}\n\n"
        f"Conversa (por ordem, a mais antiga primeiro):\n"
        + "\n".join(lines) + "\n\n"
        + RULES + "\n\n" + FINAL_INSTRUCTION
    )


def _map_prompt(topic: str, index: int, total: int, lines: Sequence[str]) -> str:
    return (
        f"Tema da sondagem: {topic}\n\n"
        f"Excerto {index} de {total}:\n"
        + "\n".join(lines) + "\n\n"
        + RULES + "\n\n" + MAP_INSTRUCTION
    )


def _reduce_prompt(topic: str, request: str, notes: Sequence[str]) -> str:
    return (
        f"Tema da sondagem: {topic}\n"
        f"Pedido: {request}\n\n"
        "Notas tiradas de cada excerto da conversa, por ordem (o que vem num excerto mais recente vale mais):\n"
        + "\n\n".join(notes) + "\n\n"
        + RULES + "\n\n" + FINAL_INSTRUCTION
    )


@dataclass
class PollDraft:
    question: str = ""
    options: List[str] = field(default_factory=list)
    dropped: List[Tuple[str, str]] = field(default_factory=list)
    invented: List[str] = field(default_factory=list)
    messages_read: int = 0
    chunks: int = 0
    status: str = STATUS_OK

    def telemetry(self) -> Dict[str, Any]:
        return {
            "poll_status": self.status,
            "poll_messages_read": self.messages_read,
            "poll_chunks": self.chunks,
            "poll_options": len(self.options),
            "poll_dropped": [name for name, _ in self.dropped],
            "poll_invented": list(self.invented),
        }


# Leading list numbering, but only when it is actually numbering: a bullet, or a
# digit run followed by "." or ")". A bare digit run is part of the name — "13 de
# Dezembro", "20/12", "12h30" — and stripping it would fold two different days
# into one option.
_BULLET_RE = re.compile(r"^\s*(?:[-*•·]+|\d+[.)])\s*")
_QUOTE_CHARS = "\"“”'«»"


def clean_item(text: str) -> str:
    """Strip the model's formatting off one option or question line."""
    text = _BULLET_RE.sub("", text or "")
    text = text.strip()
    while len(text) >= 2 and text[0] in _QUOTE_CHARS and text[-1] in _QUOTE_CHARS:
        text = text[1:-1].strip()
    for dash in (" — ", " – ", " - "):
        text = text.replace(dash, ", ")
    text = " ".join(text.split())
    return text.rstrip(".;,:")


_PERGUNTA_RE = re.compile(r"^PERGUNTA\s*:\s*(.+)$", re.IGNORECASE)
_OPCAO_RE = re.compile(r"^OP[ÇC][ÃA]O\s*:\s*(.+)$", re.IGNORECASE)
_FORA_RE = re.compile(r"^FORA\s*:\s*(.+)$", re.IGNORECASE)
_EM_JOGO_RE = re.compile(r"^EM\s+JOGO\s*:\s*(.+)$", re.IGNORECASE)


def parse_final(raw: str) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    """(question, options, dropped) from the model's final answer."""
    question = ""
    options: List[str] = []
    dropped: List[Tuple[str, str]] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = _PERGUNTA_RE.match(line)
        if match:
            question = clean_item(match.group(1))
            continue
        match = _OPCAO_RE.match(line)
        if match:
            name = clean_item(match.group(1))
            if name:
                options.append(name)
            continue
        match = _FORA_RE.match(line)
        if match:
            item = match.group(1)
            name, _, reason = item.partition("|")
            name = clean_item(name)
            reason = clean_item(reason) or "descartada"
            if name:
                dropped.append((name, reason))
    return question, options, dropped


def parse_notes(raw: str) -> List[str]:
    """The lines worth carrying forward from one map pass, or [] for NADA."""
    notes: List[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if _EM_JOGO_RE.match(line) or _FORA_RE.match(line):
            notes.append(line)
    return notes


_GENERIC = {
    "restaurante", "tasca", "casa", "cervejaria", "marisqueira", "pizzaria",
    "churrasqueira", "bar", "cafe", "sitio", "lugar", "opcao",
    "dos", "das", "que", "para", "com",
}


def is_grounded(option: str, corpus: str) -> bool:
    """Whether the option's name actually appears in the conversation.

    ``corpus`` must already be folded. An option made of generic words only is
    grounded when it appears verbatim; otherwise any significant token counts.
    """
    tokens = [t for t in re.findall(r"[a-z0-9]{3,}", _fold(option)) if t not in _GENERIC]
    if tokens:
        return any(token in corpus for token in tokens)
    return _fold(option) in corpus


def _cut(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip()


def finalize(question: str, options: Sequence[str], dropped: Sequence[Tuple[str, str]],
             corpus: str, settings: Dict[str, Any]) -> PollDraft:
    """Dedupe, ground and bound the model's answer into a sendable draft."""
    seen: set = set()
    kept: List[str] = []
    invented: List[str] = []
    for option in options:
        fold = _fold(option)
        if fold in seen:
            continue
        seen.add(fold)
        if not is_grounded(option, corpus):
            invented.append(option)
            continue
        kept.append(_cut(option, settings["max_option_chars"]))

    dropped_list: List[Tuple[str, str]] = list(dropped)
    for option in kept[settings["max_options"]:]:
        dropped_list.append((option, "não coube na sondagem"))
    kept = kept[: settings["max_options"]]

    kept_folds = {_fold(option) for option in kept}
    dropped_seen: set = set()
    final_dropped: List[Tuple[str, str]] = []
    for name, reason in dropped_list:
        fold = _fold(name)
        if fold in dropped_seen or fold in kept_folds:
            continue
        # A dropped name the model invented is not something the group said, so
        # it must not surface in the "Deixei de fora" line either.
        if not is_grounded(name, corpus):
            continue
        dropped_seen.add(fold)
        final_dropped.append((name, reason))

    question = _cut(clean_item(question), settings["max_question_chars"]) or DEFAULT_QUESTION
    return PollDraft(question=question, options=kept, dropped=final_dropped,
                     invented=invented)


def draft_poll(generate: Callable[[List[Dict[str, str]], int], str], *,
               topic: str, request: str,
               records: Sequence[Dict[str, Any]],
               settings: Dict[str, Any]) -> PollDraft:
    """Work the chat log into a poll draft. Never raises.

    One chunk means one generation. Several means one map pass per chunk, then
    one reduce pass over the notes.
    """
    if not records:
        return PollDraft(status=STATUS_NO_MESSAGES)

    lines = format_lines(records, settings["line_max_words"])
    chunks = chunk_lines(lines, settings["chunk_words"], settings["max_chunks"])
    corpus = _fold("\n".join(
        f"{rec.get('text') or ''} {rec.get('reply_to_text') or ''}" for rec in records))

    try:
        if len(chunks) == 1:
            raw = generate([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _single_prompt(topic, request, chunks[0])},
            ], settings["max_new_tokens"])
            question, options, dropped = parse_final(raw)
        else:
            notes: List[str] = []
            for index, chunk in enumerate(chunks, start=1):
                raw = generate([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _map_prompt(topic, index, len(chunks), chunk)},
                ], settings["max_new_tokens"])
                kept_notes = parse_notes(raw)
                if kept_notes:
                    notes.append(f"Excerto {index}:\n" + "\n".join(kept_notes))
            raw = generate([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _reduce_prompt(topic, request, notes)},
            ], settings["max_new_tokens"])
            question, options, dropped = parse_final(raw)
    except Exception:
        return PollDraft(status=STATUS_ERROR, messages_read=len(records),
                         chunks=len(chunks))

    draft = finalize(question, options, dropped, corpus, settings)
    draft.messages_read = len(records)
    draft.chunks = len(chunks)
    return draft


def left_out_line(dropped: Sequence[Tuple[str, str]], template: str, limit: int) -> str:
    """A short line naming what the poll left out, or "" when nothing did."""
    if not dropped:
        return ""
    items = [f"{name} ({reason})" for name, reason in dropped[:limit]]
    extra = len(dropped) - len(items)
    if extra > 0:
        items.append(f"e mais {extra}")
    return template.format(items=", ".join(items))


class PollStore:
    """The append-only record of every poll the bot sent."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def record(self, entry: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load(self, chat_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chat_id is None or entry.get("chat_id") == chat_id:
                entries.append(entry)
        return entries
