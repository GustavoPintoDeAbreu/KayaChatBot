"""Which sub-group of the Community a chat is, by name.

The group became a WhatsApp Community on 2026-09-29: the old group is "general"
and there are sub-groups for trips, cooking, the bot itself. They all share one
memory (``shared_chats``), and until this the bot could not tell them apart: it
did not know which room it was answering in, and a chunk of the trip channel
read exactly like a chunk of general. So a reply in the trip channel had no idea
it was in the trip channel, and "o que decidimos para a viagem?" asked in general
could not say where the decision was made.

The names come from WAHA (``group_info``'s ``subject``), learned by the adapter
and kept in ``data/whatsapp_chat_names.json`` (gitignored: a chat id is an
identifier). ``labels`` in the same file overrides a subject with something
shorter, by hand.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_PATH = "data/whatsapp_chat_names.json"
BASE_DIR = Path(__file__).resolve().parents[2]


class ChannelNames:
    """``{"names": {chat_id: subject}, "labels": {chat_id: label}, "checked": {chat_id: ts}}``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def label(self, chat_id: Optional[str]) -> str:
        """The channel's short name, ``""`` when unknown."""
        if not chat_id:
            return ""
        data = self._load()
        return str((data.get("labels") or {}).get(chat_id)
                   or (data.get("names") or {}).get(chat_id) or "").strip()

    def stale(self, chat_id: str, max_age_seconds: float, now: Optional[float] = None) -> bool:
        """Whether the name was never looked up, or not for ``max_age_seconds``."""
        checked = (self._load().get("checked") or {}).get(chat_id)
        if checked is None:
            return True
        return (now if now is not None else time.time()) - float(checked) >= max_age_seconds

    def remember(self, chat_id: str, subject: str, now: Optional[float] = None) -> None:
        """Store a group's subject (and when it was checked, even if it is empty)."""
        data = self._load()
        if subject:
            data.setdefault("names", {})[chat_id] = subject.strip()
        data.setdefault("checked", {})[chat_id] = int(now if now is not None else time.time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


def from_config(config: Dict[str, Any]) -> ChannelNames:
    raw = (config.get("whatsapp", {}) or {}).get("chat_names_file") or DEFAULT_PATH
    path = Path(raw)
    return ChannelNames(path if path.is_absolute() else BASE_DIR / path)


def prompt_line(label: str) -> str:
    """The line that tells the model which room it is answering in."""
    if not label:
        return ""
    return (f"Estás no canal «{label}» da comunidade Kaya (um dos grupos do grupo de amigos). "
            "O que vem das conversas pode ter sido dito noutro canal: quando importar, diz em qual.")
