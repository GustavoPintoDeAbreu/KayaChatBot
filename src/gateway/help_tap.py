"""``/help`` everywhere, answered by the Pi, from the lab's command registry.

Unlike the idea, homelab and assistant taps, ``/help`` is handed to no inbox and
needs no PC: the reply is built here, so it works while the PC is off. Like the
other taps, a message it claims is never journaled, never forwarded to the PC and
never becomes "something the group said".

What it claims (since 2026-11-25; before that, only the owner's DM ``/help``):

* ``/help`` / ``/ajuda`` in a group (``@Kaya /help`` too): the group's commands only.
  The owner's commands are never shown in a group, whoever asks.
* the owner's DM ``/help``: everything, his own commands included.
* the owner's DM ``/word`` that no earlier tap took and that is not a Kaya command:
  "Não conheço /word. Querias dizer /X?" instead of going to the PC. (The idea and
  homelab taps run first, so ``/idea`` and ``/homelab*`` never get here.)

A member's DM is left to the PC, whose adapter answers ``/help`` and a wrong
``/word`` from the same registry (it knows the whitelist; the Pi does not).

The list comes from ``src/chat/commands.py``: labwatch's ``/api/commands`` (every
repo's ``lab.yaml`` ``commands:``, so the idea pipeline's are there too), on the same
origin as ``GATEWAY_HELP_URL`` unless ``GATEWAY_COMMANDS_URL`` says otherwise; the
last good copy on disk when the hub is down; and Kaya's own ``lab.yaml``, baked into
the image, always.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Set
from urllib.parse import urlsplit

from src.chat import commands as registry
from src.chat.whatsapp_adapter import InboundMessage, _phone_from_alt

logger = logging.getLogger(__name__)

HELP_TAG = "❓[help]"
HELP_COMMANDS = ("/help", "/ajuda")
BAKED = Path(__file__).resolve().parent.parent.parent / "lab.yaml"
REFRESH_SECONDS = 600
FETCH_TIMEOUT = 3.0


def _fetch(url: str) -> List[dict]:
    import httpx

    response = httpx.get(url, timeout=FETCH_TIMEOUT)
    response.raise_for_status()
    rows = response.json().get("commands")
    if not isinstance(rows, list):
        raise ValueError("no commands list")
    return rows


@dataclass
class CommandSource:
    """The registry: the hub's list, else the cached copy, always with Kaya's own."""

    url: str = ""
    cache_path: Optional[Path] = None
    baked_path: Path = BAKED
    fetch: Callable[[str], List[dict]] = _fetch
    now: Callable[[], float] = time.time
    _live: Optional[List[registry.Command]] = field(default=None, repr=False)
    _fetched_at: float = field(default=0.0, repr=False)

    def _cached(self) -> List[registry.Command]:
        if self.cache_path is None:
            return []
        try:
            return registry.from_entries(json.loads(self.cache_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return []

    def commands(self) -> List[registry.Command]:
        if self.url and self.now() - self._fetched_at >= REFRESH_SECONDS:
            self._fetched_at = self.now()
            try:
                rows = self.fetch(self.url)
                self._live = registry.from_entries(rows)
                if self.cache_path is not None:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    self.cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 — the hub being down must not cost /help
                logger.warning("help: command registry not fetched from %s: %s", self.url, exc)
        found = self._live if self._live is not None else self._cached()
        return registry.merge(found, registry.load_lab_yaml(self.baked_path))


def _commands_url(environ: Mapping[str, str]) -> str:
    explicit = environ.get("GATEWAY_COMMANDS_URL", "").strip()
    if explicit:
        return explicit
    help_url = environ.get("GATEWAY_HELP_URL", "").strip()
    if not help_url:
        return ""
    parts = urlsplit(help_url)
    return f"{parts.scheme}://{parts.netloc}/api/commands" if parts.scheme and parts.netloc else ""


def _first_word(text: str) -> str:
    """The first word after any leading @mentions, lower-cased."""
    for word in (text or "").split():
        if not word.startswith("@"):
            return word.lower()
    return ""


@dataclass
class HelpTap:
    """Claims ``/help`` (groups, the owner's DM) and the owner's unknown ``/word``s."""

    help_url: str
    owner_numbers: Set[str]
    source: CommandSource = field(default_factory=CommandSource)

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Optional["HelpTap"]:
        """The tap, or None when no owner number is set (the feature is off).

        The owner is the idea pipeline's (``IDEA_OWNER_NUMBERS``) unless
        ``HELP_OWNER_NUMBERS`` says otherwise, the same convention as the
        assistant tap.
        """
        raw = environ.get("HELP_OWNER_NUMBERS") or environ.get("IDEA_OWNER_NUMBERS", "")
        owners = {_phone_from_alt(number.strip().lstrip("+")) for number in raw.split(",") if number.strip()}
        if not owners:
            return None
        data_dir = environ.get("GATEWAY_DATA_DIR", "").strip()
        source = CommandSource(url=_commands_url(environ),
                               cache_path=Path(data_dir) / "commands.json" if data_dir else None)
        return cls(environ.get("GATEWAY_HELP_URL", "").rstrip("/"), owners, source)

    def _is_owner(self, msg: InboundMessage) -> bool:
        candidates = {_phone_from_alt(msg.sender_phone), _phone_from_alt(msg.sender_id)}
        return bool((candidates - {""}) & self.owner_numbers)

    def classify(self, msg: InboundMessage) -> Optional[str]:
        """``"help"`` or ``"unknown"`` to answer, ``"echo"`` to drop, None to leave for Kaya."""
        text = (msg.text or "").strip()
        if msg.from_me:
            return "echo" if text.startswith(HELP_TAG) else None
        first = _first_word(text)
        if msg.is_group:
            return "help" if first in HELP_COMMANDS else None
        if not self._is_owner(msg):
            return None
        if first in HELP_COMMANDS:
            return "help"
        if first.startswith("/") and len(first) > 1 and first not in registry.tokens(self.source.commands()):
            return "unknown"
        return None

    def reply_text(self, msg: Optional[InboundMessage] = None, kind: str = "help") -> str:
        """The list for this chat (plus the Help-page link in the owner's DM), or the
        wrong-command line. With no message: the owner's DM list."""
        where = "group" if msg is not None and msg.is_group else "dm"
        is_owner = msg is None or (not msg.is_group and self._is_owner(msg))
        commands = self.source.commands()
        if kind == "unknown":
            return registry.unknown_reply(_first_word(msg.text if msg else ""), commands, where, is_owner)
        text = registry.help_text(commands, where, is_owner)
        if self.help_url and is_owner:
            return f"{text}\n\n📖 Página de ajuda: {self.help_url}"
        return text
