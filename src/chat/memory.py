"""
Local session memory for KayaChatBot.

Stores conversation history to a local JSON file only — never to any database
or external service. Privacy is a core requirement: all data stays on-device.
"""

import json
import logging
import os
import threading
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionMemory:
    """Persists chat history to a local JSON file between sessions.

    The file is stored at the path specified in config (default:
    ``data/chat_history.json``). It contains a simple list of message strings
    in the format ``"<name>: <message>"``.

    This class is intentionally simple — no encryption, no cloud sync,
    no external dependencies beyond the standard library.
    """

    MAX_SAVED_MESSAGES = 100  # Hard cap to prevent unbounded file growth

    def __init__(self, history_file: str = "data/chat_history.json"):
        self.history_file = Path(history_file)
        # Resolve relative to project root if not absolute
        if not self.history_file.is_absolute():
            # Try to resolve relative to the project root (2 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            self.history_file = project_root / self.history_file

    def load(self) -> Optional[List[str]]:
        """Load history from local file. Returns None if file doesn't exist or is invalid."""
        if not self.history_file.exists():
            return None
        try:
            raw = self.history_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, list):
                logger.warning("Chat history file has unexpected format, ignoring.")
                return None
            # Validate entries are all strings
            history = [str(entry) for entry in data if isinstance(entry, str)]
            logger.debug("Loaded %d messages from %s", len(history), self.history_file)
            return history if history else None
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load chat history from %s: %s", self.history_file, exc)
            return None

    def save(self, history: List[str]) -> bool:
        """Save history to local file. Returns True on success, False on failure.
        
        Caps the stored history at MAX_SAVED_MESSAGES to prevent unbounded growth.
        """
        if not history:
            return True
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            # Apply hard cap before saving
            to_save = history[-self.MAX_SAVED_MESSAGES:]
            # Atomic write: write to a temp file in the same directory, then
            # os.replace() so a crash mid-write can't corrupt the history file.
            # The temp name carries the writer's identity because it used to be a
            # single fixed ".tmp": two threads saving the same chat wrote to one
            # path and replaced it out from under each other, so the save failed
            # outright and a concurrent reader could catch the real file
            # mid-replace, read invalid JSON, and be told the chat had no history
            # at all. Measured at 60-95% of lines lost under three writers.
            tmp_file = self.history_file.with_suffix(
                f"{self.history_file.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                tmp_file.write_text(
                    json.dumps(to_save, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(tmp_file, self.history_file)
            finally:
                # A failed replace must not leave the scratch file behind.
                if tmp_file.exists():
                    tmp_file.unlink(missing_ok=True)
            return True
        except OSError as exc:
            logger.warning("Failed to save chat history to %s: %s", self.history_file, exc)
            return False

    def clear(self) -> bool:
        """Delete the history file. Returns True on success."""
        try:
            if self.history_file.exists():
                self.history_file.unlink()
            return True
        except OSError as exc:
            logger.warning("Failed to clear chat history at %s: %s", self.history_file, exc)
            return False


def _safe_key(key: str) -> str:
    """Turn a WhatsApp chat id (e.g. ``"3519...@g.us"``) into a safe filename."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return sanitized[:120] or "unknown"


class KeyedSessionMemory:
    """Per-conversation history for the WhatsApp bridge.

    Unlike ``SessionMemory`` (one global file for the single-user CLI), WhatsApp
    has many independent conversations — one per DM and one per group. This keeps
    a separate rolling history file per chat id under ``base_dir`` so a group's
    context never leaks into a DM (and vice versa). Each file is a list of
    ``"<who>: <text>"`` lines, reusing ``SessionMemory`` for the atomic-write and
    capping behaviour. Privacy invariant is preserved: everything stays on disk
    locally, nothing is sent anywhere.
    """

    def __init__(self, base_dir: str = "data/whatsapp_sessions", max_lines: int = 20):
        self.base_dir = Path(base_dir)
        if not self.base_dir.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.base_dir = project_root / base_dir
        self.max_lines = max_lines
        self._stores: Dict[str, SessionMemory] = {}
        # Guards the two dicts below; `_locks` then guards each chat's file.
        self._lock = threading.Lock()
        self._locks: Dict[str, threading.Lock] = {}

    def _store(self, chat_id: str) -> SessionMemory:
        with self._lock:
            if chat_id not in self._stores:
                path = self.base_dir / f"{_safe_key(chat_id)}.json"
                self._stores[chat_id] = SessionMemory(str(path))
            return self._stores[chat_id]

    def _chat_lock(self, chat_id: str) -> threading.Lock:
        """One lock per chat, so two chats never wait on each other."""
        with self._lock:
            if chat_id not in self._locks:
                self._locks[chat_id] = threading.Lock()
            return self._locks[chat_id]

    def recent(self, chat_id: str, limit: Optional[int] = None) -> List[str]:
        """Return the most recent lines for a chat (oldest→newest)."""
        history = self._store(chat_id).load() or []
        limit = limit if limit is not None else self.max_lines
        return history[-limit:]

    def append(self, chat_id: str, line: str) -> None:
        """Append one ``"<who>: <text>"`` line, capping the rolling window.

        Locked per chat, because this is a read-modify-write and more than one
        thread does it. The image path is the clear case: the webhook thread
        appends "(a preparar uma imagem…)" while the queue worker appends
        "(imagem enviada)" for the previous job, and whichever loaded first wins
        — the other line is simply gone. Losing either is what makes the bot
        answer "então e a foto?" with no idea what it agreed to make.
        """
        with self._chat_lock(chat_id):
            store = self._store(chat_id)
            history = store.load() or []
            history.append(line)
            store.save(history[-self.max_lines:])


class ChatPreferences:
    """Sticky per-chat settings, currently just the reply modality.

    "Responde-me só em áudio" is a standing instruction, not a per-message one —
    it holds until the user says otherwise, regardless of whether they later send
    text or a voice note. That makes it state, so it lives on disk next to the
    per-chat history and survives a restart.

    Same isolation guarantee as ``KeyedSessionMemory``: one file per chat, so a
    group's preference never affects a DM. Uses its own atomic write rather than
    ``SessionMemory`` because that class stores (and slices) a list, not a dict.
    """

    OUTPUT_TEXT = "text"
    OUTPUT_AUDIO = "audio"
    VALID_OUTPUT_MODES = (OUTPUT_TEXT, OUTPUT_AUDIO)

    def __init__(self, base_dir: str = "data/whatsapp_prefs"):
        self.base_dir = Path(base_dir)
        if not self.base_dir.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.base_dir = project_root / base_dir

    def _path(self, chat_id: str) -> Path:
        return self.base_dir / f"{_safe_key(chat_id)}.json"

    def get(self, chat_id: str) -> Dict[str, Any]:
        path = self._path(chat_id)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, chat_id: str, prefs: Dict[str, Any]) -> bool:
        path = self._path(chat_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
            return True
        except OSError as exc:
            logger.warning("Failed to save chat prefs to %s: %s", path, exc)
            return False

    def output_mode(self, chat_id: str) -> str:
        """``"text"`` (default) or ``"audio"``."""
        mode = self.get(chat_id).get("output_mode", self.OUTPUT_TEXT)
        return mode if mode in self.VALID_OUTPUT_MODES else self.OUTPUT_TEXT

    def set_output_mode(self, chat_id: str, mode: str) -> bool:
        if mode not in self.VALID_OUTPUT_MODES:
            raise ValueError(f"unknown output mode: {mode!r}")
        prefs = self.get(chat_id)
        prefs["output_mode"] = mode
        return self._save(chat_id, prefs)
