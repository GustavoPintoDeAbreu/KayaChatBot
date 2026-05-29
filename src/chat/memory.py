"""
Local session memory for KayaChatBot.

Stores conversation history to a local JSON file only — never to any database
or external service. Privacy is a core requirement: all data stays on-device.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

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
            tmp_file = self.history_file.with_suffix(self.history_file.suffix + ".tmp")
            tmp_file.write_text(
                json.dumps(to_save, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_file, self.history_file)
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
