"""
Sender identity resolution for WhatsApp message extraction.

Maps raw sender names from WhatsApp exports to canonical group member names
using a priority chain:

  1. Config manual overrides (highest priority — handles raw sender strings
     that cannot be matched by name tokens alone).
  2. Exact alias match (case-insensitive).
  3. Token match — each whitespace-delimited word in the sender name is
     checked against all member aliases.  Handles full names like
     "João Gil" (token "Gil" → member "Gil") or
     "Rafael Beirão Chamusca" (token "Chamusca" → member "Chamusca").
  4. If no member match is found the original sender name is returned so
     that non-group participants are preserved as-is.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Set


class SenderResolver:
    """Resolve raw sender names to canonical group member names."""

    def __init__(
        self,
        members_file: Path,
        sender_aliases: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        members_file:
            Path to ``group_members.json``.
        sender_aliases:
            Manual override map — key is the raw sender name exactly as it
            appears in the source data, value is the canonical member name
            (e.g. ``"Gil João"`` → ``"Gil"``).  Keys are matched
            case-sensitively against the raw sender string.
        """
        self._overrides: Dict[str, str] = sender_aliases or {}
        self._canonical_names: Set[str] = set()
        # alias_lowercase → canonical_name  (built from group_members.json)
        self._alias_lookup: Dict[str, str] = {}
        self._load_members(members_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_members(self, members_file: Path) -> None:
        with open(members_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Support both plain list format and {"members": [...]} dict format
        members_list = data if isinstance(data, list) else data.get("members", [])
        for member in members_list:
            name: str = member["name"]
            self._canonical_names.add(name)
            # Canonical name itself as a lookup token
            self._alias_lookup[name.lower()] = name
            for alias in member.get("aliases", []):
                self._alias_lookup[alias.lower()] = name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, raw_sender: str) -> str:
        """
        Resolve *raw_sender* to a canonical member name.

        Returns
        -------
        str
            The canonical member name when a match is found, otherwise the
            *raw_sender* string unchanged (non-group participants are
            preserved for context).
        """
        if not raw_sender:
            return raw_sender

        # 1. Config override
        if raw_sender in self._overrides:
            return self._overrides[raw_sender]

        # 2. Exact alias match (case-insensitive, stripped)
        sender_lower = raw_sender.lower().strip()
        if sender_lower in self._alias_lookup:
            return self._alias_lookup[sender_lower]

        # 3. Token match — each word of the name checked against aliases
        tokens = sender_lower.split()
        matches: Set[str] = set()
        for token in tokens:
            if token in self._alias_lookup:
                matches.add(self._alias_lookup[token])

        if len(matches) == 1:
            # Unambiguous single-member match
            return matches.pop()

        # Ambiguous (multiple members match) or no tokens matched — return
        # the raw name so non-member senders are preserved as-is.
        return raw_sender

    def is_member(self, name: str) -> bool:
        """Return True if *name* is a known canonical group member (case-insensitive)."""
        name_lower = name.lower()
        return any(n.lower() == name_lower for n in self._canonical_names)
