"""
Sender identity resolution for multi-source message extraction.

Maps raw sender names from Instagram (full names, usernames) and WhatsApp to
canonical group member names using a priority chain:

  1. Config manual overrides (highest priority — handles opaque usernames
     like "peteroupedro" that cannot be matched by name tokens alone).
  2. Exact alias match (case-insensitive).
  3. Token match — each whitespace-delimited word in the decoded sender name
     is checked against all member aliases.  Handles full names like
     "João Gil" (token "Gil" → member "Gil") or
     "Rafael Beirão Chamusca" (token "Chamusca" → member "Chamusca").
  4. If no member match is found the original (decoded) sender name is
     returned so that non-group participants are preserved as-is.
  5. Anonymous Instagram senders ("Instagram user") are mapped to None so
     callers can drop them.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Set


# Anonymous placeholder used by Instagram for deleted / private accounts
_INSTAGRAM_ANONYMOUS = "instagram user"


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
            appears in the source data (e.g. ``"peteroupedro"``), value is the
            canonical member name (e.g. ``"Peter"``).  Keys are matched
            case-sensitively against the raw *and* decoded sender string.
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

    @staticmethod
    def _decode_double_utf8(text: str) -> str:
        """Decode Instagram's double-encoded UTF-8 where necessary."""
        try:
            return text.encode("latin1").decode("utf-8")
        except Exception:
            return text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, raw_sender: str) -> Optional[str]:
        """
        Resolve *raw_sender* to a canonical member name.

        Returns
        -------
        str or None
            - The canonical member name when a match is found.
            - The decoded *raw_sender* string when no member matches (i.e. the
              sender is a non-group participant — preserved for context).
            - ``None`` for anonymous senders such as "Instagram user" (callers
              should drop these messages).
        """
        if not raw_sender:
            return raw_sender

        # Decode double-encoded UTF-8 (Instagram data often arrives this way)
        decoded: str = self._decode_double_utf8(raw_sender)

        # 1. Config override — try raw key first, then decoded key
        for key in (raw_sender, decoded):
            if key in self._overrides:
                return self._overrides[key]

        # 2. Drop anonymous Instagram senders
        if decoded.lower().strip() == _INSTAGRAM_ANONYMOUS:
            return None

        # 3. Exact alias match on decoded name (case-insensitive, stripped)
        decoded_lower = decoded.lower().strip()
        if decoded_lower in self._alias_lookup:
            return self._alias_lookup[decoded_lower]

        # 4. Token match — each word of the decoded name checked against aliases
        tokens = decoded_lower.split()
        matches: Set[str] = set()
        for token in tokens:
            if token in self._alias_lookup:
                matches.add(self._alias_lookup[token])

        if len(matches) == 1:
            # Unambiguous single-member match
            return matches.pop()

        # Ambiguous (multiple members match) or no tokens matched — return
        # the decoded name so non-member senders are preserved as-is.
        return decoded

    def is_member(self, name: str) -> bool:
        """Return True if *name* is a known canonical group member (case-insensitive)."""
        name_lower = name.lower()
        return any(n.lower() == name_lower for n in self._canonical_names)
