"""
language_validator.py

Utilities for validating that model responses comply with the language policy:
  - European Portuguese only
  - No Brazilian Portuguese expressions
  - No emojis
  - No mid-sentence language switches to English (outside of /en mode)

Used by the testing framework and the LLM judge scorer.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Allow running directly from src/testing/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.language_filters import (
    _BR_PATTERN,
    _EMOJI_PATTERN,
    contains_brazilian_portuguese,
    contains_emojis,
)

# ---------------------------------------------------------------------------
# English-detection heuristic
# Common English function words that would not normally appear in a PT-EU reply.
# ---------------------------------------------------------------------------

_ENGLISH_FUNCTION_WORDS: list[str] = [
    "the", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might",
    "and", "or", "but", "so", "if", "then",
    "he", "she", "they", "we", "you", "it",
    "his", "her", "their", "our", "your", "its",
    "not", "no", "yes", "for", "with", "from", "about",
    "there", "here", "where", "when", "how",
]

# Build a pattern that matches 3+ distinct English function words in a row
# (to avoid false positives on words like "a", "no" which also exist in PT)
_EN_HIGH_SIGNAL_WORDS = [
    "the", "this", "that", "these", "those",
    "is", "are", "was", "were",
    "have", "has", "had",
    "will", "would", "could", "should",
    "they", "we", "you", "he", "she",
    "not", "for", "with", "from", "about",
    "there", "where", "when", "how",
]
_EN_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _EN_HIGH_SIGNAL_WORDS) + r")\b",
    re.IGNORECASE,
)

# Minimum consecutive English words to flag a language switch
_ENGLISH_WORD_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LanguageReport:
    """Result of language validation for a single response text."""

    text: str
    has_emojis: bool = False
    emoji_count: int = 0
    has_brazilian_portuguese: bool = False
    brazilian_matches: List[str] = field(default_factory=list)
    has_language_switch: bool = False
    english_word_count: int = 0
    is_clean: bool = True  # True only if all checks pass

    def to_dict(self) -> dict:
        return {
            "is_clean": self.is_clean,
            "has_emojis": self.has_emojis,
            "emoji_count": self.emoji_count,
            "has_brazilian_portuguese": self.has_brazilian_portuguese,
            "brazilian_matches": self.brazilian_matches,
            "has_language_switch": self.has_language_switch,
            "english_word_count": self.english_word_count,
        }

    def issues(self) -> List[str]:
        """Return a human-readable list of issues found."""
        found: List[str] = []
        if self.has_emojis:
            found.append(f"Contains {self.emoji_count} emoji(s)")
        if self.has_brazilian_portuguese:
            found.append(
                f"Brazilian Portuguese detected: {', '.join(self.brazilian_matches[:5])}"
            )
        if self.has_language_switch:
            found.append(
                f"Possible language switch to English "
                f"({self.english_word_count} English function words)"
            )
        return found


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def detect_emojis(text: str) -> tuple[bool, int]:
    """Return (has_emojis, emoji_count) where count is individual emoji characters."""
    matches = _EMOJI_PATTERN.findall(text)
    # Each match may be a run of consecutive emojis — sum their lengths for individual count
    count = sum(len(m) for m in matches)
    return bool(matches), count


def detect_brazilian_portuguese(text: str) -> tuple[bool, List[str]]:
    """Return (has_br, [matched_terms])."""
    matches = [m.group(0) for m in _BR_PATTERN.finditer(text)]
    return bool(matches), matches


def detect_language_switch(text: str, threshold: int = _ENGLISH_WORD_THRESHOLD) -> tuple[bool, int]:
    """Detect if *text* contains a suspicious block of English words.

    Counts high-signal English function words.  Returns (switched, count).
    A response with ``threshold`` or more such words in total is flagged.
    """
    matches = _EN_WORD_PATTERN.findall(text)
    count = len(matches)
    return count >= threshold, count


def validate_response_language(
    text: str,
    english_threshold: int = _ENGLISH_WORD_THRESHOLD,
) -> LanguageReport:
    """Run all language checks on *text* and return a :class:`LanguageReport`.

    Args:
        text: The model response string to validate.
        english_threshold: Minimum English function-word count to flag a switch.

    Returns:
        A :class:`LanguageReport` with all results populated.
    """
    has_emojis, emoji_count = detect_emojis(text)
    has_br, br_matches = detect_brazilian_portuguese(text)
    has_switch, en_count = detect_language_switch(text, threshold=english_threshold)

    is_clean = not (has_emojis or has_br or has_switch)

    return LanguageReport(
        text=text,
        has_emojis=has_emojis,
        emoji_count=emoji_count,
        has_brazilian_portuguese=has_br,
        brazilian_matches=br_matches,
        has_language_switch=has_switch,
        english_word_count=en_count,
        is_clean=is_clean,
    )


def language_consistency_score(text: str) -> float:
    """Return a 0–5 float score for language consistency.

    5.0 = pure European Portuguese, no emojis, no BR, no language switch
    3.0 = minor issue (1–2 BR words OR 1–2 emojis)
    1.0 = moderate issue (>2 BR words AND emojis)
    0.0 = heavy English switch (>= 8 English function words)
    """
    report = validate_response_language(text)

    if report.is_clean:
        return 5.0

    penalty = 0.0

    # Emojis
    if report.has_emojis:
        penalty += 1.0 if report.emoji_count <= 2 else 2.0

    # Brazilian Portuguese
    if report.has_brazilian_portuguese:
        penalty += 1.0 if len(report.brazilian_matches) <= 2 else 2.0

    # Language switch (English)
    if report.has_language_switch:
        if report.english_word_count >= 8:
            return 0.0
        penalty += 2.0

    return max(0.0, 5.0 - penalty)
