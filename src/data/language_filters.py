"""
language_filters.py

Shared utilities for enforcing European Portuguese in training data.
Provides Brazilian → European Portuguese term substitution and emoji removal.
Applied as a post-processing step during synthetic data generation and dataset merging.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Brazilian → European Portuguese substitution map
# Covers the most common BR expressions found in chat data.
# ---------------------------------------------------------------------------

BRAZILIAN_TO_EUROPEAN: dict[str, str] = {
    # Pronouns / address
    "você": "tu",
    "voce": "tu",
    "vocês": "vocês",   # same form, keep
    # Slang / informal
    "rolê": "saída",
    "role": "cena",      # context-dependent; conservative replacement
    "cara": "gajo",
    "mano": "gajo",
    "galera": "malta",
    "top": "óptimo",
    "maneiro": "fixe",
    "bacana": "fixe",
    "massa": "fixe",
    "irado": "fixe",
    "show": "fixe",
    "legal": "fixe",
    "tudo bem": "tudo bem",  # keep (same in PT-EU)
    "valeu": "obrigado",
    "tchau": "adeus",
    "xau": "adeus",
    # Transport
    "ônibus": "autocarro",
    "onibus": "autocarro",
    "trem": "comboio",
    "metrô": "metro",
    "metro": "metro",      # keep (same)
    # Technology / everyday
    "celular": "telemóvel",
    "celulares": "telemóveis",
    "computador": "computador",  # keep (same)
    "notebook": "portátil",
    # Food
    "suco": "sumo",
    "sorvete": "gelado",
    "biscoito": "bolacha",
    "bolacha": "bolacha",   # keep
    "sanduíche": "sandes",
    "sanduiche": "sandes",
    # Places / buildings
    "apartamento": "apartamento",  # keep
    "sala": "sala",                # keep
    "banheiro": "casa de banho",
    "privada": "sanita",
    # Verbs / expressions
    "curtir": "gostar",
    "curti": "gostei",
    "curtindo": "a gostar",
    "tá": "tá",       # keep (same colloquial)
    "tô": "estou",
    "tava": "estava",
    "tava na": "estava na",
    "vou nessa": "vou já",
    "que saudade": "que saudades",   # PT-EU uses plural
    # Social
    "fim de semana": "fim de semana",   # keep
    "fds": "fds",                        # keep abbreviation
    "véi": "irmão",
    "brother": "amigo",
    # Intensifiers
    "muito bom": "muito bom",   # keep
    "demais": "demais",          # keep
    "caramba": "porra",
    "nossa": "uau",
    "poxa": "puxa",
    # Greetings
    "oi": "olá",
    "e aí": "e então",
    "firmeza": "tudo bem",
    "blz": "fixe",
    "vlw": "obrigado",
    "tmj": "juntos",
    # Other common BR → PT-EU
    "agora": "agora",       # keep
    "achar": "achar",       # keep
    "bagunça": "confusão",
    "boca": "boca",         # keep
    "burro": "burro",       # keep
}

# Regex to detect (not replace) known BR marker words in a response
_BR_MARKERS: list[str] = [
    r"\brol[eê]\b",
    r"\bcara\b",
    r"\bmaneiro\b",
    r"\bmano\b",
    r"\bgalera\b",
    r"\bvoc[eê]\b",
    r"\bônibus\b",
    r"\bonibus\b",
    r"\bcelular\b(?!mente)",  # avoid catching "celularmente" (doesn't exist but safe)
    r"\btrem\b",
    r"\bvaleu\b",
    r"\btchau\b",
    r"\bsuco\b",
    r"\bsorvete\b",
    r"\bbanh[ei]ro\b",
    r"\bcurti[r]?\b",
    r"\bvlw\b",
    r"\bblz\b",
    r"\btmj\b",
    r"\bfirmeza\b",
    r"\bvou nessa\b",
    r"\bpuxa\b",
    r"\bnossa\b(?! senhora)",   # "nossa senhora" is used in PT-EU
]

_BR_PATTERN = re.compile("|".join(_BR_MARKERS), re.IGNORECASE)

# Emoji unicode ranges (covers all common emoji blocks)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"   # Emoticons
    "\U0001f300-\U0001f5ff"   # Symbols & Pictographs
    "\U0001f680-\U0001f6ff"   # Transport & Map
    "\U0001f1e0-\U0001f1ff"   # Flags
    "\U00002702-\U000027b0"   # Dingbats
    "\U000024c2-\U0001f251"   # Enclosed characters
    "\U0001f900-\U0001f9ff"   # Supplemental Symbols & Pictographs
    "\U0001fa00-\U0001fa6f"   # Chess, Suits
    "\U0001fa70-\U0001faff"   # Food, Drink, etc.
    "\U00002500-\U00002bef"   # Misc symbols
    "\U00010000-\U0010ffff"   # Supplementary Multilingual Plane (broad)
    "]+",
    flags=re.UNICODE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def strip_emojis(text: str) -> str:
    """Remove all emoji characters from *text*."""
    return _EMOJI_PATTERN.sub("", text).strip()


def apply_brazilian_replacements(text: str) -> str:
    """Substitute common Brazilian Portuguese terms with European equivalents.

    Uses word-boundary matching (case-insensitive) to avoid partial replacements.
    """
    for br_term, pt_term in BRAZILIAN_TO_EUROPEAN.items():
        if br_term == pt_term:
            continue
        # Build a word-boundary pattern; escape special regex chars in term
        pattern = r"\b" + re.escape(br_term) + r"\b"
        # Preserve capitalisation: if the match starts uppercase, capitalise replacement
        def _replace(m: re.Match, replacement: str = pt_term) -> str:
            original = m.group(0)
            if original[0].isupper():
                return replacement[0].upper() + replacement[1:]
            return replacement

        text = re.sub(pattern, _replace, text, flags=re.IGNORECASE)
    return text


def clean_training_text(text: str) -> str:
    """Apply all language filters to a training text string.

    1. Remove emojis
    2. Apply Brazilian → European Portuguese substitutions

    Returns the cleaned text.
    """
    text = strip_emojis(text)
    text = apply_brazilian_replacements(text)
    return text


def contains_brazilian_portuguese(text: str) -> bool:
    """Return True if *text* contains known Brazilian Portuguese marker words."""
    return bool(_BR_PATTERN.search(text))


def contains_emojis(text: str) -> bool:
    """Return True if *text* contains emoji characters."""
    return bool(_EMOJI_PATTERN.search(text))
