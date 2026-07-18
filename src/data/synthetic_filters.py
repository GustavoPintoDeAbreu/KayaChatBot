"""Pure filters for locally-generated synthetic answers.

Quality gate for the on-prem teacher's output before it becomes a training
target. No model/network deps so it is fully unit-testable. Shared emoji regex is
the single source of truth (readers.py imports it) so the train-time scrub and
the generation-time scrub can't drift.
"""

import re

# Emoji ranges (incl. the trailing-😊 habit), variation selectors and ZWJ.
# Deliberately excludes the general-punctuation block so em-dashes, curly quotes
# and ellipses used in Portuguese survive.
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U0000FE00-\U0000FE0F\U00002B00-\U00002BFF\U0000200D]+"
)

# Deflections/refusals the group disliked — targets containing these are dropped
# so the fine-tune never re-learns them.
_REFUSAL_PATTERNS = [
    r"sou apenas um",
    r"como assistente",
    r"enquanto assistente",
    r"sou um bot",
    r"não tenho opini",
    r"não tenho prefer",
    r"não tenho informaç",
    r"não é possível determinar",
    r"não participo",
    r"estou aqui para ajudar com quest",
    r"as an ai",
    r"i'?m just an? (ai|assistant|bot)",
    r"i (don'?t|do not) have (personal )?(opinions|preferences)",
    r"i can'?t determine",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove a leading <think>…</think> reasoning block (Qwen3/Qwen3.5 etc.).

    Reasoning-mode teachers prepend their chain-of-thought; without this it would
    leak into the training target. Also drops a dangling unclosed <think> opener.
    """
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text)
    # Unclosed think block (generation cut off mid-reasoning) — drop from the tag.
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


def strip_emojis(text: str) -> str:
    """Remove emojis and tidy the whitespace/punctuation they leave behind."""
    cleaned = EMOJI_RE.sub("", text or "")
    cleaned = re.sub(r"\s+([.!?,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned


def is_refusal(text: str) -> bool:
    """True if the answer deflects like the old persona ('sou apenas um bot…')."""
    return bool(_REFUSAL_RE.search(text or ""))


def is_too_short(text: str, min_words: int = 6) -> bool:
    """True if the answer is too short to be a real synthesized response."""
    return len((text or "").split()) < min_words


def is_echo(answer: str, context: str, threshold: float = 0.9) -> bool:
    """True if the answer is essentially a verbatim slice of the context.

    Catches the "fancy Ctrl+F" failure mode where the model just quotes a
    retrieved message instead of synthesizing. Compared on a normalized,
    punctuation-stripped basis so quote marks don't hide an echo.
    """
    if not answer or not context:
        return False

    def _norm(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s.lower())

    ans_norm = _norm(answer)
    ctx_norm = _norm(context)
    if not ans_norm:
        return False
    # Direct containment of the bulk of the answer inside the context.
    if len(ans_norm) >= 12 and ans_norm in ctx_norm:
        return True
    # Token-overlap heuristic: most answer tokens appearing contiguously-ish.
    ans_tokens = ans_norm.split()
    if not ans_tokens:
        return False
    ctx_tokens = set(ctx_norm.split())
    overlap = sum(1 for t in ans_tokens if t in ctx_tokens) / len(ans_tokens)
    return overlap >= threshold


def clean_and_accept(answer: str, context: str, min_words: int = 6) -> str:
    """Strip reasoning + emojis then accept/reject. Returns cleaned answer or ""."""
    cleaned = strip_emojis(strip_thinking(answer))
    if not cleaned or is_refusal(cleaned) or is_too_short(cleaned, min_words):
        return ""
    if is_echo(cleaned, context):
        return ""
    return cleaned
