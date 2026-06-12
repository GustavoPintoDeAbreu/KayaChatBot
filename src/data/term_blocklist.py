"""Term blocklist for the Kaya knowledge base.

A small, config-driven redaction layer that strips unwanted terms from the data
that feeds the model. Its job is to stop a term from leaking out of the context
where it actually belongs (e.g. cinema chat) into member bios and interests —
the canonical example is "Dolby Atmos" being over-attributed to members.

The blocklist is read from ``data.blocked_terms`` in config.yaml and applied in
two places:
  * src/data/build_vector_db.py — drops matching conversation messages and KB
    facts before they are embedded, so they never surface in retrieval.
  * src/data/generate_knowledge_base.py — filters interests/topics and bio
    sentences as profiles are merged, so blocked terms never re-enter on
    regeneration.

Pure functions only (regex + string ops) so this stays import-light and unit
testable without the model or data stack.
"""

import re
from typing import Iterable, List, Pattern


def compile_blocklist(terms: Iterable[str]) -> List[Pattern]:
    """Compile blocked terms into case-insensitive whole-word/phrase patterns.

    Whole-word boundaries keep short terms from firing inside unrelated words.
    """
    patterns: List[Pattern] = []
    for term in terms or []:
        term = (term or "").strip()
        if term:
            patterns.append(re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE))
    return patterns


def is_blocked(text: str, patterns: List[Pattern]) -> bool:
    """Return True if any blocked term appears in ``text``."""
    if not text or not patterns:
        return False
    return any(pat.search(text) for pat in patterns)


def filter_list(items: Iterable[str], patterns: List[Pattern]) -> List[str]:
    """Drop list entries (key_facts, interests, topics) mentioning a blocked term."""
    if not items:
        return []
    if not patterns:
        return list(items)
    return [item for item in items if not is_blocked(item, patterns)]


def redact_sentences(text: str, patterns: List[Pattern]) -> str:
    """Remove whole sentences containing a blocked term from a paragraph.

    Dropping the sentence (rather than just the term) avoids leaving dangling,
    ungrammatical fragments like "interested in music tech, particularly  and …".
    Returns the cleaned paragraph (may be empty if every sentence was blocked).
    """
    if not text or not patterns:
        return text or ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not is_blocked(s, patterns)]
    return " ".join(kept).strip()


def filter_messages(messages: List[dict], patterns: List[Pattern]) -> List[dict]:
    """Drop chat messages whose ``text`` mentions a blocked term.

    Used at vector-DB build time so blocked content is never chunked/embedded.
    """
    if not patterns:
        return messages
    return [msg for msg in messages if not is_blocked(msg.get("text", ""), patterns)]
