"""
Context injection utilities for RAG-augmented chat.

Provides helpers to inject recent member summaries into the prompt and to
truncate context blocks so the total stays within a token budget.
"""

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4  # rough approximation: ~4 chars per token


def estimate_tokens(text: str) -> int:
    """Estimate token count using a simple character-based heuristic."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Recent summary injection
# ---------------------------------------------------------------------------

def build_recent_summaries(
    members_data: Dict[str, Any],
    mentioned_members: List[str],
) -> str:
    """
    Build a recent-summaries block for the members listed in *mentioned_members*.

    Parameters
    ----------
    members_data:
        Dict loaded from ``group_members.json`` with a ``"members"`` list.
        Each member dict may contain a ``"recent_summary"`` key.
    mentioned_members:
        Lower-cased names / aliases of members whose summaries should be
        included.

    Returns
    -------
    Formatted multi-line string, or an empty string when no summaries are
    available.
    """
    members = members_data.get("members", [])
    parts: List[str] = []

    for member in members:
        # Match by name or any alias
        name_lower = member.get("name", "").lower()
        aliases_lower = [a.lower() for a in member.get("aliases", [])]
        identifiers = {name_lower} | set(aliases_lower)

        if not identifiers.intersection(set(m.lower() for m in mentioned_members)):
            continue

        summary = member.get("recent_summary", "")
        if not summary:
            continue  # skip members without a recent_summary field

        display_name = member.get("name", name_lower.title())
        parts.append(f"[Recent summary — {display_name}]\n{summary}")

    if not parts:
        return ""

    return "=== Resumos recentes ===\n" + "\n\n".join(parts) + "\n=== Fim dos resumos ==="


def inject_recent_summaries(
    context: str,
    members_data: Dict[str, Any],
    mentioned_members: List[str],
    enabled: bool = True,
) -> str:
    """
    Prepend recent summaries to *context* when ``enabled`` is ``True``.

    Parameters
    ----------
    context:
        Existing RAG context string.
    members_data:
        Dict loaded from ``group_members.json``.
    mentioned_members:
        Lower-cased names of members mentioned in the user query.
    enabled:
        When ``False`` the context is returned unchanged.

    Returns
    -------
    Augmented context string.
    """
    if not enabled:
        return context

    summaries = build_recent_summaries(members_data, mentioned_members)
    if not summaries:
        return context

    if context:
        return f"{summaries}\n\n{context}"
    return summaries


# ---------------------------------------------------------------------------
# Token-budget truncation
# ---------------------------------------------------------------------------

def truncate_to_budget(
    conv_chunks: List[str],
    kb_facts: List[str],
    recent_summaries: List[str],
    max_tokens: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Truncate context lists so total estimated tokens stay within *max_tokens*.

    Truncation priority (highest = removed first):
      1. Conversation chunks (least important when budget is tight)
      2. Knowledge-base facts
      3. Recent summaries (highest priority — preserved as long as possible)

    Parameters
    ----------
    conv_chunks:
        List of conversation text strings.
    kb_facts:
        List of knowledge-base fact strings.
    recent_summaries:
        List of recent-summary strings.
    max_tokens:
        Maximum allowed estimated tokens across all three lists combined.

    Returns
    -------
    (truncated_conv_chunks, truncated_kb_facts, truncated_recent_summaries)
    """

    def _total_tokens(chunks: List[str]) -> int:
        return sum(estimate_tokens(c) for c in chunks)

    # Work with mutable copies; prefer keeping summaries > facts > convs
    convs = list(conv_chunks)
    facts = list(kb_facts)
    summaries = list(recent_summaries)

    while _total_tokens(convs) + _total_tokens(facts) + _total_tokens(summaries) > max_tokens:
        if convs:
            convs.pop()
        elif facts:
            facts.pop()
        elif summaries:
            summaries.pop()
        else:
            break  # nothing left to remove

    return convs, facts, summaries
