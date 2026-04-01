"""
Unit tests for src/chat/context_injection.py

Covers:
  - Recent summary injection when members are mentioned
  - Token-budget truncation with correct priority order
  - Toggle: inject_recent_summaries=False disables injection
  - Members without recent_summary field cause no errors
  - Token estimation helper
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.chat.context_injection import (
    build_recent_summaries,
    estimate_tokens,
    inject_recent_summaries,
    truncate_to_budget,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def members_data():
    """Minimal members_data dict with recent_summary fields for some members."""
    return {
        "group_name": "Kaya",
        "members": [
            {
                "name": "Peter",
                "aliases": ["peter", "pe"],
                "recent_summary": "Peter has been busy with work on audio projects recently.",
            },
            {
                "name": "Gil",
                "aliases": ["gil"],
                "recent_summary": "Gil went on a trip to Porto last week.",
            },
            {
                "name": "David",
                "aliases": ["david"],
                # No recent_summary — intentional
            },
        ],
    }


# ---------------------------------------------------------------------------
# test_estimate_tokens_*
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_estimate_tokens_nonempty_string(self):
        # 40 characters → roughly 10 tokens at 4 chars/token
        result = estimate_tokens("a" * 40)
        assert result == 10

    def test_estimate_tokens_returns_at_least_one(self):
        assert estimate_tokens("") >= 1
        assert estimate_tokens("a") >= 1

    def test_estimate_tokens_longer_is_bigger(self):
        assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


# ---------------------------------------------------------------------------
# test_build_recent_summaries_*
# ---------------------------------------------------------------------------

class TestBuildRecentSummaries:
    def test_build_recent_summaries_mentioned_member_included(self, members_data):
        result = build_recent_summaries(members_data, ["peter"])
        assert "Peter" in result
        assert "audio projects" in result

    def test_build_recent_summaries_multiple_members(self, members_data):
        result = build_recent_summaries(members_data, ["peter", "gil"])
        assert "Peter" in result
        assert "Gil" in result
        assert "Porto" in result

    def test_build_recent_summaries_unmentioned_member_excluded(self, members_data):
        result = build_recent_summaries(members_data, ["peter"])
        assert "Gil" not in result

    def test_build_recent_summaries_member_without_summary_no_error(self, members_data):
        # David has no recent_summary — should not raise, and should return empty for David
        result = build_recent_summaries(members_data, ["david"])
        assert result == ""  # no summary available, returns empty string

    def test_build_recent_summaries_empty_mentioned_returns_empty(self, members_data):
        result = build_recent_summaries(members_data, [])
        assert result == ""

    def test_build_recent_summaries_unknown_member_returns_empty(self, members_data):
        result = build_recent_summaries(members_data, ["nonexistent"])
        assert result == ""

    def test_build_recent_summaries_alias_matching(self, members_data):
        # "pe" is an alias for Peter
        result = build_recent_summaries(members_data, ["pe"])
        assert "Peter" in result

    def test_build_recent_summaries_contains_header(self, members_data):
        result = build_recent_summaries(members_data, ["peter"])
        assert "Resumos recentes" in result


# ---------------------------------------------------------------------------
# test_inject_recent_summaries_*
# ---------------------------------------------------------------------------

class TestInjectRecentSummaries:
    def test_inject_recent_summaries_prepends_to_context(self, members_data):
        existing_ctx = "=== Conversas relevantes ==="
        result = inject_recent_summaries(existing_ctx, members_data, ["peter"])
        assert result.startswith("=== Resumos recentes ===")
        assert existing_ctx in result

    def test_inject_recent_summaries_disabled_returns_context_unchanged(self, members_data):
        ctx = "some context"
        result = inject_recent_summaries(ctx, members_data, ["peter"], enabled=False)
        assert result == ctx

    def test_inject_recent_summaries_no_summary_available_returns_context_unchanged(self, members_data):
        ctx = "some context"
        result = inject_recent_summaries(ctx, members_data, ["david"])
        assert result == ctx  # David has no summary

    def test_inject_recent_summaries_empty_context_returns_only_summaries(self, members_data):
        result = inject_recent_summaries("", members_data, ["gil"])
        assert "Resumos recentes" in result
        assert "Porto" in result

    def test_inject_recent_summaries_no_mentioned_returns_context_unchanged(self, members_data):
        ctx = "existing context"
        result = inject_recent_summaries(ctx, members_data, [])
        assert result == ctx


# ---------------------------------------------------------------------------
# test_truncate_to_budget_*
# ---------------------------------------------------------------------------

class TestTruncateToBudget:
    def test_truncate_to_budget_no_truncation_needed(self):
        convs = ["short"]
        facts = ["fact"]
        summaries = ["summary"]
        rc, rf, rs = truncate_to_budget(convs, facts, summaries, max_tokens=1000)
        assert rc == convs
        assert rf == facts
        assert rs == summaries

    def test_truncate_to_budget_removes_conv_chunks_first(self):
        # Make convs very large and budget very small
        long_conv = "x" * 400   # ~100 tokens
        convs = [long_conv, long_conv, long_conv]
        facts = ["short fact"]
        summaries = ["short summary"]
        # Budget only allows facts + summaries (a few tokens each)
        rc, rf, rs = truncate_to_budget(convs, facts, summaries, max_tokens=10)
        # Conversation chunks should be reduced or empty
        assert len(rc) < len(convs)

    def test_truncate_to_budget_keeps_summaries_over_facts(self):
        convs: list = []
        facts = ["fact " * 100]  # ~25 tokens per fact, 1 fact
        summaries = ["summary " * 100]  # ~25 tokens
        # Budget is 0 — everything should be stripped eventually
        # but summaries have higher priority (removed last)
        # With very tiny budget, facts removed before summaries
        rc, rf, rs = truncate_to_budget(convs, facts, summaries, max_tokens=1)
        # After facts removed, summaries may also be removed
        # Key invariant: summaries are protected longer than facts
        # We verify that facts are emptied before summaries
        # (Since both lists start equal, the one that gets truncated first = facts)
        # Test: with budget large enough for summaries but not facts
        facts2 = ["f" * 200]   # ~50 tokens
        summaries2 = ["s" * 100]  # ~25 tokens
        # Budget = 30 tokens: summaries fit (25), facts don't (50)
        _, rf2, rs2 = truncate_to_budget([], facts2, summaries2, max_tokens=30)
        assert rf2 == []        # facts removed
        assert rs2 == summaries2  # summaries preserved

    def test_truncate_to_budget_empty_inputs(self):
        rc, rf, rs = truncate_to_budget([], [], [], max_tokens=100)
        assert rc == []
        assert rf == []
        assert rs == []

    def test_truncate_to_budget_zero_budget_removes_all(self):
        convs = ["x" * 400]
        facts = ["y" * 400]
        summaries = ["z" * 400]
        rc, rf, rs = truncate_to_budget(convs, facts, summaries, max_tokens=0)
        assert rc == []
        assert rf == []
        assert rs == []

    def test_truncate_to_budget_returns_lists(self):
        rc, rf, rs = truncate_to_budget(["a"], ["b"], ["c"], max_tokens=50)
        assert isinstance(rc, list)
        assert isinstance(rf, list)
        assert isinstance(rs, list)
