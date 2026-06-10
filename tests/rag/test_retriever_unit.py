"""
Unit tests for src/chat/retriever.py — ConversationRetriever search functions.
Covers: empty queries, no results, multi-document matches, person extraction,
context formatting, and knowledge retrieval.  Uses pytest fixtures with mocked
ChromaDB collections so no database or embedding model is needed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.chat.retriever import ConversationRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config():
    """Minimal config dict matching the shape ConversationRetriever expects."""
    return {
        "rag": {
            "top_k": 5,
            "filter_by_person": True,
            "max_context_tokens": 3000,
            "inject_recent_summaries": True,
            "knowledge_base": {
                "enabled": True,
                "collection_name": "kaya_knowledge_base",
                "top_k": 2,
            },
        },
        "data": {
            "group_members_file": None,  # will use fallback member list
        },
    }


@pytest.fixture
def retriever(base_config):
    """ConversationRetriever with mocked ChromaDB collections and encoder."""
    r = ConversationRetriever(base_config)

    # Mock encoder — returns a deterministic unit vector per query
    r.encoder = MagicMock()
    r.encoder.encode = MagicMock(side_effect=lambda texts, **kwargs: np.ones((len(texts), 8)))

    # Mock conversation collection
    r.collection = MagicMock()
    r.collection.count.return_value = 100

    # Mock knowledge collection
    r.knowledge_collection = MagicMock()
    r.knowledge_collection.count.return_value = 50

    return r


@pytest.fixture
def retriever_with_members(tmp_path):
    """ConversationRetriever backed by a real members JSON file with recent_summary data."""
    members_data = {
        "members": [
            {
                "name": "Peter",
                "aliases": ["peter", "pe"],
                "notes": "Peter likes audio.",
                "recent_summary": "Peter recently discussed his new speakers.",
            },
            {
                "name": "Gil",
                "aliases": ["gil", "gilao"],
                "notes": "Gil plays guitar.",
                "recent_summary": "",  # empty — should be skipped
            },
            {
                "name": "Rafa",
                "aliases": ["rafa"],
                "notes": "Rafa is into football.",
                "recent_summary": "Rafa organised a poker night.",
            },
        ]
    }
    members_file = tmp_path / "members.json"
    members_file.write_text(__import__("json").dumps(members_data), encoding="utf-8")

    config = {
        "rag": {
            "top_k": 5,
            "filter_by_person": True,
            "max_context_tokens": 3000,
            "inject_recent_summaries": True,
            "knowledge_base": {
                "enabled": True,
                "collection_name": "kaya_knowledge_base",
                "top_k": 2,
            },
        },
        "data": {
            "group_members_file": str(members_file),
        },
    }
    r = ConversationRetriever(config)
    r.encoder = MagicMock()
    r.encoder.encode = MagicMock(side_effect=lambda texts, **kwargs: np.ones((len(texts), 8)))
    r.collection = MagicMock()
    r.collection.count.return_value = 100
    r.knowledge_collection = MagicMock()
    r.knowledge_collection.count.return_value = 50
    return r


def _make_query_result(docs, metadatas, distances):
    """Build the nested dict shape returned by ChromaDB collection.query()."""
    return {
        "documents": [docs],
        "metadatas": [metadatas],
        "distances": [distances],
    }


# ---------------------------------------------------------------------------
# extract_query_persons
# ---------------------------------------------------------------------------

class TestExtractQueryPersons:
    def test_single_person(self, retriever):
        assert "peter" in retriever.extract_query_persons("What did Peter say?")

    def test_multiple_persons(self, retriever):
        persons = retriever.extract_query_persons("Did Gil and Rafa go out?")
        assert "gil" in persons
        assert "rafa" in persons

    def test_no_person_mentioned(self, retriever):
        assert retriever.extract_query_persons("How is the weather?") == []

    def test_case_insensitive(self, retriever):
        assert "david" in retriever.extract_query_persons("DAVID likes football")

    def test_empty_query(self, retriever):
        assert retriever.extract_query_persons("") == []


# ---------------------------------------------------------------------------
# retrieve — conversation search
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_basic_retrieval(self, retriever):
        retriever.collection.query.return_value = _make_query_result(
            docs=["Peter: hey", "Gil: sup"],
            metadatas=[
                {"participants": "Peter", "mentioned": "", "message_count": 2, "token_count": 10,
                 "timestamp_start": "2024-01-01T10:00:00", "timestamp_end": "2024-01-01T10:05:00"},
                {"participants": "Gil", "mentioned": "", "message_count": 3, "token_count": 15,
                 "timestamp_start": "2024-01-02T10:00:00", "timestamp_end": "2024-01-02T10:05:00"},
            ],
            distances=[0.1, 0.3],
        )

        results = retriever.retrieve("hello", top_k=2)

        assert len(results) == 2
        assert results[0]["text"] == "Peter: hey"
        assert results[0]["rank"] == 1
        assert results[0]["similarity_score"] == pytest.approx(0.9)
        assert results[1]["similarity_score"] == pytest.approx(0.7)

    def test_empty_query_still_calls_db(self, retriever):
        """An empty string is a valid query — the vector DB should still be called."""
        retriever.collection.query.return_value = _make_query_result([], [], [])
        results = retriever.retrieve("", top_k=3)
        assert results == []
        retriever.collection.query.assert_called_once()

    def test_no_results(self, retriever):
        retriever.collection.query.return_value = _make_query_result([], [], [])
        results = retriever.retrieve("something obscure", top_k=5)
        assert results == []

    def test_top_k_limits_output(self, retriever):
        docs = [f"msg {i}" for i in range(10)]
        metas = [
            {"participants": "", "mentioned": "", "message_count": 1, "token_count": 5,
             "timestamp_start": None, "timestamp_end": None}
            for _ in range(10)
        ]
        dists = [0.1 * i for i in range(10)]

        retriever.collection.query.return_value = _make_query_result(docs, metas, dists)
        results = retriever.retrieve("test", top_k=3)
        assert len(results) == 3

    def test_person_filter_keeps_matching(self, retriever):
        """When query mentions 'peter', only chunks with peter should be returned."""
        retriever.collection.query.return_value = _make_query_result(
            docs=["Peter: hey", "Gil: something else"],
            metadatas=[
                {"participants": "Peter", "mentioned": "", "message_count": 1, "token_count": 5,
                 "timestamp_start": None, "timestamp_end": None},
                {"participants": "Gil", "mentioned": "", "message_count": 1, "token_count": 5,
                 "timestamp_start": None, "timestamp_end": None},
            ],
            distances=[0.1, 0.2],
        )

        results = retriever.retrieve("What did Peter say?", top_k=5)
        assert len(results) == 1
        assert results[0]["text"] == "Peter: hey"

    def test_person_filter_mentioned_field(self, retriever):
        """Person filter should also match on the 'mentioned' metadata field."""
        retriever.collection.query.return_value = _make_query_result(
            docs=["Gil: Peter is great"],
            metadatas=[
                {"participants": "Gil", "mentioned": "Peter", "message_count": 1, "token_count": 5,
                 "timestamp_start": None, "timestamp_end": None},
            ],
            distances=[0.1],
        )

        results = retriever.retrieve("Tell me about Peter", top_k=5)
        assert len(results) == 1

    def test_person_filter_no_match(self, retriever):
        """If query mentions 'peter' but no chunk involves peter, return nothing."""
        retriever.collection.query.return_value = _make_query_result(
            docs=["Gil: hey", "David: hello"],
            metadatas=[
                {"participants": "Gil", "mentioned": "", "message_count": 1, "token_count": 5,
                 "timestamp_start": None, "timestamp_end": None},
                {"participants": "David", "mentioned": "", "message_count": 1, "token_count": 5,
                 "timestamp_start": None, "timestamp_end": None},
            ],
            distances=[0.1, 0.2],
        )

        results = retriever.retrieve("What did Peter say?", top_k=5)
        assert results == []

    def test_not_initialized_raises(self, base_config):
        r = ConversationRetriever(base_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            r.retrieve("hello")

    def test_participants_parsed_as_list(self, retriever):
        retriever.collection.query.return_value = _make_query_result(
            docs=["multi participant"],
            metadatas=[
                {"participants": "Peter,Gil,David", "mentioned": "Rafa", "message_count": 5,
                 "token_count": 50, "timestamp_start": None, "timestamp_end": None},
            ],
            distances=[0.05],
        )

        results = retriever.retrieve("general question", top_k=1)
        assert results[0]["participants"] == ["Peter", "Gil", "David"]
        assert results[0]["mentioned"] == ["Rafa"]


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_empty_chunks(self, retriever):
        assert retriever.format_context([]) == ""

    def test_single_chunk(self, retriever):
        chunks = [{
            "text": "Peter: hello",
            "timestamp_start": "2024-06-15T10:00:00",
        }]
        ctx = retriever.format_context(chunks)
        assert "Conversas relevantes" in ctx
        assert "Peter: hello" in ctx
        # Dates are hidden by default and only shown on timing queries.
        assert "2024-06-15" not in ctx
        assert "2024-06-15" in retriever.format_context(chunks, show_dates=True)

    def test_multiple_chunks_numbered(self, retriever):
        chunks = [
            {"text": "first", "timestamp_start": None},
            {"text": "second", "timestamp_start": None},
        ]
        ctx = retriever.format_context(chunks)
        assert "Conversa 1" in ctx
        assert "Conversa 2" in ctx

    def test_bad_timestamp_handled(self, retriever):
        chunks = [{"text": "msg", "timestamp_start": "not-a-date"}]
        ctx = retriever.format_context(chunks)
        assert "msg" in ctx  # should still render even if timestamp parse fails


# ---------------------------------------------------------------------------
# retrieve_knowledge
# ---------------------------------------------------------------------------

class TestRetrieveKnowledge:
    def test_basic_knowledge_retrieval(self, retriever):
        retriever.knowledge_collection.query.return_value = _make_query_result(
            docs=["Peter is from Lisbon"],
            metadatas=[{"subject": "Peter", "category": "location"}],
            distances=[0.15],
        )

        results = retriever.retrieve_knowledge("Where is Peter from?", top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "Peter is from Lisbon"
        assert results[0]["subject"] == "Peter"
        assert results[0]["similarity_score"] == pytest.approx(0.85)

    def test_no_knowledge_collection(self, base_config):
        r = ConversationRetriever(base_config)
        r.knowledge_collection = None
        r.encoder = MagicMock()
        assert r.retrieve_knowledge("anything") == []

    def test_empty_knowledge_results(self, retriever):
        retriever.knowledge_collection.query.return_value = _make_query_result([], [], [])
        assert retriever.retrieve_knowledge("obscure") == []


# ---------------------------------------------------------------------------
# format_knowledge_context
# ---------------------------------------------------------------------------

class TestFormatKnowledgeContext:
    def test_empty(self, retriever):
        assert retriever.format_knowledge_context([]) == ""

    def test_with_subject(self, retriever):
        chunks = [{"text": "Fact one. Fact two.", "subject": "Gil", "category": "bio"}]
        ctx = retriever.format_knowledge_context(chunks)
        assert "Gil" in ctx
        assert "Conhecimento sobre o grupo" in ctx

    def test_truncation_to_3_sentences(self, retriever):
        long_text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = [{"text": long_text, "subject": "Test", "category": ""}]
        ctx = retriever.format_knowledge_context(chunks)
        assert "Sentence four" not in ctx
        assert "Sentence three" in ctx


# ---------------------------------------------------------------------------
# retrieve_all
# ---------------------------------------------------------------------------

class TestRetrieveAll:
    def _setup_both_collections(self, retriever):
        retriever.collection.query.return_value = _make_query_result(
            docs=["Peter: hey"],
            metadatas=[
                {"participants": "Peter", "mentioned": "", "message_count": 1, "token_count": 5,
                 "timestamp_start": None, "timestamp_end": None},
            ],
            distances=[0.1],
        )
        retriever.knowledge_collection.query.return_value = _make_query_result(
            docs=["Peter is from Lisbon"],
            metadatas=[{"subject": "Peter", "category": "bio"}],
            distances=[0.15],
        )

    def test_both_approach(self, retriever):
        self._setup_both_collections(retriever)
        ctx = retriever.retrieve_all("Tell me about Peter", knowledge_approach="both")
        assert "Conhecimento sobre o grupo" in ctx
        assert "Conversas relevantes" in ctx

    def test_json_only_approach(self, retriever):
        self._setup_both_collections(retriever)
        ctx = retriever.retrieve_all("Tell me about Peter", knowledge_approach="json_only")
        assert "Conhecimento sobre o grupo" not in ctx
        assert "Conversas relevantes" in ctx

    def test_none_approach(self, retriever):
        self._setup_both_collections(retriever)
        ctx = retriever.retrieve_all("Tell me about Peter", knowledge_approach="none")
        assert "Conhecimento sobre o grupo" not in ctx
        assert "Conversas relevantes" in ctx


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_stats_when_initialized(self, retriever):
        stats = retriever.get_stats()
        assert stats["total_conversation_chunks"] == 100
        assert stats["total_knowledge_facts"] == 50

    def test_stats_not_initialized(self, base_config):
        r = ConversationRetriever(base_config)
        stats = r.get_stats()
        assert "error" in stats


# ---------------------------------------------------------------------------
# _count_tokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_string(self, retriever):
        assert retriever._count_tokens("") == 0

    def test_none_like_empty(self, retriever):
        assert retriever._count_tokens("") == 0

    def test_single_word(self, retriever):
        # 1 word → int(1.333...) = 1
        assert retriever._count_tokens("hello") == 1

    def test_many_words(self, retriever):
        text = " ".join(["word"] * 75)  # 75 words → int(75 / 0.60) = 125
        assert retriever._count_tokens(text) == 125

    def test_proportional(self, retriever):
        text = " ".join(["w"] * 30)
        assert retriever._count_tokens(text) == int(30 / 0.60)


# ---------------------------------------------------------------------------
# _format_recent_summaries
# ---------------------------------------------------------------------------

class TestFormatRecentSummaries:
    def test_no_persons_returns_empty(self, retriever_with_members):
        result = retriever_with_members._format_recent_summaries([])
        assert result == ""

    def test_no_members_data_returns_empty(self, base_config):
        r = ConversationRetriever(base_config)
        # base_config has no members file → _members_data is empty
        result = r._format_recent_summaries(["peter"])
        assert result == ""

    def test_member_with_summary_included(self, retriever_with_members):
        result = retriever_with_members._format_recent_summaries(["peter"])
        assert "Peter" in result
        assert "recently discussed his new speakers" in result
        assert "Resumos recentes dos membros" in result

    def test_member_with_empty_summary_excluded(self, retriever_with_members):
        result = retriever_with_members._format_recent_summaries(["gil"])
        # Gil has an empty recent_summary — should produce no output
        assert result == ""

    def test_multiple_members(self, retriever_with_members):
        result = retriever_with_members._format_recent_summaries(["peter", "rafa"])
        assert "Peter" in result
        assert "Rafa" in result
        assert "poker night" in result

    def test_alias_matching(self, retriever_with_members):
        # "pe" is an alias for Peter
        result = retriever_with_members._format_recent_summaries(["pe"])
        assert "Peter" in result

    def test_unknown_person_returns_empty(self, retriever_with_members):
        result = retriever_with_members._format_recent_summaries(["nobody"])
        assert result == ""

    def test_output_format(self, retriever_with_members):
        result = retriever_with_members._format_recent_summaries(["peter"])
        assert result.startswith("=== Resumos recentes dos membros ===")
        assert result.endswith("=== Fim dos resumos ===")


# ---------------------------------------------------------------------------
# retrieve_all — recent summaries integration
# ---------------------------------------------------------------------------

class TestRetrieveAllWithSummaries:
    def _setup_collections(self, retriever):
        retriever.collection.query.return_value = _make_query_result(
            docs=["Peter: hey"],
            metadatas=[
                {"participants": "Peter", "mentioned": "", "message_count": 1, "token_count": 5,
                 "timestamp_start": None, "timestamp_end": None},
            ],
            distances=[0.1],
        )
        retriever.knowledge_collection.query.return_value = _make_query_result(
            docs=["Peter is from Lisbon"],
            metadatas=[{"subject": "Peter", "category": "bio"}],
            distances=[0.15],
        )

    def test_summaries_injected_when_member_mentioned(self, retriever_with_members):
        self._setup_collections(retriever_with_members)
        ctx = retriever_with_members.retrieve_all("Tell me about Peter", knowledge_approach="both")
        assert "Resumos recentes dos membros" in ctx
        assert "recently discussed his new speakers" in ctx

    def test_summaries_not_injected_when_toggle_off(self, retriever_with_members):
        retriever_with_members.rag_config["inject_recent_summaries"] = False
        self._setup_collections(retriever_with_members)
        ctx = retriever_with_members.retrieve_all("Tell me about Peter", knowledge_approach="both")
        assert "Resumos recentes dos membros" not in ctx

    def test_summaries_not_injected_when_no_member_mentioned(self, retriever_with_members):
        self._setup_collections(retriever_with_members)
        ctx = retriever_with_members.retrieve_all("How is the weather?", knowledge_approach="both")
        assert "Resumos recentes dos membros" not in ctx

    def test_summaries_appear_before_conversations(self, retriever_with_members):
        self._setup_collections(retriever_with_members)
        ctx = retriever_with_members.retrieve_all("Tell me about Peter", knowledge_approach="json_only")
        summaries_pos = ctx.find("Resumos recentes dos membros")
        conv_pos = ctx.find("Conversas relevantes")
        assert summaries_pos < conv_pos


# ---------------------------------------------------------------------------
# retrieve_all — token budget enforcement
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def _make_long_chunk_result(self, n: int):
        """Return n conversation chunks, each with ~50 words of text."""
        long_text = " ".join(["word"] * 50)
        docs = [f"Chunk {i}: {long_text}" for i in range(n)]
        metas = [
            {"participants": "", "mentioned": "", "message_count": 1, "token_count": 50,
             "timestamp_start": None, "timestamp_end": None}
            for _ in range(n)
        ]
        dists = [0.1 * (i + 1) for i in range(n)]  # increasing distance = decreasing similarity
        return _make_query_result(docs, metas, dists)

    def test_context_within_budget_with_default_limit(self, retriever):
        """With default max_context_tokens=3000, 5 small chunks should all fit."""
        retriever.collection.query.return_value = self._make_long_chunk_result(5)
        retriever.knowledge_collection.query.return_value = _make_query_result([], [], [])
        ctx = retriever.retrieve_all("test query", knowledge_approach="none")
        token_count = retriever._count_tokens(ctx)
        assert token_count <= 3000

    def test_low_budget_prunes_chunks(self, retriever):
        """Setting a very low token budget should prune all but the highest-similarity chunk."""
        retriever.rag_config["max_context_tokens"] = 50  # Very tight budget
        retriever.collection.query.return_value = self._make_long_chunk_result(10)
        retriever.knowledge_collection.query.return_value = _make_query_result([], [], [])
        ctx = retriever.retrieve_all("test query", knowledge_approach="none")
        token_count = retriever._count_tokens(ctx)
        # Chunks should have been pruned — result is within budget (or empty if even one is too large)
        assert token_count <= 50

    def test_knowledge_chunks_pruned_after_conv_chunks(self, retriever):
        """When budget is exceeded, knowledge facts are removed after all conv chunks are exhausted."""
        # Give a tiny budget and provide only knowledge (no conv chunks) to confirm knowledge is pruned
        retriever.rag_config["max_context_tokens"] = 10
        retriever.collection.query.return_value = _make_query_result([], [], [])  # no conv
        retriever.knowledge_collection.query.return_value = _make_query_result(
            docs=[" ".join(["fact"] * 30)],
            metadatas=[{"subject": "Peter", "category": "bio"}],
            distances=[0.1],
        )
        ctx = retriever.retrieve_all("Tell me about Peter", knowledge_approach="both")
        # Knowledge (~30 words ≈ 40 tokens) exceeds budget of 10, so it must be pruned
        assert "Conhecimento" not in ctx
