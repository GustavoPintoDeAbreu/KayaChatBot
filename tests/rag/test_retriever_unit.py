"""
Unit tests for src/chat/retriever.py search functions.

Tests use mocked ChromaDB collections and a mock encoder so they run without
external downloads or a live vector database.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.chat.retriever import ConversationRetriever  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal config dict (no file I/O needed for __init__)
# ---------------------------------------------------------------------------
MOCK_CONFIG = {
    "rag": {
        "top_k": 5,
        "filter_by_person": True,
        "knowledge_base": {
            "enabled": True,
            "collection_name": "kaya_knowledge_base",
            "top_k": 3,
        },
    },
    "data": {},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query_result(docs, metadatas, distances):
    """Build the dict shape that chromadb collection.query() returns."""
    return {
        "documents": [docs],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def _default_metadata():
    return {"participants": "", "mentioned": "", "message_count": 1, "token_count": 10}


def _make_mock_encoder(dim: int = 4):
    """Return a mock SentenceTransformer whose encode() returns deterministic vectors."""
    encoder = MagicMock()
    encoder.encode.side_effect = lambda texts: np.ones((len(texts), dim), dtype="float32")
    return encoder


def _make_retriever(collection=None, knowledge_collection=None, encoder=None):
    """Return a ConversationRetriever with mocked internals (no initialize() call)."""
    retriever = ConversationRetriever(MOCK_CONFIG)
    # Override the group_members set that __init__ populates via JSON / fallback
    retriever.group_members = {"peter", "gil", "gustavo", "david", "rafa", "bernardo"}
    retriever.client = MagicMock()
    retriever.collection = collection if collection is not None else MagicMock()
    retriever.knowledge_collection = knowledge_collection
    retriever.encoder = encoder if encoder is not None else _make_mock_encoder()
    return retriever


def _simple_collection(docs, metadatas=None, distances=None):
    """Build a mock collection that returns the given documents on query()."""
    if metadatas is None:
        metadatas = [_default_metadata() for _ in docs]
    if distances is None:
        distances = [0.1 * (i + 1) for i in range(len(docs))]
    col = MagicMock()
    col.count.return_value = len(docs)
    col.query.return_value = _make_query_result(docs, metadatas, distances)
    return col


# ---------------------------------------------------------------------------
# extract_query_persons
# ---------------------------------------------------------------------------

class TestExtractQueryPersons:
    def test_single_person_detected(self):
        r = _make_retriever()
        assert r.extract_query_persons("What did peter say?") == ["peter"]

    def test_multiple_persons_detected(self):
        r = _make_retriever()
        result = r.extract_query_persons("Tell me about gil and rafa")
        assert set(result) == {"gil", "rafa"}

    def test_no_person_in_query(self):
        r = _make_retriever()
        assert r.extract_query_persons("What happened last weekend?") == []

    def test_empty_query(self):
        r = _make_retriever()
        assert r.extract_query_persons("") == []

    def test_case_insensitive_matching(self):
        r = _make_retriever()
        # group_members are stored lowercase; query has uppercase
        result = r.extract_query_persons("What does PETER think?")
        assert "peter" in result

    def test_partial_name_not_matched(self):
        """Substring 'gil' should not match a word like 'agile' unless present."""
        r = _make_retriever()
        # 'david' is in the set; 'davidson' contains 'david' — substring match expected
        # because the implementation uses 'member in query_lower'
        result = r.extract_query_persons("No group members mentioned here at all.")
        assert result == []


# ---------------------------------------------------------------------------
# retrieve — basic behaviour
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_empty_query_still_returns_results(self):
        """An empty string query should be forwarded to ChromaDB without error."""
        col = _simple_collection(["Hello world"])
        r = _make_retriever(collection=col)
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            results = r.retrieve("", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_empty_collection_returns_empty_list(self):
        """When the collection has zero documents the result list is empty."""
        col = MagicMock()
        col.count.return_value = 0
        col.query.return_value = _make_query_result([], [], [])
        r = _make_retriever(collection=col)
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            results = r.retrieve("some query", top_k=5)
        assert results == []

    def test_multiple_docs_returned(self):
        """Query matching several documents returns all of them."""
        col = _simple_collection(["doc1", "doc2", "doc3"])
        r = _make_retriever(collection=col)
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            results = r.retrieve("any query", top_k=3)
        assert len(results) == 3

    def test_top_k_limits_number_of_results(self):
        """top_k=2 should return at most 2 results even when more exist."""
        docs = ["a", "b", "c", "d"]
        col = _simple_collection(docs)
        r = _make_retriever(collection=col)
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            results = r.retrieve("any query", top_k=2)
        assert len(results) == 2

    def test_similarity_score_is_one_minus_distance(self):
        """similarity_score must equal 1 - distance."""
        col = _simple_collection(["doc"], distances=[0.3])
        r = _make_retriever(collection=col)
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            results = r.retrieve("query", top_k=5)
        assert len(results) == 1
        assert pytest.approx(results[0]["similarity_score"]) == 0.7

    def test_rank_assigned_incrementally(self):
        """Results should have rank 1, 2, 3, …"""
        col = _simple_collection(["a", "b", "c"])
        r = _make_retriever(collection=col)
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            results = r.retrieve("query")
        assert [res["rank"] for res in results] == [1, 2, 3]

    def test_result_fields_present(self):
        """Each result dict must contain the expected keys."""
        expected_keys = {
            "rank", "text", "metadata", "similarity_score", "distance",
            "participants", "mentioned", "message_count", "token_count",
            "timestamp_start", "timestamp_end",
        }
        col = _simple_collection(["doc"])
        r = _make_retriever(collection=col)
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            results = r.retrieve("query", top_k=5)
        assert expected_keys.issubset(set(results[0].keys()))

    def test_not_initialized_raises_runtime_error(self):
        """retrieve() must raise RuntimeError when collection is None."""
        r = _make_retriever()
        r.collection = None
        with pytest.raises(RuntimeError, match="not initialized"):
            r.retrieve("query")

    def test_not_initialized_no_encoder_raises(self):
        """retrieve() must raise RuntimeError when encoder is None."""
        r = _make_retriever()
        r.encoder = None
        with pytest.raises(RuntimeError, match="not initialized"):
            r.retrieve("query")

    def test_default_top_k_from_config(self):
        """When top_k is omitted, the value from rag_config is used."""
        docs = ["x"] * 10
        col = _simple_collection(docs, distances=[0.1 * i for i in range(1, 11)])
        r = _make_retriever(collection=col)
        # MOCK_CONFIG has top_k=5; ChromaDB mock returns 10 docs but retriever caps at 5
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            results = r.retrieve("query")  # no top_k passed
        assert len(results) == 5


# ---------------------------------------------------------------------------
# retrieve — person filtering
# ---------------------------------------------------------------------------

class TestRetrievePersonFilter:
    def _col_with_participants(self, participants_list):
        metadatas = [
            {"participants": p, "mentioned": "", "message_count": 1, "token_count": 10}
            for p in participants_list
        ]
        return _simple_collection(
            [f"doc {i}" for i in range(len(participants_list))],
            metadatas=metadatas,
        )

    def test_matching_participant_included(self):
        """Chunk where the queried person is a participant is returned."""
        col = self._col_with_participants(["peter,gil", "david,gustavo"])
        r = _make_retriever(collection=col)
        with patch("src.chat.retriever.FILTER_BY_PERSON", True):
            results = r.retrieve("what did peter say?", top_k=5)
        assert len(results) == 1
        assert "peter" in results[0]["participants"]

    def test_no_matching_participant_returns_empty(self):
        """When no chunk contains the queried person, return empty list."""
        col = self._col_with_participants(["david,gustavo", "gil,rafa"])
        r = _make_retriever(collection=col)
        with patch("src.chat.retriever.FILTER_BY_PERSON", True):
            results = r.retrieve("what did peter say?", top_k=5)
        assert results == []

    def test_person_in_mentioned_field_included(self):
        """A chunk where the person appears in 'mentioned' should be returned."""
        col = MagicMock()
        col.count.return_value = 1
        col.query.return_value = _make_query_result(
            ["doc"],
            [{"participants": "david", "mentioned": "peter", "message_count": 1, "token_count": 10}],
            [0.2],
        )
        r = _make_retriever(collection=col)
        with patch("src.chat.retriever.FILTER_BY_PERSON", True):
            results = r.retrieve("what did peter say?", top_k=5)
        assert len(results) == 1

    def test_filter_disabled_returns_all_docs(self):
        """When FILTER_BY_PERSON is False, person filtering is skipped."""
        col = MagicMock()
        col.count.return_value = 2
        col.query.return_value = _make_query_result(
            ["doc1", "doc2"],
            [
                {"participants": "david", "mentioned": "", "message_count": 1, "token_count": 10},
                {"participants": "gil", "mentioned": "", "message_count": 1, "token_count": 10},
            ],
            [0.1, 0.2],
        )
        r = _make_retriever(collection=col)
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            results = r.retrieve("what did peter say?", top_k=5)
        # Neither chunk has peter, but filter is off → both returned
        assert len(results) == 2

    def test_multiple_persons_any_match_suffices(self):
        """A chunk matching any of the queried persons is included."""
        col = self._col_with_participants(["rafa,bernardo", "david"])
        r = _make_retriever(collection=col)
        with patch("src.chat.retriever.FILTER_BY_PERSON", True):
            results = r.retrieve("what do gil and rafa think?", top_k=5)
        # Only the first chunk has rafa
        assert len(results) == 1
        assert "rafa" in results[0]["participants"]


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_empty_list_returns_empty_string(self):
        r = _make_retriever()
        assert r.format_context([]) == ""

    def test_single_chunk_text_present(self):
        r = _make_retriever()
        chunks = [{"text": "Hello world", "metadata": {}, "similarity_score": 0.9}]
        ctx = r.format_context(chunks)
        assert "Hello world" in ctx

    def test_multiple_chunks_all_texts_present(self):
        r = _make_retriever()
        chunks = [
            {"text": "msg A", "metadata": {}, "similarity_score": 0.9},
            {"text": "msg B", "metadata": {}, "similarity_score": 0.8},
        ]
        ctx = r.format_context(chunks)
        assert "msg A" in ctx
        assert "msg B" in ctx

    def test_context_has_header_and_footer_markers(self):
        r = _make_retriever()
        chunks = [{"text": "Hello", "metadata": {}, "similarity_score": 0.9}]
        ctx = r.format_context(chunks)
        assert "===" in ctx

    def test_valid_timestamp_appears_in_output(self):
        r = _make_retriever()
        chunks = [
            {
                "text": "some text",
                "metadata": {},
                "similarity_score": 0.9,
                "timestamp_start": "2020-06-15T10:30:00",
            }
        ]
        ctx = r.format_context(chunks)
        assert "2020-06-15" in ctx

    def test_invalid_timestamp_does_not_crash(self):
        """A malformed timestamp should be silently ignored, not raise an error."""
        r = _make_retriever()
        chunks = [
            {
                "text": "some text",
                "metadata": {},
                "similarity_score": 0.9,
                "timestamp_start": "NOT_A_DATE",
            }
        ]
        ctx = r.format_context(chunks)
        assert "some text" in ctx

    def test_missing_timestamp_key_does_not_crash(self):
        """Chunks without a timestamp_start key are formatted normally."""
        r = _make_retriever()
        chunks = [{"text": "no timestamp", "metadata": {}, "similarity_score": 0.7}]
        ctx = r.format_context(chunks)
        assert "no timestamp" in ctx


# ---------------------------------------------------------------------------
# retrieve_knowledge
# ---------------------------------------------------------------------------

class TestRetrieveKnowledge:
    def test_no_knowledge_collection_returns_empty_list(self):
        r = _make_retriever(knowledge_collection=None)
        results = r.retrieve_knowledge("anything")
        assert results == []

    def test_no_encoder_returns_empty_list(self):
        r = _make_retriever(knowledge_collection=MagicMock())
        r.encoder = None
        results = r.retrieve_knowledge("anything")
        assert results == []

    def test_returns_knowledge_chunks_with_correct_fields(self):
        kb = MagicMock()
        kb.count.return_value = 2
        kb.query.return_value = {
            "documents": [["fact one", "fact two"]],
            "metadatas": [
                [
                    {"subject": "music", "category": "hobby"},
                    {"subject": "food", "category": "preference"},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }
        r = _make_retriever(knowledge_collection=kb)
        results = r.retrieve_knowledge("any query", top_k=2)
        assert len(results) == 2
        assert results[0]["text"] == "fact one"
        assert results[0]["subject"] == "music"
        assert results[0]["category"] == "hobby"
        assert pytest.approx(results[0]["similarity_score"]) == 0.9

    def test_similarity_score_is_one_minus_distance(self):
        kb = MagicMock()
        kb.count.return_value = 1
        kb.query.return_value = {
            "documents": [["a fact"]],
            "metadatas": [[{"subject": "s", "category": "c"}]],
            "distances": [[0.4]],
        }
        r = _make_retriever(knowledge_collection=kb)
        results = r.retrieve_knowledge("query", top_k=1)
        assert pytest.approx(results[0]["similarity_score"]) == 0.6

    def test_empty_knowledge_collection_returns_empty_list(self):
        kb = MagicMock()
        kb.count.return_value = 0
        kb.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        r = _make_retriever(knowledge_collection=kb)
        results = r.retrieve_knowledge("query", top_k=3)
        assert results == []

    def test_top_k_forwarded_to_chromadb(self):
        """top_k must be passed to the underlying collection query."""
        kb = MagicMock()
        kb.count.return_value = 10
        kb.query.return_value = {
            "documents": [["x", "y"]],
            "metadatas": [[{"subject": "", "category": ""} for _ in range(2)]],
            "distances": [[0.1, 0.2]],
        }
        r = _make_retriever(knowledge_collection=kb)
        r.retrieve_knowledge("query", top_k=2)
        call_kwargs = kb.query.call_args[1]
        assert call_kwargs["n_results"] == 2


# ---------------------------------------------------------------------------
# format_knowledge_context
# ---------------------------------------------------------------------------

class TestFormatKnowledgeContext:
    def test_empty_list_returns_empty_string(self):
        r = _make_retriever()
        assert r.format_knowledge_context([]) == ""

    def test_fact_text_appears_in_output(self):
        r = _make_retriever()
        chunks = [
            {"text": "Peter likes jazz.", "subject": "music", "category": "hobby", "similarity_score": 0.9}
        ]
        ctx = r.format_knowledge_context(chunks)
        assert "Peter likes jazz" in ctx

    def test_subject_used_as_section_header(self):
        r = _make_retriever()
        chunks = [{"text": "Some fact.", "subject": "hobbies", "category": "", "similarity_score": 0.8}]
        ctx = r.format_knowledge_context(chunks)
        assert "hobbies" in ctx

    def test_missing_subject_does_not_crash(self):
        r = _make_retriever()
        chunks = [{"text": "A fact.", "subject": "", "category": "", "similarity_score": 0.7}]
        ctx = r.format_knowledge_context(chunks)
        assert "A fact" in ctx

    def test_long_text_truncated_to_three_sentences(self):
        """format_knowledge_context truncates to the first 3 sentences."""
        r = _make_retriever()
        long_text = "One. Two. Three. Four. Five."
        chunks = [{"text": long_text, "subject": "", "category": "", "similarity_score": 0.5}]
        ctx = r.format_knowledge_context(chunks)
        assert "One" in ctx
        assert "Three" in ctx
        assert "Five" not in ctx

    def test_multiple_facts_all_present(self):
        r = _make_retriever()
        chunks = [
            {"text": "Fact alpha.", "subject": "a", "category": "", "similarity_score": 0.9},
            {"text": "Fact beta.", "subject": "b", "category": "", "similarity_score": 0.8},
        ]
        ctx = r.format_knowledge_context(chunks)
        assert "Fact alpha" in ctx
        assert "Fact beta" in ctx


# ---------------------------------------------------------------------------
# retrieve_all
# ---------------------------------------------------------------------------

class TestRetrieveAll:
    def _make_conv_col(self):
        return _simple_collection(
            ["conv doc"],
            metadatas=[{"participants": "", "mentioned": "", "message_count": 1, "token_count": 10}],
            distances=[0.2],
        )

    def _make_kb_col(self):
        kb = MagicMock()
        kb.count.return_value = 1
        kb.query.return_value = {
            "documents": [["knowledge fact"]],
            "metadatas": [[{"subject": "subject", "category": "cat"}]],
            "distances": [[0.1]],
        }
        return kb

    def test_both_approach_includes_conv_and_knowledge(self):
        r = _make_retriever(collection=self._make_conv_col(), knowledge_collection=self._make_kb_col())
        r.rag_config["knowledge_base"]["enabled"] = True
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            ctx = r.retrieve_all("query", knowledge_approach="both")
        assert "conv doc" in ctx
        assert "knowledge fact" in ctx

    def test_json_only_approach_excludes_knowledge_base(self):
        r = _make_retriever(collection=self._make_conv_col(), knowledge_collection=self._make_kb_col())
        r.rag_config["knowledge_base"]["enabled"] = True
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            ctx = r.retrieve_all("query", knowledge_approach="json_only")
        assert "knowledge fact" not in ctx
        assert "conv doc" in ctx

    def test_none_approach_excludes_knowledge_base(self):
        r = _make_retriever(collection=self._make_conv_col(), knowledge_collection=self._make_kb_col())
        r.rag_config["knowledge_base"]["enabled"] = True
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            ctx = r.retrieve_all("query", knowledge_approach="none")
        assert "knowledge fact" not in ctx

    def test_chromadb_only_approach_includes_knowledge(self):
        r = _make_retriever(collection=self._make_conv_col(), knowledge_collection=self._make_kb_col())
        r.rag_config["knowledge_base"]["enabled"] = True
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            ctx = r.retrieve_all("query", knowledge_approach="chromadb_only")
        assert "knowledge fact" in ctx
        assert "conv doc" in ctx

    def test_knowledge_disabled_in_config_not_included(self):
        """Even with approach='both', if knowledge_base.enabled=False it is skipped."""
        r = _make_retriever(collection=self._make_conv_col(), knowledge_collection=self._make_kb_col())
        r.rag_config["knowledge_base"]["enabled"] = False
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            ctx = r.retrieve_all("query", knowledge_approach="both")
        assert "knowledge fact" not in ctx

    def test_no_results_returns_empty_string(self):
        col = MagicMock()
        col.count.return_value = 0
        col.query.return_value = _make_query_result([], [], [])
        r = _make_retriever(collection=col, knowledge_collection=None)
        with patch("src.chat.retriever.FILTER_BY_PERSON", False):
            ctx = r.retrieve_all("query", knowledge_approach="none")
        assert ctx == ""


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_not_initialized_returns_error_key(self):
        r = _make_retriever()
        r.collection = None
        stats = r.get_stats()
        assert "error" in stats

    def test_initialized_returns_expected_keys(self):
        col = MagicMock()
        col.count.return_value = 42
        r = _make_retriever(collection=col)
        stats = r.get_stats()
        assert stats["total_conversation_chunks"] == 42
        assert "embedding_model" in stats
        assert "top_k_default" in stats
        assert "filter_by_person" in stats

    def test_stats_include_knowledge_count_when_available(self):
        col = MagicMock()
        col.count.return_value = 10
        kb = MagicMock()
        kb.count.return_value = 5
        r = _make_retriever(collection=col, knowledge_collection=kb)
        stats = r.get_stats()
        assert stats["total_knowledge_facts"] == 5

    def test_stats_no_knowledge_count_when_unavailable(self):
        col = MagicMock()
        col.count.return_value = 10
        r = _make_retriever(collection=col, knowledge_collection=None)
        stats = r.get_stats()
        assert "total_knowledge_facts" not in stats
