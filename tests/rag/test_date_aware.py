"""
Unit tests for date-aware facts:
- temporal-intent detection on queries
- relative-age rendering
- conditional date surfacing in the context formatters
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.chat.retriever import (
    ConversationRetriever,
    _has_temporal_intent,
    _recency_window,
    _relative_age,
)


@pytest.fixture
def retriever():
    """Retriever instance without DB/model — only the pure formatters are used."""
    config = {
        "rag": {"top_k": 5, "filter_by_person": True},
        "data": {"group_members_file": None},
    }
    return ConversationRetriever(config)


class TestTemporalIntent:
    @pytest.mark.parametrize("query", [
        "Quando é que o Gil partiu o dedo?",
        "Há quanto tempo o Manuel vive em Malta?",
        "When did Rafa have his son?",
        "How long ago was that?",
        "Isso é recente?",
        "Qual foi a última vez que se juntaram?",
        "What's the latest on the project?",
    ])
    def test_detects_timing_questions(self, query):
        assert _has_temporal_intent(query) is True

    @pytest.mark.parametrize("query", [
        "Quem é o Gil?",
        "O que é que o grupo costuma fazer?",
        "Tell me about Gustavo",
        "Onde é que o grupo sai?",
        "",
    ])
    def test_ignores_non_timing_questions(self, query):
        assert _has_temporal_intent(query) is False


class TestRecencyWindow:
    """The failure this exists for, live on 2026-08-29:

        "Consegues me dizer o que é que o grupo fez ontem? Fomos jantar fora."
        -> "Não tenho registos claros sobre o que aconteceu ontem."

    after retrieving 6,109 characters. The dinner was in ChromaDB, correctly
    chunked, from the previous evening; the top hits were November 2025,
    December 2025 and July 2026. Nearest-neighbour cannot find "ontem" — the
    question shares no vocabulary with an evening of restaurant chatter.
    """

    NOW = datetime(2026, 9, 4, 12, 0)

    @pytest.mark.parametrize("query,expected_start,expected_end", [
        ("o que é que o grupo fez ontem?", "2026-09-03", "2026-09-03"),
        ("what did we do yesterday", "2026-09-03", "2026-09-03"),
        ("last night", "2026-09-03", "2026-09-03"),
        ("o que fizemos esta semana", "2026-08-28", "2026-09-04"),
        ("what happened this week", "2026-08-28", "2026-09-04"),
        ("alguma coisa hoje?", "2026-09-04", "2026-09-04"),
    ])
    def test_names_a_window(self, query, expected_start, expected_end):
        start, end, _label = _recency_window(query, self.NOW)
        assert start.startswith(expected_start)
        assert end.startswith(expected_end)

    def test_the_end_bound_covers_the_whole_day(self):
        """A chunk written at 23:41 on the last day is inside the window."""
        _start, end, _label = _recency_window("ontem", self.NOW)
        assert end.endswith("T23:59:59")

    def test_longest_phrase_wins(self):
        """"semana passada" must not be shadowed by "esta semana"."""
        start, end, label = _recency_window("o que aconteceu na semana passada?", self.NOW)
        assert label == "a semana passada"
        assert start.startswith("2026-08-21") and end.startswith("2026-08-28")

    @pytest.mark.parametrize("query", [
        "Quem é o Gil?",
        "o jantar de sexta",
        "quando foi o jantar?",   # temporal intent, but names no period
        "",
    ])
    def test_no_window_for_a_query_that_names_no_period(self, query):
        assert _recency_window(query, self.NOW) is None

    @pytest.mark.parametrize("query", [
        "o que é que o grupo fez ontem?",
        "what did we do yesterday",
        "o que fizemos esta semana",
        "last night",
    ])
    def test_these_are_also_temporal_intent(self, query):
        """They were not, which is half of why the answer had no dates either."""
        assert _has_temporal_intent(query) is True


class FakeCollection:
    """Minimal ChromaDB stand-in: get(include=...) and get(ids=...)."""

    def __init__(self, rows):
        self.rows = rows  # [(id, document, metadata)]

    def count(self):
        return len(self.rows)

    def get(self, ids=None, where=None, limit=None, include=None):
        rows = self.rows if ids is None else [r for r in self.rows if r[0] in ids]
        out = {"ids": [r[0] for r in rows]}
        if include and "documents" in include:
            out["documents"] = [r[1] for r in rows]
        if not include or "metadatas" in include:
            out["metadatas"] = [r[2] for r in rows]
        return out


def _dated_retriever(retriever, rows):
    retriever.collection = FakeCollection(rows)
    return retriever


class TestTheWindowReachesTheResults:
    def test_yesterdays_chunks_are_put_in_front(self, retriever, monkeypatch):
        monkeypatch.setattr("src.chat.retriever._recency_window",
                            lambda query, now=None: ("2026-08-28T00:00:00",
                                                     "2026-08-28T23:59:59", "ontem"))
        _dated_retriever(retriever, [
            ("a", "Rafa: vao entrando, ta em meu nome",
             {"timestamp_start": "2026-08-28T19:44", "timestamp_end": "2026-08-28T19:50",
              "scope": "shared"}),
            ("b", "conversa de dezembro",
             {"timestamp_start": "2025-12-24T15:16", "timestamp_end": "2025-12-24T15:20",
              "scope": "shared"}),
        ])
        semantic = [{"text": "conversa de dezembro", "similarity_score": 0.5, "rank": 1}]

        merged = retriever._prepend_recency_window(
            "o que fizemos ontem?", semantic, top_k=5, scope="shared", exclude_from=None)

        assert merged[0]["text"].startswith("Rafa: vao entrando")
        assert merged[0]["rank"] == 1

    def test_a_chunk_already_retrieved_is_not_duplicated(self, retriever, monkeypatch):
        monkeypatch.setattr("src.chat.retriever._recency_window",
                            lambda query, now=None: ("2026-08-28T00:00:00",
                                                     "2026-08-28T23:59:59", "ontem"))
        _dated_retriever(retriever, [
            ("a", "o jantar de ontem",
             {"timestamp_start": "2026-08-28T19:44", "timestamp_end": "2026-08-28T19:50",
              "scope": "shared"}),
        ])
        semantic = [{"text": "o jantar de ontem", "similarity_score": 0.9, "rank": 1}]

        merged = retriever._prepend_recency_window(
            "o que fizemos ontem?", semantic, top_k=5, scope="shared", exclude_from=None)

        assert [c["text"] for c in merged] == ["o jantar de ontem"]

    def test_scope_is_enforced_on_this_path_too(self, retriever, monkeypatch):
        """This path has no `where` clause in front of it, so the in-Python
        scope check is load-bearing rather than defence in depth. A DM must
        never reach the group through a date query."""
        monkeypatch.setattr("src.chat.retriever._recency_window",
                            lambda query, now=None: ("2026-08-28T00:00:00",
                                                     "2026-08-28T23:59:59", "ontem"))
        _dated_retriever(retriever, [
            ("secret", "segredo dito em privado",
             {"timestamp_start": "2026-08-28T19:44", "timestamp_end": "2026-08-28T19:50",
              "scope": "dm:abc"}),
        ])

        merged = retriever._prepend_recency_window(
            "o que fizemos ontem?", [], top_k=5, scope="shared", exclude_from=None)

        assert merged == []

    def test_a_query_with_no_window_is_left_exactly_as_it_was(self, retriever):
        semantic = [{"text": "x", "similarity_score": 0.5, "rank": 1}]

        assert retriever._prepend_recency_window(
            "quem é o Gil?", semantic, top_k=5, scope="shared",
            exclude_from=None) is semantic

    def test_a_broken_store_never_costs_an_answer(self, retriever, monkeypatch):
        monkeypatch.setattr("src.chat.retriever._recency_window",
                            lambda query, now=None: ("2026-08-28T00:00:00",
                                                     "2026-08-28T23:59:59", "ontem"))

        class Exploding:
            def count(self):
                return 1

            def get(self, **kwargs):
                raise RuntimeError("no date metadata in this store")

        retriever.collection = Exploding()
        semantic = [{"text": "x", "similarity_score": 0.5, "rank": 1}]

        assert retriever._prepend_recency_window(
            "o que fizemos ontem?", semantic, top_k=5, scope="shared",
            exclude_from=None) == semantic

    def test_the_index_is_cached_until_the_count_changes(self, retriever):
        rows = [("a", "x", {"timestamp_start": "2026-08-28T19:44", "scope": "shared"})]
        collection = FakeCollection(rows)
        retriever.collection = collection

        assert retriever._date_index() == [("2026-08-28T19:44", "a")]
        rows.append(("b", "y", {"timestamp_start": "2026-08-29T10:00", "scope": "shared"}))
        assert len(retriever._date_index()) == 2, "a new chunk must invalidate the cache"


class TestRelativeAge:
    def test_empty_for_missing_or_bad(self):
        assert _relative_age(None) == ""
        assert _relative_age("not-a-date") == ""

    def test_today(self):
        today = datetime(2026, 6, 9)
        assert _relative_age("2026-06-09T10:00:00", today=today) == "hoje"

    def test_days_and_weeks(self):
        today = datetime(2026, 6, 9)
        assert _relative_age("2026-06-06T10:00:00", today=today) == "há ~3 dias"
        assert "semana" in _relative_age("2026-05-20T10:00:00", today=today)

    def test_months_and_years(self):
        today = datetime(2026, 6, 9)
        assert "mes" in _relative_age("2026-03-09T10:00:00", today=today).replace("ê", "e")
        assert "ano" in _relative_age("2024-06-09T10:00:00", today=today)

    def test_future_date_returns_empty(self):
        today = datetime(2026, 6, 9)
        assert _relative_age("2026-12-09T10:00:00", today=today) == ""


class TestConditionalDateSurfacing:
    def _kb_chunks(self):
        return [{
            "text": "Manuel vai casar em breve.",
            "subject": "Manuel",
            "category": "member",
            "event_date_hint": "no próximo mês",
            "last_updated": "2026-05-01T10:00:00",
            "source_date_start": "2026-04-01T10:00:00",
            "source_date_end": "2026-05-01T10:00:00",
        }]

    def test_knowledge_dates_hidden_by_default(self, retriever):
        out = retriever.format_knowledge_context(self._kb_chunks(), show_dates=False)
        assert "Manuel" in out
        assert "referência temporal" not in out
        assert "atualizado" not in out

    def test_knowledge_dates_shown_on_demand(self, retriever):
        out = retriever.format_knowledge_context(self._kb_chunks(), show_dates=True)
        # Explicit text hint wins over message timestamps (mixed rule).
        assert "no próximo mês" in out

    def test_knowledge_falls_back_to_message_dates(self, retriever):
        chunks = self._kb_chunks()
        chunks[0]["event_date_hint"] = ""  # no explicit hint → use last_updated
        out = retriever.format_knowledge_context(chunks, show_dates=True)
        assert "atualizado" in out

    def test_conversation_dates_hidden_by_default(self, retriever):
        chunks = [{"text": "Gil: olá", "timestamp_start": "2026-01-01T10:00:00"}]
        out = retriever.format_context(chunks, show_dates=False)
        assert "2026-01-01" not in out

    def test_conversation_dates_shown_on_demand(self, retriever):
        chunks = [{"text": "Gil: olá", "timestamp_start": "2026-01-01T10:00:00"}]
        out = retriever.format_context(chunks, show_dates=True)
        assert "2026-01-01" in out
