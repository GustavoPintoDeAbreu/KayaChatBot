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
