"""Unit tests for src/chat/suggestions.parse_suggestions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.suggestions import parse_suggestions


class TestParseSuggestions:
    def test_plain_lines(self):
        raw = "O que faz o Gil ao fim de semana?\nQuem organiza os jantares?\nOnde vive o Manuel?"
        result = parse_suggestions(raw, count=3)
        assert len(result) == 3
        assert result[0] == "O que faz o Gil ao fim de semana?"

    def test_strips_numbering_and_bullets(self):
        raw = "1. Quem é o Rafa?\n2) O que faz o Peter?\n- Onde sai o grupo?"
        result = parse_suggestions(raw, count=3)
        assert result == ["Quem é o Rafa?", "O que faz o Peter?", "Onde sai o grupo?"]

    def test_drops_non_questions(self):
        raw = "Aqui estão as perguntas:\nQuem é o Gil?\nIsto não é pergunta."
        result = parse_suggestions(raw, count=5)
        assert result == ["Quem é o Gil?"]

    def test_dedupes_case_insensitive(self):
        raw = "Quem é o Gil?\nquem é o gil?\nOnde vive o Manuel?"
        result = parse_suggestions(raw, count=5)
        assert result == ["Quem é o Gil?", "Onde vive o Manuel?"]

    def test_respects_count_cap(self):
        raw = "\n".join(f"Pergunta {i}?" for i in range(10))
        assert len(parse_suggestions(raw, count=2)) == 2

    def test_strips_quotes(self):
        assert parse_suggestions('"Quem é o Gil?"', count=1) == ["Quem é o Gil?"]

    def test_trims_after_question_mark(self):
        raw = "Quem é o Gil? extra texto a mais aqui"
        assert parse_suggestions(raw, count=1) == ["Quem é o Gil?"]

    def test_empty_input(self):
        assert parse_suggestions("", count=3) == []
        assert parse_suggestions(None, count=3) == []
