"""Unit tests for src/data/term_blocklist."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.term_blocklist import (
    compile_blocklist,
    filter_list,
    filter_messages,
    is_blocked,
    redact_sentences,
)

PATTERNS = compile_blocklist(["Dolby Atmos", "8D audio"])


class TestIsBlocked:
    def test_case_insensitive_phrase_match(self):
        assert is_blocked("usa dolby ATMOS aqui", PATTERNS)
        assert is_blocked("Dolby atmos rocks", PATTERNS)
        assert is_blocked("loves 8D Audio", PATTERNS)

    def test_unrelated_text_not_blocked(self):
        assert not is_blocked("Gil enjoys techno music.", PATTERNS)

    def test_word_boundary_avoids_substring_false_positive(self):
        # "atmosphere" must not trip the "Dolby Atmos" phrase pattern.
        assert not is_blocked("the atmosphere was great", PATTERNS)

    def test_empty_inputs(self):
        assert not is_blocked("", PATTERNS)
        assert not is_blocked("Dolby Atmos", [])


class TestFilterList:
    def test_drops_blocked_entries(self):
        items = ["techno", "Dolby Atmos setup", "running"]
        assert filter_list(items, PATTERNS) == ["techno", "running"]

    def test_no_patterns_keeps_all(self):
        items = ["a", "Dolby Atmos"]
        assert filter_list(items, []) == items

    def test_empty_list(self):
        assert filter_list([], PATTERNS) == []


class TestRedactSentences:
    def test_drops_only_blocked_sentence(self):
        text = "He likes music. He uses Dolby Atmos. He runs."
        assert redact_sentences(text, PATTERNS) == "He likes music. He runs."

    def test_all_blocked_returns_empty(self):
        assert redact_sentences("Big fan of 8D audio.", PATTERNS) == ""

    def test_no_blocked_unchanged(self):
        text = "Gil owns a dog. Gil runs."
        assert redact_sentences(text, PATTERNS) == text


class TestFilterMessages:
    def test_drops_blocked_messages(self):
        msgs = [{"text": "hi"}, {"text": "8D audio is cool"}, {"text": "bye"}]
        assert filter_messages(msgs, PATTERNS) == [{"text": "hi"}, {"text": "bye"}]

    def test_no_patterns_keeps_all(self):
        msgs = [{"text": "Dolby Atmos"}]
        assert filter_messages(msgs, []) == msgs
