"""Unit tests for src/data/question_bank."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.question_bank import build_questions

MEMBERS = {
    "members": [
        {"name": "Gustavo", "aliases": ["gustavo", "gugu"]},
        {"name": "Peter", "aliases": ["peter", "piteru"]},
    ]
}


class TestBuildQuestions:
    def test_deterministic_with_seed(self):
        a = build_questions(MEMBERS, seed=1)
        b = build_questions(MEMBERS, seed=1)
        assert a == b

    def test_no_duplicates(self):
        qs = build_questions(MEMBERS)
        assert len(qs) == len(set(qs))

    def test_covers_targeted_behaviors(self):
        qs = " || ".join(build_questions(MEMBERS))
        # per-member, group-wide superlative, opinion, and general coverage
        assert "Quem é o Gustavo?" in qs or "Who is Gustavo?" in qs
        assert "mais convencido do grupo" in qs
        assert "O que achas de" in qs
        assert "Quem são os membros do grupo?" in qs

    def test_uses_aliases(self):
        qs = " || ".join(build_questions(MEMBERS))
        assert "gugu" in qs  # alias surfaced in a question

    def test_per_category_caps(self):
        small = build_questions(MEMBERS, per_category=1)
        full = build_questions(MEMBERS)
        assert len(small) < len(full)
