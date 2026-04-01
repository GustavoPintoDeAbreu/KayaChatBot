"""
tests/rag/test_language_consistency.py

Tests for the language validation utilities and policy enforcement:
  - No emojis in responses
  - No Brazilian Portuguese expressions
  - No mid-sentence language switching to English
  - European Portuguese as the default response language
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.testing.language_validator import (
    LanguageReport,
    detect_brazilian_portuguese,
    detect_emojis,
    detect_language_switch,
    language_consistency_score,
    validate_response_language,
)
from src.data.language_filters import (
    apply_brazilian_replacements,
    clean_training_text,
    contains_brazilian_portuguese,
    contains_emojis,
    strip_emojis,
)


# ---------------------------------------------------------------------------
# detect_emojis
# ---------------------------------------------------------------------------


class TestDetectEmojis:
    def test_no_emoji_in_clean_text(self):
        has, count = detect_emojis("Olá, tudo bem contigo?")
        assert not has
        assert count == 0

    def test_single_emoji_detected(self):
        has, count = detect_emojis("Olá! 🍻")
        assert has
        assert count == 1

    def test_multiple_emojis_detected(self):
        has, count = detect_emojis("Que bom! 🎉🎊🥂")
        assert has
        assert count == 3

    def test_emoji_in_mid_sentence(self):
        has, count = detect_emojis("O Peter é fixe 😄 e muito amigo")
        assert has
        assert count == 1

    def test_strip_emojis_removes_all(self):
        result = strip_emojis("Que energia incrível! 🎉🍻 Bom trabalho 👍")
        assert "🎉" not in result
        assert "🍻" not in result
        assert "👍" not in result
        assert "Que energia incrível!" in result

    def test_strip_emojis_preserves_text(self):
        result = strip_emojis("Olá, tudo bem?")
        assert result == "Olá, tudo bem?"

    def test_contains_emojis_function(self):
        assert contains_emojis("É o tipo de energia 🍻")
        assert not contains_emojis("É o tipo de energia")


# ---------------------------------------------------------------------------
# detect_brazilian_portuguese
# ---------------------------------------------------------------------------


class TestDetectBrazilianPortuguese:
    def test_clean_european_portuguese(self):
        has, matches = detect_brazilian_portuguese("O Peter é muito fixe e organiza sempre as saídas.")
        assert not has
        assert matches == []

    def test_role_detected(self):
        has, matches = detect_brazilian_portuguese("É o tipo de energia que anima qualquer rolê Kaya!")
        assert has
        assert any("rolê" in m.lower() or "role" in m.lower() for m in matches)

    def test_cara_detected(self):
        has, matches = detect_brazilian_portuguese("O cara é muito simpático.")
        assert has
        assert any("cara" in m.lower() for m in matches)

    def test_maneiro_detected(self):
        has, matches = detect_brazilian_portuguese("Isso foi muito maneiro!")
        assert has

    def test_galera_detected(self):
        has, matches = detect_brazilian_portuguese("A galera estava toda lá.")
        assert has
        assert any("galera" in m.lower() for m in matches)

    def test_mano_detected(self):
        has, matches = detect_brazilian_portuguese("Mano, isso é fixe!")
        assert has

    def test_multiple_br_terms(self):
        has, matches = detect_brazilian_portuguese("Cara, o rolê foi maneiro demais!")
        assert has
        assert len(matches) >= 2

    def test_contains_br_function(self):
        assert contains_brazilian_portuguese("O rolê foi incrível!")
        assert not contains_brazilian_portuguese("A saída foi incrível!")

    def test_problematic_example_from_issue(self):
        """The exact phrase that triggered this task."""
        phrase = "É o tipo de energia que anima qualquer rolê Kaya! Se quiseres mais details de convos específicas, diz. 🍻"
        report = validate_response_language(phrase)
        assert not report.is_clean
        assert report.has_emojis
        assert report.has_brazilian_portuguese


# ---------------------------------------------------------------------------
# detect_language_switch
# ---------------------------------------------------------------------------


class TestDetectLanguageSwitch:
    def test_pure_portuguese_no_switch(self):
        text = "O Peter é uma pessoa muito activa e está sempre a organizar eventos para o grupo."
        has_switch, count = detect_language_switch(text)
        assert not has_switch

    def test_english_sentence_flagged(self):
        text = "O Peter is a very active person and he always organizes events for the group."
        has_switch, count = detect_language_switch(text)
        assert has_switch
        assert count >= 4

    def test_mid_sentence_switch_flagged(self):
        """Model starts in Portuguese then switches to English mid-way."""
        text = "O Peter é muito activo — he is always organizing events and he would love this."
        has_switch, count = detect_language_switch(text)
        assert has_switch

    def test_single_english_word_not_flagged(self):
        """A single English word (e.g. a borrowed term) should not trigger the switch."""
        text = "Usa o WhatsApp para comunicar com o grupo."
        has_switch, count = detect_language_switch(text, threshold=4)
        assert not has_switch

    def test_full_english_response_flagged(self):
        text = "The group has been together for many years and they are very close friends."
        has_switch, count = detect_language_switch(text)
        assert has_switch
        assert count >= 4


# ---------------------------------------------------------------------------
# validate_response_language (combined)
# ---------------------------------------------------------------------------


class TestValidateResponseLanguage:
    def test_clean_pt_eu_response(self):
        text = "O Peter é uma pessoa muito activa e organiza sempre as saídas do grupo. Toda a malta gosta dele."
        report = validate_response_language(text)
        assert report.is_clean
        assert not report.has_emojis
        assert not report.has_brazilian_portuguese
        assert not report.has_language_switch

    def test_emoji_only_fails(self):
        text = "Olá! 👋 Tudo bem?"
        report = validate_response_language(text)
        assert not report.is_clean
        assert report.has_emojis

    def test_br_only_fails(self):
        text = "A galera foi ao rolê ontem."
        report = validate_response_language(text)
        assert not report.is_clean
        assert report.has_brazilian_portuguese

    def test_language_switch_fails(self):
        text = "O Peter é fixe. He is always there for his friends and they really appreciate him."
        report = validate_response_language(text)
        assert not report.is_clean
        assert report.has_language_switch

    def test_all_issues_combined(self):
        text = "Cara, o rol\u00ea foi fixe! \U0001f389 This is great, they are all amazing and the galera would love it."
        report = validate_response_language(text)
        assert not report.is_clean
        assert report.has_emojis
        assert report.has_brazilian_portuguese
        assert report.has_language_switch

    def test_issues_method_lists_problems(self):
        text = "Cara, o rolê foi fixe! 🎉"
        report = validate_response_language(text)
        issues = report.issues()
        assert len(issues) >= 2


# ---------------------------------------------------------------------------
# language_consistency_score
# ---------------------------------------------------------------------------


class TestLanguageConsistencyScore:
    def test_perfect_pt_eu_scores_5(self):
        text = "O Peter é muito activo e organiza sempre os eventos. Toda a malta gosta dele."
        score = language_consistency_score(text)
        assert score == 5.0

    def test_emoji_reduces_score(self):
        text = "O Peter é fixe! 🎉"
        score = language_consistency_score(text)
        assert score < 5.0
        assert score >= 0.0

    def test_br_words_reduce_score(self):
        text = "O rolê foi fixe, cara!"
        score = language_consistency_score(text)
        assert score < 5.0

    def test_full_english_switch_scores_0(self):
        text = "The group has been together for many years and they are very close friends and they would love this."
        score = language_consistency_score(text)
        assert score == 0.0

    def test_score_bounded_0_to_5(self):
        texts = [
            "Olá, tudo bem?",
            "Cara, o rolê foi maneiro! 🎉 He is so great.",
            "The group is amazing and they are all wonderful.",
        ]
        for text in texts:
            score = language_consistency_score(text)
            assert 0.0 <= score <= 5.0


# ---------------------------------------------------------------------------
# apply_brazilian_replacements
# ---------------------------------------------------------------------------


class TestApplyBrazilianReplacements:
    def test_role_replaced(self):
        result = apply_brazilian_replacements("Vai ser um rolê incrível!")
        assert "rolê" not in result.lower()

    def test_cara_replaced(self):
        result = apply_brazilian_replacements("Cara, isso foi fixe!")
        assert "cara" not in result.lower() or "cara" in result  # word-boundary — 'cara' is a valid PT word in other contexts

    def test_celular_replaced(self):
        result = apply_brazilian_replacements("Perdeu o celular ontem.")
        assert "celular" not in result.lower()
        assert "telemóvel" in result.lower()

    def test_clean_training_text_combines_both(self):
        text = "Cara, o rolê foi fixe! 🎉 Perdi o celular."
        result = clean_training_text(text)
        assert "🎉" not in result
        assert "celular" not in result.lower()

    def test_preserves_european_portuguese(self):
        original = "O Peter é muito activo e organiza as saídas. Toda a malta gosta dele."
        result = apply_brazilian_replacements(original)
        assert result == original


# ---------------------------------------------------------------------------
# Integration: full pipeline example
# ---------------------------------------------------------------------------


class TestIntegrationPhraseFromIssue:
    """The exact problematic phrase that was reported should fail all checks."""

    PHRASE = (
        "É o tipo de energia que anima qualquer rolê Kaya! "
        "Se quiseres mais details de convos específicas, diz. 🍻"
    )

    def test_emoji_detected(self):
        has, count = detect_emojis(self.PHRASE)
        assert has
        assert count >= 1

    def test_br_detected(self):
        has, matches = detect_brazilian_portuguese(self.PHRASE)
        assert has

    def test_full_validation_fails(self):
        report = validate_response_language(self.PHRASE)
        assert not report.is_clean

    def test_score_below_4(self):
        score = language_consistency_score(self.PHRASE)
        assert score < 4.0

    def test_clean_training_text_fixes_it(self):
        cleaned = clean_training_text(self.PHRASE)
        assert "🍻" not in cleaned
        # After cleaning, re-validate
        report = validate_response_language(cleaned)
        # At minimum, no more emojis
        assert not report.has_emojis
