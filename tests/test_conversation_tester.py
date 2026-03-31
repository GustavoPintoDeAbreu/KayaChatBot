"""
Unit tests for src/testing/conversation_tester.py

Covers:
  - Scenario generation from knowledge facts (valid question templates)
  - Judge-score parsing from various LLM response formats
  - Full scenario scoring with a mock judge function
  - JSON report format: required fields, correct averages, failure detection
  - Edge case: 0 scenarios
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.testing.conversation_tester import (
    QUESTION_TEMPLATES,
    ConversationScenario,
    ScoredScenario,
    generate_report,
    generate_scenarios_from_knowledge,
    parse_judge_score,
    score_scenario,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_knowledge_facts():
    return [
        {"subject": "Peter", "text": "Peter is passionate about audio engineering and has built his own speakers."},
        {"subject": "Gil", "text": "Gil enjoys travelling and recently visited Porto and Lisbon."},
        {"subject": "David", "text": "David is a football fan who supports Benfica."},
    ]


# ---------------------------------------------------------------------------
# test_generate_scenarios_from_knowledge_*
# ---------------------------------------------------------------------------

class TestGenerateScenariosFromKnowledge:
    def test_generates_one_scenario_per_fact(self, sample_knowledge_facts):
        scenarios = generate_scenarios_from_knowledge(sample_knowledge_facts)
        assert len(scenarios) == 3

    def test_scenario_question_contains_subject(self, sample_knowledge_facts):
        scenarios = generate_scenarios_from_knowledge(sample_knowledge_facts)
        for sc in scenarios:
            assert sc.subject in sc.question

    def test_scenario_uses_valid_template(self, sample_knowledge_facts):
        scenarios = generate_scenarios_from_knowledge(sample_knowledge_facts)
        for sc in scenarios:
            # The question should come from one of the templates
            matched = any(
                t.replace("{subject}", sc.subject) == sc.question
                for t in QUESTION_TEMPLATES
            )
            assert matched, f"Question '{sc.question}' does not match any template"

    def test_scenario_has_knowledge_fact(self, sample_knowledge_facts):
        scenarios = generate_scenarios_from_knowledge(sample_knowledge_facts)
        for sc, fact in zip(scenarios, sample_knowledge_facts):
            assert sc.knowledge_fact == fact["text"]

    def test_scenario_expected_keywords_nonempty(self, sample_knowledge_facts):
        scenarios = generate_scenarios_from_knowledge(sample_knowledge_facts)
        for sc in scenarios:
            assert isinstance(sc.expected_keywords, list)

    def test_custom_templates_used(self, sample_knowledge_facts):
        custom = ["Custom question about {subject}!"]
        scenarios = generate_scenarios_from_knowledge(sample_knowledge_facts, templates=custom)
        for sc in scenarios:
            assert sc.question == f"Custom question about {sc.subject}!"

    def test_empty_facts_returns_empty_list(self):
        assert generate_scenarios_from_knowledge([]) == []

    def test_facts_missing_subject_skipped(self):
        facts = [{"text": "some fact without subject"}]
        result = generate_scenarios_from_knowledge(facts)
        assert result == []

    def test_facts_missing_text_skipped(self):
        facts = [{"subject": "Peter"}]  # no 'text' key
        result = generate_scenarios_from_knowledge(facts)
        assert result == []

    def test_returns_list_of_conversation_scenarios(self, sample_knowledge_facts):
        scenarios = generate_scenarios_from_knowledge(sample_knowledge_facts)
        assert all(isinstance(s, ConversationScenario) for s in scenarios)


# ---------------------------------------------------------------------------
# test_parse_judge_score_*
# ---------------------------------------------------------------------------

class TestParseJudgeScore:
    def test_parse_score_colon_format(self):
        assert parse_judge_score("Score: 8") == pytest.approx(8.0)

    def test_parse_score_slash_format(self):
        assert parse_judge_score("I give this 7/10 overall.") == pytest.approx(7.0)

    def test_parse_score_bare_integer_line(self):
        assert parse_judge_score("Some text\n6\nMore text") == pytest.approx(6.0)

    def test_parse_score_float_value(self):
        assert parse_judge_score("Score: 7.5") == pytest.approx(7.5)

    def test_parse_score_case_insensitive(self):
        assert parse_judge_score("SCORE: 9") == pytest.approx(9.0)
        assert parse_judge_score("score: 4") == pytest.approx(4.0)

    def test_parse_score_caps_at_10(self):
        assert parse_judge_score("Score: 11") == pytest.approx(10.0)
        assert parse_judge_score("15/10") == pytest.approx(10.0)

    def test_parse_score_no_score_returns_zero(self):
        assert parse_judge_score("The answer was good but I cannot give a score.") == pytest.approx(0.0)

    def test_parse_score_empty_string_returns_zero(self):
        assert parse_judge_score("") == pytest.approx(0.0)

    def test_parse_score_zero_score(self):
        assert parse_judge_score("Score: 0") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# test_score_scenario_*
# ---------------------------------------------------------------------------

class TestScoreScenario:
    def _make_scenario(self, subject="Peter", question="What do you know about Peter?"):
        return ConversationScenario(
            question=question,
            expected_keywords=["audio"],
            knowledge_fact="Peter likes audio.",
            subject=subject,
        )

    def test_score_scenario_calls_judge_fn(self):
        judge_fn = MagicMock(return_value="Score: 8")
        scenario = self._make_scenario()
        result = score_scenario(scenario, "Peter likes audio.", judge_fn)
        judge_fn.assert_called_once()

    def test_score_scenario_passes_question_and_response(self):
        judge_fn = MagicMock(return_value="Score: 7")
        scenario = self._make_scenario()
        model_response = "Peter is known for audio work."
        score_scenario(scenario, model_response, judge_fn)
        args = judge_fn.call_args[0]
        assert scenario.question in args
        assert model_response in args

    def test_score_scenario_score_parsed_correctly(self):
        judge_fn = MagicMock(return_value="Score: 9")
        result = score_scenario(self._make_scenario(), "response", judge_fn)
        assert result.score == pytest.approx(9.0)

    def test_score_scenario_passed_when_score_gte_5(self):
        judge_fn = MagicMock(return_value="Score: 5")
        result = score_scenario(self._make_scenario(), "response", judge_fn)
        assert result.passed is True

    def test_score_scenario_failed_when_score_lt_5(self):
        judge_fn = MagicMock(return_value="Score: 3")
        result = score_scenario(self._make_scenario(), "response", judge_fn)
        assert result.passed is False

    def test_score_scenario_returns_scored_scenario(self):
        judge_fn = MagicMock(return_value="Score: 6")
        result = score_scenario(self._make_scenario(), "answer", judge_fn)
        assert isinstance(result, ScoredScenario)


# ---------------------------------------------------------------------------
# test_generate_report_*
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def _make_scored(self, score: float, subject="Peter") -> ScoredScenario:
        scenario = ConversationScenario(
            question=f"What do you know about {subject}?",
            subject=subject,
            knowledge_fact="some fact",
            expected_keywords=[],
        )
        return ScoredScenario(
            scenario=scenario,
            model_response="some response",
            judge_response=f"Score: {score}",
            score=score,
            passed=score >= 5.0,
        )

    def test_report_required_fields_present(self):
        report = generate_report([self._make_scored(7.0)])
        for field in ("total_scenarios", "passed", "failed", "average_score", "pass_rate", "scenarios"):
            assert field in report, f"Missing field: {field}"

    def test_report_correct_totals(self):
        scored = [self._make_scored(8.0), self._make_scored(3.0), self._make_scored(6.0)]
        report = generate_report(scored)
        assert report["total_scenarios"] == 3
        assert report["passed"] == 2
        assert report["failed"] == 1

    def test_report_average_score(self):
        scored = [self._make_scored(6.0), self._make_scored(8.0)]
        report = generate_report(scored)
        assert report["average_score"] == pytest.approx(7.0, abs=0.01)

    def test_report_pass_rate(self):
        scored = [self._make_scored(8.0), self._make_scored(3.0)]
        report = generate_report(scored)
        assert report["pass_rate"] == pytest.approx(0.5, abs=0.001)

    def test_report_scenarios_list_has_correct_length(self):
        scored = [self._make_scored(5.0), self._make_scored(7.0), self._make_scored(2.0)]
        report = generate_report(scored)
        assert len(report["scenarios"]) == 3

    def test_report_scenario_fields(self):
        report = generate_report([self._make_scored(8.0, "Gil")])
        sc = report["scenarios"][0]
        for f in ("question", "subject", "model_response", "score", "passed"):
            assert f in sc

    def test_report_zero_scenarios(self):
        report = generate_report([])
        assert report["total_scenarios"] == 0
        assert report["passed"] == 0
        assert report["failed"] == 0
        assert report["average_score"] == pytest.approx(0.0)
        assert report["pass_rate"] == pytest.approx(0.0)
        assert report["scenarios"] == []

    def test_report_is_json_serializable(self):
        import json
        scored = [self._make_scored(7.0), self._make_scored(4.0)]
        report = generate_report(scored)
        serialized = json.dumps(report)
        loaded = json.loads(serialized)
        assert loaded["total_scenarios"] == 2

    def test_report_failure_detected(self):
        scored = [self._make_scored(2.0), self._make_scored(1.0)]
        report = generate_report(scored)
        assert report["failed"] == 2
        assert report["passed"] == 0
        assert report["pass_rate"] == pytest.approx(0.0)
