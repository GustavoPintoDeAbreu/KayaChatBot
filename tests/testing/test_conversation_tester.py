"""
Unit tests for src/testing/conversation_tester.py.

All tests run without real LLM API calls or a GPU — providers and the local
model are fully mocked.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.testing.conversation_tester import (
    LLMJudgeTester,
    LocalModel,
    LLMScenarioResult,
    ScoreBreakdown,
    ConversationTurn,
    build_scoring_prompt,
    generate_scenarios,
    parse_scores,
)
# Aliases to avoid changing all test code
ConversationTester = LLMJudgeTester
ScenarioResult = LLMScenarioResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_GOOD_SCORE_JSON = (
    '{"factual_accuracy":4,"relevance":5,"language_quality":4,"tone":4}'
)

SAMPLE_FACTS = [
    {
        "id": "member_peter",
        "category": "member",
        "subject": "Peter",
        "text": "Peter is a member who loves music and audio technology.",
    },
    {
        "id": "member_gil",
        "category": "member",
        "subject": "Gil",
        "text": "Gil enjoys 8D audio and Dolby Atmos. He has a playful communication style.",
    },
    {
        "id": "group_overview",
        "category": "group",
        "subject": "Kaya group overview",
        "text": "The Kaya group is a friend group from Lisbon that communicates via WhatsApp.",
    },
]


def _mock_provider(response: str = _GOOD_SCORE_JSON):
    """Return a mock provider whose chat_completion always returns *response*."""
    provider = MagicMock()
    provider.chat_completion.return_value = response
    return provider


def _mock_local_model(response: str = "This is a model response about the subject."):
    """Return a mock LocalModel whose generate method returns *response*."""
    model = MagicMock(spec=LocalModel)
    model.available = False
    model.generate.return_value = response
    return model


@pytest.fixture
def base_config():
    return {
        "training": {"output_dir": ""},
        "model": {"max_seq_length": 4096},
        "inference": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
        "rag": {
            "knowledge_base": {"file": "data/group_knowledge.json"},
        },
    }


@pytest.fixture
def tester(base_config):
    return ConversationTester(
        provider=_mock_provider(),
        local_model=_mock_local_model(),
        config=base_config,
        turns_per_scenario=2,
    )


# ---------------------------------------------------------------------------
# ScoreBreakdown
# ---------------------------------------------------------------------------


class TestScoreBreakdown:
    def test_average(self):
        s = ScoreBreakdown(4.0, 5.0, 3.0, 4.0)
        assert s.average == pytest.approx(4.0)

    def test_failed_all_pass(self):
        s = ScoreBreakdown(3.0, 4.0, 5.0, 3.0)
        assert not s.failed

    def test_failed_one_below(self):
        s = ScoreBreakdown(2.9, 4.0, 5.0, 3.0)
        assert s.failed

    def test_to_dict_includes_average(self):
        s = ScoreBreakdown(4.0, 4.0, 4.0, 4.0)
        d = s.to_dict()
        assert "average" in d
        assert d["average"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# parse_scores
# ---------------------------------------------------------------------------


class TestParseScores:
    def test_clean_json(self):
        raw = '{"factual_accuracy":4,"relevance":5,"language_quality":3,"tone":4}'
        s = parse_scores(raw)
        assert s.factual_accuracy == 4.0
        assert s.relevance == 5.0
        assert s.language_quality == 3.0
        assert s.tone == 4.0

    def test_markdown_fenced(self):
        raw = "```json\n{\"factual_accuracy\":3,\"relevance\":3,\"language_quality\":3,\"tone\":3}\n```"
        s = parse_scores(raw)
        assert s.factual_accuracy == 3.0

    def test_extra_text_around_json(self):
        raw = 'Here are my scores: {"factual_accuracy":5,"relevance":5,"language_quality":5,"tone":5} done.'
        s = parse_scores(raw)
        assert s.factual_accuracy == 5.0

    def test_invalid_json_returns_zeros(self):
        s = parse_scores("not valid json at all")
        assert s.factual_accuracy == 0.0
        assert s.relevance == 0.0

    def test_clamps_above_5(self):
        s = parse_scores('{"factual_accuracy":9,"relevance":5,"language_quality":5,"tone":5}')
        assert s.factual_accuracy == 5.0

    def test_clamps_below_0(self):
        s = parse_scores('{"factual_accuracy":-1,"relevance":5,"language_quality":5,"tone":5}')
        assert s.factual_accuracy == 0.0

    def test_missing_keys_default_to_zero(self):
        s = parse_scores('{"factual_accuracy":4}')
        assert s.relevance == 0.0
        assert s.language_quality == 0.0
        assert s.tone == 0.0


# ---------------------------------------------------------------------------
# build_scoring_prompt
# ---------------------------------------------------------------------------


class TestBuildScoringPrompt:
    def test_contains_all_sections(self):
        prompt = build_scoring_prompt(
            question="Who is Peter?",
            reference_fact="Peter is a member who loves music.",
            response="Peter is someone who likes music a lot.",
        )
        assert "Peter is a member who loves music." in prompt
        assert "Who is Peter?" in prompt
        assert "Peter is someone who likes music a lot." in prompt


# ---------------------------------------------------------------------------
# generate_scenarios
# ---------------------------------------------------------------------------


class TestGenerateScenarios:
    def test_basic_generation(self):
        scenarios = generate_scenarios(SAMPLE_FACTS, 3)
        assert len(scenarios) == 3
        for s in scenarios:
            assert "id" in s
            assert "subject" in s
            assert "opening_question" in s
            assert "followup_question" in s
            assert "fact_text" in s
            assert "fact_excerpt" in s

    def test_scenario_id_format(self):
        scenarios = generate_scenarios(SAMPLE_FACTS, 2)
        for s in scenarios:
            assert s["id"].startswith("scenario_")

    def test_more_scenarios_than_facts(self):
        """Should cycle through facts without error."""
        scenarios = generate_scenarios(SAMPLE_FACTS, 10)
        assert len(scenarios) == 10

    def test_empty_facts(self):
        assert generate_scenarios([], 5) == []

    def test_zero_scenarios(self):
        assert generate_scenarios(SAMPLE_FACTS, 0) == []

    def test_fact_excerpt_truncated(self):
        long_text = "A" * 500
        facts = [{"id": "x", "category": "member", "subject": "X", "text": long_text}]
        scenarios = generate_scenarios(facts, 1)
        assert len(scenarios[0]["fact_excerpt"]) <= 300

    def test_member_question_contains_subject(self):
        facts = [{"id": "m", "category": "member", "subject": "Alice", "text": "Alice is nice."}]
        for _ in range(20):  # try multiple random draws
            scenarios = generate_scenarios(facts, 1)
            assert "Alice" in scenarios[0]["opening_question"]


# ---------------------------------------------------------------------------
# LocalModel
# ---------------------------------------------------------------------------


class TestLocalModel:
    def test_unavailable_when_no_model_dir(self, base_config):
        model = LocalModel(base_config)
        assert not model.available

    def test_mock_response_when_unavailable(self, base_config):
        model = LocalModel(base_config)
        resp = model.generate("Hello?")
        assert "[MOCK]" in resp

    def test_mock_response_contains_placeholder(self, base_config):
        model = LocalModel(base_config)
        assert "placeholder" in model.generate("any prompt").lower()


# ---------------------------------------------------------------------------
# ConversationTester.run_scenario
# ---------------------------------------------------------------------------


class TestRunScenario:
    def _scenario(self):
        return {
            "id": "scenario_001",
            "subject": "Peter",
            "category": "member",
            "fact_text": "Peter loves music.",
            "fact_excerpt": "Peter loves music.",
            "opening_question": "Who is Peter?",
            "followup_question": "Tell me more about Peter.",
        }

    def test_turns_recorded(self, tester):
        result = tester.run_scenario(self._scenario())
        # 2 turns × 2 (judge + model per turn) = 4 total turn objects
        assert len(result.turns) == 4

    def test_turn_roles_alternate(self, tester):
        result = tester.run_scenario(self._scenario())
        roles = [t.role for t in result.turns]
        assert roles == ["judge", "model", "judge", "model"]

    def test_model_turns_have_scores(self, tester):
        result = tester.run_scenario(self._scenario())
        model_turns = [t for t in result.turns if t.role == "model"]
        for t in model_turns:
            assert t.scores is not None

    def test_judge_turns_have_no_scores(self, tester):
        result = tester.run_scenario(self._scenario())
        judge_turns = [t for t in result.turns if t.role == "judge"]
        for t in judge_turns:
            assert t.scores is None

    def test_scenario_result_has_correct_subject(self, tester):
        result = tester.run_scenario(self._scenario())
        assert result.subject == "Peter"

    def test_high_scores_no_failure(self, base_config):
        provider = _mock_provider(
            '{"factual_accuracy":5,"relevance":5,"language_quality":5,"tone":5}'
        )
        t = ConversationTester(
            provider=provider,
            local_model=_mock_local_model(),
            config=base_config,
            turns_per_scenario=1,
        )
        result = t.run_scenario(self._scenario())
        assert not result.failure

    def test_low_scores_trigger_failure(self, base_config):
        provider = _mock_provider(
            '{"factual_accuracy":1,"relevance":1,"language_quality":1,"tone":1}'
        )
        t = ConversationTester(
            provider=provider,
            local_model=_mock_local_model(),
            config=base_config,
            turns_per_scenario=1,
        )
        result = t.run_scenario(self._scenario())
        assert result.failure
        assert len(result.failure_reasons) > 0

    def test_exception_in_model_marks_failure(self, base_config):
        model = MagicMock(spec=LocalModel)
        model.available = False
        model.generate.side_effect = RuntimeError("GPU unavailable")
        t = ConversationTester(
            provider=_mock_provider(),
            local_model=model,
            config=base_config,
            turns_per_scenario=1,
        )
        result = t.run_scenario(self._scenario())
        assert result.failure
        assert result.error is not None

    def test_single_turn_mode(self, base_config):
        t = ConversationTester(
            provider=_mock_provider(),
            local_model=_mock_local_model(),
            config=base_config,
            turns_per_scenario=1,
        )
        result = t.run_scenario(self._scenario())
        assert len(result.turns) == 2  # 1 judge + 1 model


# ---------------------------------------------------------------------------
# ConversationTester.run (full evaluation)
# ---------------------------------------------------------------------------


class TestConversationTesterRun:
    def test_report_schema(self, tester):
        scenarios = generate_scenarios(SAMPLE_FACTS, 2)
        report = tester.run(scenarios, verbose=False)

        required_keys = [
            "generated_at",
            "total_scenarios",
            "total_failures",
            "overall_averages",
            "failure_analysis",
            "scenarios",
        ]
        for key in required_keys:
            assert key in report, f"Report missing key: {key}"

    def test_report_scenario_count(self, tester):
        scenarios = generate_scenarios(SAMPLE_FACTS, 3)
        report = tester.run(scenarios, verbose=False)
        assert report["total_scenarios"] == 3
        assert len(report["scenarios"]) == 3

    def test_report_json_serialisable(self, tester):
        scenarios = generate_scenarios(SAMPLE_FACTS, 2)
        report = tester.run(scenarios, verbose=False)
        # Should not raise
        dumped = json.dumps(report)
        assert len(dumped) > 0

    def test_failure_analysis_populated_on_failure(self, base_config):
        provider = _mock_provider(
            '{"factual_accuracy":1,"relevance":1,"language_quality":1,"tone":1}'
        )
        t = ConversationTester(
            provider=provider,
            local_model=_mock_local_model(),
            config=base_config,
            turns_per_scenario=1,
        )
        scenarios = generate_scenarios(SAMPLE_FACTS, 1)
        report = t.run(scenarios, verbose=False)
        assert report["total_failures"] >= 1
        assert len(report["failure_analysis"]) >= 1

    def test_empty_scenarios(self, tester):
        report = tester.run([], verbose=False)
        assert report["total_scenarios"] == 0
        assert report["total_failures"] == 0
        assert report["overall_averages"] == {}

    def test_overall_averages_present_when_scored(self, tester):
        scenarios = generate_scenarios(SAMPLE_FACTS, 2)
        report = tester.run(scenarios, verbose=False)
        avg = report["overall_averages"]
        assert "factual_accuracy" in avg
        assert "relevance" in avg
        assert "language_quality" in avg
        assert "tone" in avg
        assert "average" in avg

    def test_generated_at_is_utc_iso(self, tester):
        scenarios = generate_scenarios(SAMPLE_FACTS, 1)
        report = tester.run(scenarios, verbose=False)
        assert report["generated_at"].endswith("Z")

    def test_turns_per_scenario_clamped_to_max_3(self, base_config):
        t = ConversationTester(
            provider=_mock_provider(),
            local_model=_mock_local_model(),
            config=base_config,
            turns_per_scenario=99,  # should be clamped to 3
        )
        assert t.turns_per_scenario == 3

    def test_turns_per_scenario_clamped_to_min_1(self, base_config):
        t = ConversationTester(
            provider=_mock_provider(),
            local_model=_mock_local_model(),
            config=base_config,
            turns_per_scenario=0,  # should be clamped to 1
        )
        assert t.turns_per_scenario == 1


# ---------------------------------------------------------------------------
# ScenarioResult serialisation
# ---------------------------------------------------------------------------


class TestScenarioResultToDict:
    def test_to_dict_schema(self):
        result = ScenarioResult(
            scenario_id="test_001",
            subject="Peter",
            category="member",
            fact_excerpt="Peter likes music.",
        )
        d = result.to_dict()
        assert d["scenario_id"] == "test_001"
        assert d["subject"] == "Peter"
        assert "turns" in d
        assert "failure" in d
        assert "failure_reasons" in d
        assert "error" in d
        assert "average_scores" in d

    def test_average_scores_none_when_no_turns(self):
        result = ScenarioResult(
            scenario_id="x",
            subject="X",
            category="member",
            fact_excerpt="",
        )
        assert result.average_scores() is None
        assert result.to_dict()["average_scores"] is None

    def test_average_scores_computed_from_turns(self):
        result = ScenarioResult(
            scenario_id="x",
            subject="X",
            category="member",
            fact_excerpt="",
        )
        result.turns.append(
            ConversationTurn(
                role="model",
                content="response",
                scores=ScoreBreakdown(4.0, 4.0, 4.0, 4.0),
            )
        )
        avg = result.average_scores()
        assert avg is not None
        assert avg["average"] == pytest.approx(4.0)
