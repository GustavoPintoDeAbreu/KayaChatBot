"""
Unit tests for src/testing/conversation_tester.py and src/testing/benchmark.py.

Covers: keyword scoring, single-scenario execution, run_all with limits,
summary structure, matrix building, markdown/JSON formatting, and report saving.
Uses mock response functions — no real model loading required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.testing.conversation_tester import (
    SCENARIOS,
    ConversationTester,
    ScenarioResult,
)
from src.testing.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tester():
    """ConversationTester with the default scenarios."""
    return ConversationTester()


@pytest.fixture
def tiny_tester():
    """ConversationTester with a small custom scenario list."""
    scenarios = [
        {
            "id": "t001",
            "category": "factual",
            "question_pt": "Qual é a capital de Portugal?",
            "question_en": "What is the capital of Portugal?",
            "expected_keywords": ["lisboa", "lisbon"],
        },
        {
            "id": "t002",
            "category": "conversational",
            "question_pt": "Olá, tudo bem?",
            "question_en": "Hello, how are you?",
            "expected_keywords": ["olá", "hello", "bem", "good"],
        },
        {
            "id": "t003",
            "category": "member_knowledge",
            "question_pt": "Quem organiza os jantares?",
            "question_en": "Who organizes the dinners?",
            "expected_keywords": ["jantar", "dinner"],
        },
    ]
    return ConversationTester(scenarios=scenarios)


@pytest.fixture
def base_config():
    """Minimal config dict matching the shape the benchmark runner expects."""
    return {
        "model": {
            "model_id": "test-model/v1",
            "max_seq_length": 4096,
        },
        "rag": {
            "top_k": 5,
        },
        "benchmark": {
            "scenarios_per_config": 2,
            "output_dir": "reports/benchmarks/",
            "dimensions": ["knowledge_approaches", "languages"],
            "knowledge_approaches": ["both", "none"],
            "languages": ["pt", "en"],
            "context_sizes": [
                {"max_seq_length": 2048, "top_k": 3},
                {"max_seq_length": 8192, "top_k": 10},
            ],
            "models": [],
        },
    }


# ---------------------------------------------------------------------------
# ConversationTester — score_response
# ---------------------------------------------------------------------------

class TestScoreResponse:
    def test_all_keywords_match(self, tester):
        score, matched = tester.score_response(
            "Lisboa é a capital de Portugal", ["lisboa", "portugal"]
        )
        assert score == 1.0
        assert set(matched) == {"lisboa", "portugal"}

    def test_no_keywords_match(self, tester):
        score, matched = tester.score_response(
            "The quick brown fox", ["lisboa", "portugal"]
        )
        assert score == 0.0
        assert matched == []

    def test_partial_match(self, tester):
        score, matched = tester.score_response(
            "Lisboa is nice", ["lisboa", "portugal"]
        )
        assert score == pytest.approx(0.5)
        assert matched == ["lisboa"]

    def test_case_insensitive(self, tester):
        score, matched = tester.score_response(
            "LISBOA IS GREAT", ["lisboa"]
        )
        assert score == 1.0

    def test_empty_keywords(self, tester):
        score, matched = tester.score_response("anything", [])
        assert score == 1.0
        assert matched == []

    def test_empty_response(self, tester):
        score, matched = tester.score_response("", ["keyword"])
        assert score == 0.0

    def test_substring_matching(self, tester):
        """Keywords should match as substrings, not just whole words."""
        score, matched = tester.score_response(
            "organizado pelo grupo", ["organiz"]
        )
        assert score == 1.0
        assert matched == ["organiz"]


# ---------------------------------------------------------------------------
# ConversationTester — run_scenario
# ---------------------------------------------------------------------------

class TestRunScenario:
    def test_basic_run(self, tiny_tester):
        mock_fn = MagicMock(return_value="A capital é Lisboa")
        scenario = tiny_tester.scenarios[0]

        result = tiny_tester.run_scenario(scenario, "pt", mock_fn)

        assert isinstance(result, ScenarioResult)
        assert result.scenario_id == "t001"
        assert result.language == "pt"
        assert result.score > 0.0
        assert "lisboa" in result.matched_keywords
        mock_fn.assert_called_once_with("Qual é a capital de Portugal?")

    def test_english_question_selected(self, tiny_tester):
        mock_fn = MagicMock(return_value="Lisbon is the capital")
        scenario = tiny_tester.scenarios[0]

        result = tiny_tester.run_scenario(scenario, "en", mock_fn)

        assert result.language == "en"
        mock_fn.assert_called_once_with("What is the capital of Portugal?")

    def test_duration_is_positive(self, tiny_tester):
        mock_fn = MagicMock(return_value="some answer")
        result = tiny_tester.run_scenario(tiny_tester.scenarios[0], "pt", mock_fn)
        assert result.duration_seconds >= 0.0

    def test_empty_response_scores_zero(self, tiny_tester):
        mock_fn = MagicMock(return_value="")
        result = tiny_tester.run_scenario(tiny_tester.scenarios[0], "pt", mock_fn)
        assert result.score == 0.0
        assert result.matched_keywords == []


# ---------------------------------------------------------------------------
# ConversationTester — run_all
# ---------------------------------------------------------------------------

class TestRunAll:
    def test_run_all_default(self, tiny_tester):
        mock_fn = MagicMock(return_value="olá tudo bem lisboa jantar")
        results = tiny_tester.run_all(mock_fn, language="pt")
        assert len(results) == 3  # all three tiny scenarios

    def test_run_all_with_limit(self, tiny_tester):
        mock_fn = MagicMock(return_value="resposta qualquer")
        results = tiny_tester.run_all(mock_fn, language="pt", limit=1)
        assert len(results) == 1
        assert results[0].scenario_id == "t001"

    def test_limit_larger_than_scenarios(self, tiny_tester):
        mock_fn = MagicMock(return_value="")
        results = tiny_tester.run_all(mock_fn, language="pt", limit=100)
        assert len(results) == 3  # capped at actual number

    def test_each_result_is_scenario_result(self, tiny_tester):
        mock_fn = MagicMock(return_value="olá hello")
        results = tiny_tester.run_all(mock_fn, language="en")
        for r in results:
            assert isinstance(r, ScenarioResult)


# ---------------------------------------------------------------------------
# ConversationTester — summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_summary_structure(self, tiny_tester):
        mock_fn = MagicMock(return_value="lisboa hello jantar")
        results = tiny_tester.run_all(mock_fn, language="pt")
        summary = tiny_tester.summarize(results)

        assert "avg_score" in summary
        assert "total_scenarios" in summary
        assert "by_category" in summary
        assert summary["total_scenarios"] == 3

    def test_by_category_keys(self, tiny_tester):
        mock_fn = MagicMock(return_value="lisboa hello jantar")
        results = tiny_tester.run_all(mock_fn, language="pt")
        summary = tiny_tester.summarize(results)

        cats = summary["by_category"]
        assert "factual" in cats
        assert "conversational" in cats
        assert "member_knowledge" in cats

    def test_empty_results(self, tiny_tester):
        summary = tiny_tester.summarize([])
        assert summary["avg_score"] == 0.0
        assert summary["total_scenarios"] == 0
        assert summary["by_category"] == {}

    def test_category_count(self, tiny_tester):
        mock_fn = MagicMock(return_value="")
        results = tiny_tester.run_all(mock_fn, language="pt")
        summary = tiny_tester.summarize(results)

        for cat_info in summary["by_category"].values():
            assert "avg_score" in cat_info
            assert "count" in cat_info
            assert cat_info["count"] >= 1


# ---------------------------------------------------------------------------
# Default SCENARIOS list
# ---------------------------------------------------------------------------

class TestDefaultScenarios:
    def test_at_least_15_scenarios(self):
        assert len(SCENARIOS) >= 15

    def test_scenario_has_required_keys(self):
        required = {"id", "category", "question_pt", "question_en", "expected_keywords"}
        for s in SCENARIOS:
            assert required.issubset(s.keys()), f"Scenario {s.get('id')} missing keys"

    def test_unique_ids(self):
        ids = [s["id"] for s in SCENARIOS]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs"


# ---------------------------------------------------------------------------
# BenchmarkRunner — build_matrix
# ---------------------------------------------------------------------------

class TestBuildMatrix:
    def test_matrix_size_two_dimensions(self, base_config):
        runner = BenchmarkRunner(base_config)
        matrix = runner.build_matrix()
        # 2 knowledge_approaches × 2 languages = 4
        assert len(matrix) == 4

    def test_matrix_entries_are_benchmark_config(self, base_config):
        runner = BenchmarkRunner(base_config)
        for cfg in runner.build_matrix():
            assert isinstance(cfg, BenchmarkConfig)

    def test_no_dimensions_defaults(self, base_config):
        base_config["benchmark"]["dimensions"] = []
        runner = BenchmarkRunner(base_config)
        matrix = runner.build_matrix()
        # No dimensions active → single default config
        assert len(matrix) == 1

    def test_context_sizes_dimension(self, base_config):
        base_config["benchmark"]["dimensions"] = ["context_sizes"]
        runner = BenchmarkRunner(base_config)
        matrix = runner.build_matrix()
        # 2 context_sizes × 1 default for others = 2
        assert len(matrix) == 2

    def test_all_dimensions(self, base_config):
        base_config["benchmark"]["dimensions"] = [
            "knowledge_approaches",
            "languages",
            "context_sizes",
        ]
        runner = BenchmarkRunner(base_config)
        matrix = runner.build_matrix()
        # 2 × 2 × 2 = 8
        assert len(matrix) == 8


# ---------------------------------------------------------------------------
# BenchmarkRunner — run (dry-run)
# ---------------------------------------------------------------------------

class TestBenchmarkRun:
    def test_dry_run_returns_results(self, base_config):
        runner = BenchmarkRunner(base_config)
        results = runner.run()
        assert len(results) == 4  # matches matrix size
        for r in results:
            assert isinstance(r, BenchmarkResult)

    def test_scenarios_per_config_respected(self, base_config):
        base_config["benchmark"]["scenarios_per_config"] = 2
        runner = BenchmarkRunner(base_config)
        results = runner.run()
        for r in results:
            assert len(r.scenario_results) == 2

    def test_custom_response_fn_factory(self, base_config):
        def factory(cfg: BenchmarkConfig):
            return lambda q: f"mock answer for {cfg.language}"

        runner = BenchmarkRunner(base_config, response_fn_factory=factory)
        results = runner.run()
        assert len(results) > 0
        # At least one non-empty response
        some_response = results[0].scenario_results[0].response
        assert "mock answer" in some_response


# ---------------------------------------------------------------------------
# BenchmarkRunner — format_markdown
# ---------------------------------------------------------------------------

class TestFormatMarkdown:
    def test_contains_headers(self, base_config):
        runner = BenchmarkRunner(base_config)
        results = runner.run()
        md = runner.format_markdown(results)

        assert "Knowledge Approach" in md
        assert "Language" in md
        assert "Context" in md
        assert "Model" in md
        assert "Avg Score" in md
        assert "Scenarios" in md
        assert "Duration" in md

    def test_contains_table_separator(self, base_config):
        runner = BenchmarkRunner(base_config)
        results = runner.run()
        md = runner.format_markdown(results)
        assert "|---|" in md

    def test_contains_report_title(self, base_config):
        runner = BenchmarkRunner(base_config)
        results = runner.run()
        md = runner.format_markdown(results)
        assert "Benchmark Report" in md

    def test_empty_results(self, base_config):
        runner = BenchmarkRunner(base_config)
        md = runner.format_markdown([])
        assert "Benchmark Report" in md


# ---------------------------------------------------------------------------
# BenchmarkRunner — format_json
# ---------------------------------------------------------------------------

class TestFormatJSON:
    def test_json_serializable(self, base_config):
        runner = BenchmarkRunner(base_config)
        results = runner.run()
        data = runner.format_json(results)

        # Must not raise
        serialized = json.dumps(data, ensure_ascii=False)
        assert isinstance(serialized, str)

    def test_json_structure(self, base_config):
        runner = BenchmarkRunner(base_config)
        results = runner.run()
        data = runner.format_json(results)

        assert "metadata" in data
        assert "results" in data
        assert data["metadata"]["total_configs"] == len(results)

    def test_json_results_have_config(self, base_config):
        runner = BenchmarkRunner(base_config)
        results = runner.run()
        data = runner.format_json(results)

        for entry in data["results"]:
            assert "config" in entry
            assert "knowledge_approach" in entry["config"]


# ---------------------------------------------------------------------------
# BenchmarkRunner — save_reports
# ---------------------------------------------------------------------------

class TestSaveReports:
    def test_creates_files(self, base_config, tmp_path):
        base_config["benchmark"]["output_dir"] = str(tmp_path)
        runner = BenchmarkRunner(base_config)
        results = runner.run()

        md_path, json_path = runner.save_reports(results)

        assert md_path.exists()
        assert json_path.exists()
        assert md_path.suffix == ".md"
        assert json_path.suffix == ".json"

    def test_markdown_content(self, base_config, tmp_path):
        base_config["benchmark"]["output_dir"] = str(tmp_path)
        runner = BenchmarkRunner(base_config)
        results = runner.run()

        md_path, _ = runner.save_reports(results)
        content = md_path.read_text(encoding="utf-8")
        assert "Benchmark Report" in content

    def test_json_content_loadable(self, base_config, tmp_path):
        base_config["benchmark"]["output_dir"] = str(tmp_path)
        runner = BenchmarkRunner(base_config)
        results = runner.run()

        _, json_path = runner.save_reports(results)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "metadata" in data
        assert "results" in data

    def test_creates_output_dir(self, base_config, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        base_config["benchmark"]["output_dir"] = str(nested)
        runner = BenchmarkRunner(base_config)
        results = runner.run()

        md_path, json_path = runner.save_reports(results)
        assert nested.exists()
        assert md_path.exists()
