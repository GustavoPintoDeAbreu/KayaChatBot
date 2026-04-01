"""
Unit tests for src/testing/benchmark.py

Covers:
  - Config matrix generation: correct number of combinations
  - Config matrix with a single dimension / single value
  - Empty dimensions returns a single empty-config
  - Benchmark runner calls run_fn for each config and aggregates results
  - Best-config detection based on average_score
  - Markdown table format: well-formed headers, separator row, data rows
  - JSON report is valid and parseable
  - Minimal matrix (1 config × 1 scenario)
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.testing.benchmark import (
    format_json_report,
    format_markdown_table,
    generate_config_matrix,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# test_generate_config_matrix_*
# ---------------------------------------------------------------------------

class TestGenerateConfigMatrix:
    def test_single_dimension_single_value(self):
        matrix = generate_config_matrix({"approach": ["both"]})
        assert matrix == [{"approach": "both"}]

    def test_single_dimension_multiple_values(self):
        matrix = generate_config_matrix({"approach": ["both", "json_only", "none"]})
        assert len(matrix) == 3
        approaches = [c["approach"] for c in matrix]
        assert "both" in approaches
        assert "json_only" in approaches
        assert "none" in approaches

    def test_two_dimensions_cartesian_product(self):
        matrix = generate_config_matrix({
            "approach": ["both", "none"],
            "top_k": [3, 5],
        })
        assert len(matrix) == 4

    def test_three_dimensions_correct_count(self):
        matrix = generate_config_matrix({
            "approach": ["both", "none"],
            "top_k": [3, 5],
            "temperature": [0.7, 0.9, 1.0],
        })
        assert len(matrix) == 2 * 2 * 3

    def test_each_combination_is_unique(self):
        matrix = generate_config_matrix({
            "a": [1, 2],
            "b": ["x", "y"],
        })
        tuples = [tuple(sorted(c.items())) for c in matrix]
        assert len(tuples) == len(set(tuples))

    def test_all_keys_present_in_each_config(self):
        matrix = generate_config_matrix({"k1": [1], "k2": [2], "k3": [3]})
        for cfg in matrix:
            assert "k1" in cfg
            assert "k2" in cfg
            assert "k3" in cfg

    def test_empty_dimensions_returns_one_empty_config(self):
        matrix = generate_config_matrix({})
        assert matrix == [{}]

    def test_minimal_1x1_matrix(self):
        matrix = generate_config_matrix({"mode": ["test"]})
        assert len(matrix) == 1
        assert matrix[0] == {"mode": "test"}


# ---------------------------------------------------------------------------
# test_run_benchmark_*
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def _make_run_fn(self, scores: dict):
        """Return a run_fn that looks up average_score from the scores dict."""
        def run_fn(cfg):
            key = frozenset(cfg.items())
            return {"average_score": scores.get(key, 0.0), "pass_rate": 0.5}
        return run_fn

    def test_run_benchmark_calls_run_fn_for_each_config(self):
        configs = generate_config_matrix({"a": [1, 2, 3]})
        run_fn = MagicMock(return_value={"average_score": 5.0})
        run_benchmark(configs, run_fn)
        assert run_fn.call_count == 3

    def test_run_benchmark_configs_tested_count(self):
        configs = generate_config_matrix({"x": [1, 2]})
        run_fn = MagicMock(return_value={"average_score": 7.0})
        result = run_benchmark(configs, run_fn)
        assert result["configs_tested"] == 2

    def test_run_benchmark_results_list_length(self):
        configs = generate_config_matrix({"x": [1, 2, 3]})
        run_fn = MagicMock(return_value={"average_score": 5.0})
        result = run_benchmark(configs, run_fn)
        assert len(result["results"]) == 3

    def test_run_benchmark_best_config_has_highest_score(self):
        configs = [
            {"approach": "both"},
            {"approach": "none"},
            {"approach": "json_only"},
        ]
        scores = {
            frozenset({"approach": "both"}.items()): 8.5,
            frozenset({"approach": "none"}.items()): 4.0,
            frozenset({"approach": "json_only"}.items()): 6.0,
        }
        result = run_benchmark(configs, self._make_run_fn(scores))
        assert result["best_config"] == {"approach": "both"}

    def test_run_benchmark_minimal_1x1(self):
        configs = generate_config_matrix({"mode": ["test"]})
        run_fn = MagicMock(return_value={"average_score": 7.0})
        result = run_benchmark(configs, run_fn)
        assert result["configs_tested"] == 1
        assert result["best_config"] == {"mode": "test"}

    def test_run_benchmark_result_has_required_keys(self):
        configs = generate_config_matrix({"k": ["v"]})
        run_fn = MagicMock(return_value={"average_score": 5.0})
        result = run_benchmark(configs, run_fn)
        for key in ("configs_tested", "results", "best_config"):
            assert key in result

    def test_run_benchmark_each_result_has_config_and_metrics(self):
        configs = [{"approach": "both"}]
        run_fn = MagicMock(return_value={"average_score": 6.0, "pass_rate": 0.8})
        result = run_benchmark(configs, run_fn)
        entry = result["results"][0]
        assert "config" in entry
        assert "metrics" in entry

    def test_run_benchmark_empty_matrix(self):
        result = run_benchmark([], lambda cfg: {"average_score": 0.0})
        assert result["configs_tested"] == 0
        assert result["best_config"] is None


# ---------------------------------------------------------------------------
# test_format_markdown_table_*
# ---------------------------------------------------------------------------

class TestFormatMarkdownTable:
    def _build_output(self, configs, scores):
        """Build a benchmark output dict from configs and scores list."""
        results = [
            {"config": cfg, "metrics": {"average_score": s}}
            for cfg, s in zip(configs, scores)
        ]
        return {"configs_tested": len(configs), "results": results, "best_config": None}

    def test_markdown_table_has_header_row(self):
        output = self._build_output([{"approach": "both"}], [7.0])
        table = format_markdown_table(output)
        lines = table.strip().split("\n")
        assert lines[0].startswith("|")

    def test_markdown_table_has_separator_row(self):
        output = self._build_output([{"approach": "both"}], [7.0])
        table = format_markdown_table(output)
        lines = table.strip().split("\n")
        assert "---" in lines[1]

    def test_markdown_table_has_data_rows(self):
        output = self._build_output(
            [{"approach": "both"}, {"approach": "none"}],
            [8.0, 4.0],
        )
        table = format_markdown_table(output)
        lines = table.strip().split("\n")
        # header + separator + 2 data rows = 4 lines
        assert len(lines) >= 4

    def test_markdown_table_config_keys_in_header(self):
        output = self._build_output([{"approach": "both", "top_k": 5}], [7.0])
        table = format_markdown_table(output)
        assert "approach" in table
        assert "top_k" in table

    def test_markdown_table_metric_keys_in_header(self):
        output = self._build_output([{"approach": "both"}], [7.5])
        table = format_markdown_table(output)
        assert "average_score" in table

    def test_markdown_table_empty_results_returns_placeholder(self):
        output = {"configs_tested": 0, "results": [], "best_config": None}
        table = format_markdown_table(output)
        assert "|" in table  # still a table shape

    def test_markdown_table_all_rows_start_with_pipe(self):
        output = self._build_output(
            [{"approach": "both"}, {"approach": "none"}],
            [7.0, 4.0],
        )
        table = format_markdown_table(output)
        for line in table.strip().split("\n"):
            assert line.startswith("|"), f"Row does not start with '|': {line}"


# ---------------------------------------------------------------------------
# test_format_json_report_*
# ---------------------------------------------------------------------------

class TestFormatJsonReport:
    def _simple_output(self):
        return {
            "configs_tested": 2,
            "results": [
                {"config": {"approach": "both"}, "metrics": {"average_score": 7.0}},
                {"config": {"approach": "none"}, "metrics": {"average_score": 4.0}},
            ],
            "best_config": {"approach": "both"},
        }

    def test_format_json_report_is_parseable(self):
        output = self._simple_output()
        serialized = format_json_report(output)
        loaded = json.loads(serialized)
        assert loaded["configs_tested"] == 2

    def test_format_json_report_preserves_best_config(self):
        output = self._simple_output()
        loaded = json.loads(format_json_report(output))
        assert loaded["best_config"] == {"approach": "both"}

    def test_format_json_report_pretty_printed(self):
        output = self._simple_output()
        serialized = format_json_report(output)
        assert "\n" in serialized  # indented = multi-line

    def test_format_json_report_empty_output(self):
        output = {"configs_tested": 0, "results": [], "best_config": None}
        loaded = json.loads(format_json_report(output))
        assert loaded["configs_tested"] == 0
        assert loaded["best_config"] is None
