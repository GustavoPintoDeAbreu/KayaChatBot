"""
Benchmarking utilities for RAG configuration evaluation.

Provides helpers to generate a configuration matrix from a set of dimensions,
run a benchmark over that matrix, and format results as markdown or JSON.
"""

import itertools
import json
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Config matrix generation
# ---------------------------------------------------------------------------

def generate_config_matrix(dimensions: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate the Cartesian product of configuration dimensions.

    Parameters
    ----------
    dimensions:
        Dict mapping parameter names to lists of candidate values, e.g.::

            {
                "knowledge_approach": ["both", "json_only", "none"],
                "top_k": [3, 5],
            }

    Returns
    -------
    List of dicts, one per unique combination of parameter values.

    Examples
    --------
    >>> generate_config_matrix({"a": [1, 2], "b": ["x"]})
    [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'x'}]
    """
    if not dimensions:
        return [{}]

    keys = list(dimensions.keys())
    value_lists = [dimensions[k] for k in keys]

    configs: List[Dict[str, Any]] = []
    for combination in itertools.product(*value_lists):
        configs.append(dict(zip(keys, combination)))

    return configs


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    config_matrix: List[Dict[str, Any]],
    run_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    scenarios: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Run *run_fn* for each configuration in *config_matrix*.

    Parameters
    ----------
    config_matrix:
        List of config dicts produced by :func:`generate_config_matrix`.
    run_fn:
        Callable that accepts a config dict and returns a results dict with at
        least an ``"average_score"`` key.
    scenarios:
        Optional list of evaluation scenarios passed to ``run_fn`` (unused by
        default; subclasses / callers may use it via closures).

    Returns
    -------
    Dict with:
    - ``"configs_tested"`` — int
    - ``"results"`` — list of ``{"config": ..., "metrics": ...}`` dicts
    - ``"best_config"`` — config with the highest ``average_score``
    """
    results = []
    for cfg in config_matrix:
        metrics = run_fn(cfg)
        results.append({"config": cfg, "metrics": metrics})

    best = max(results, key=lambda r: r["metrics"].get("average_score", 0.0)) if results else None

    return {
        "configs_tested": len(config_matrix),
        "results": results,
        "best_config": best["config"] if best else None,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_markdown_table(benchmark_output: Dict[str, Any]) -> str:
    """
    Format benchmark results as a Markdown table.

    Parameters
    ----------
    benchmark_output:
        Dict returned by :func:`run_benchmark`.

    Returns
    -------
    Multi-line Markdown string.
    """
    results = benchmark_output.get("results", [])
    if not results:
        return "| Config | average_score |\n|--------|---------------|\n| — | — |"

    # Collect all config keys and metric keys
    all_config_keys = sorted({k for r in results for k in r["config"]})
    all_metric_keys = sorted({k for r in results for k in r["metrics"]})

    header_cells = all_config_keys + all_metric_keys
    header = "| " + " | ".join(header_cells) + " |"
    separator = "| " + " | ".join(["---"] * len(header_cells)) + " |"

    rows = [header, separator]
    for r in results:
        cells = []
        for k in all_config_keys:
            cells.append(str(r["config"].get(k, "")))
        for k in all_metric_keys:
            val = r["metrics"].get(k, "")
            cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def format_json_report(benchmark_output: Dict[str, Any]) -> str:
    """Return a pretty-printed JSON string of the benchmark output."""
    return json.dumps(benchmark_output, indent=2, ensure_ascii=False)
