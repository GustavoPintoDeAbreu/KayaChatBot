"""
Benchmarking orchestrator for KayaChatBot.

Builds a matrix of RAG / model configurations, runs the conversation tester
against each one, and produces markdown + JSON reports.

Usage (dry-run):
    python src/testing/benchmark.py --dry-run

Usage (with custom config):
    python src/testing/benchmark.py --config config.yaml --scenarios 5
"""

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Ensure project root is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.testing.conversation_tester import (
    ConversationTester,
    ScenarioResult,
    LLMJudgeTester,
    GoldenTestRunner,
    load_provider,
    LocalModel,
)


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """A single benchmark configuration point in the test matrix."""

    knowledge_approach: str  # "both", "json_only", "chromadb_only", "none"
    language: str            # "pt" or "en"
    max_seq_length: int      # e.g. 2048, 4096, 8192
    top_k: int               # e.g. 3, 5, 10
    model_id: str            # model path or ID
    max_new_tokens: int = 256  # generation length to test


@dataclass
class BenchmarkResult:
    """Aggregated benchmark result for one configuration."""

    config: BenchmarkConfig
    scenario_results: List[ScenarioResult]
    avg_score: float
    duration_seconds: float
    timestamp: str  # ISO-8601
    tokens_per_second: Optional[float] = None  # throughput (None if not measured)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _noop_response_fn(question: str) -> str:
    """No-op response function used for dry-runs (returns empty string)."""
    return ""


class BenchmarkRunner:
    """Builds a configuration matrix, runs scenarios, and formats reports."""

    def __init__(
        self,
        config: dict,
        response_fn_factory: Optional[Callable[[BenchmarkConfig], Callable[[str], str]]] = None,
    ):
        """Initialise the benchmark runner.

        Args:
            config: The full ``config.yaml`` dict.
            response_fn_factory: Optional factory that, given a
                :class:`BenchmarkConfig`, returns a ``Callable[[str], str]``
                used to generate responses.  When *None* a no-op placeholder
                is used (returns empty string — useful for dry-runs).
        """
        self.config = config
        self.response_fn_factory = response_fn_factory
        self.tester = ConversationTester()

    # ------------------------------------------------------------------
    # Matrix building
    # ------------------------------------------------------------------

    def build_matrix(self) -> List[BenchmarkConfig]:
        """Build the Cartesian product of dimensions defined in config.

        Reads ``config["benchmark"]["dimensions"]`` to determine which axes
        to vary.  Returns a list of :class:`BenchmarkConfig` — one per
        unique combination.
        """
        bench = self.config.get("benchmark", {})
        dimensions = bench.get("dimensions", [])

        # Defaults (single-value lists so the product still works)
        knowledge_approaches = ["both"]
        languages = ["pt"]
        context_sizes = [
            {
                "max_seq_length": self.config.get("model", {}).get("max_seq_length", 4096),
                "top_k": self.config.get("rag", {}).get("top_k", 5),
            }
        ]
        models = [self.config.get("model", {}).get("model_id", "unknown")]

        # Override from config when dimension is active
        if "knowledge_approaches" in dimensions:
            knowledge_approaches = bench.get("knowledge_approaches", knowledge_approaches)
        if "languages" in dimensions:
            languages = bench.get("languages", languages)
        if "context_sizes" in dimensions:
            raw = bench.get("context_sizes", context_sizes)
            if raw:
                context_sizes = raw
        if "models" in dimensions:
            raw_models = bench.get("models", [])
            if raw_models:
                models = raw_models

        # max_new_tokens sweep: produce one config per value in the list
        default_mnt = self.config.get("inference", {}).get("max_new_tokens", 256)
        max_new_tokens_values = [default_mnt]
        if "max_new_tokens" in dimensions:
            raw_mnt = bench.get("max_new_tokens_values", [256, 384, 512])
            if raw_mnt:
                max_new_tokens_values = raw_mnt

        matrix: List[BenchmarkConfig] = []
        for ka, lang, ctx, model, mnt in itertools.product(
            knowledge_approaches, languages, context_sizes, models, max_new_tokens_values
        ):
            matrix.append(
                BenchmarkConfig(
                    knowledge_approach=ka,
                    language=lang,
                    max_seq_length=ctx.get("max_seq_length", 4096),
                    top_k=ctx.get("top_k", 5),
                    model_id=model,
                    max_new_tokens=mnt,
                )
            )

        return matrix

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self, matrix: Optional[List[BenchmarkConfig]] = None
    ) -> List[BenchmarkResult]:
        """Run the conversation tester for every configuration in *matrix*.

        Args:
            matrix: List of configs to evaluate.  Defaults to
                     :meth:`build_matrix` when *None*.

        Returns:
            List of :class:`BenchmarkResult` objects.
        """
        if matrix is None:
            matrix = self.build_matrix()

        bench = self.config.get("benchmark", {})
        limit = bench.get("scenarios_per_config", None)
        results: List[BenchmarkResult] = []

        for i, cfg in enumerate(matrix, 1):
            print(f"📊 Running config {i}/{len(matrix)}: "
                  f"{cfg.knowledge_approach} / {cfg.language} / "
                  f"seq={cfg.max_seq_length} top_k={cfg.top_k} max_new_tokens={cfg.max_new_tokens}")

            response_fn: Callable[[str], str]
            if self.response_fn_factory is not None:
                response_fn = self.response_fn_factory(cfg)
            else:
                response_fn = _noop_response_fn

            start = time.time()
            scenario_results = self.tester.run_all(
                response_fn, language=cfg.language, limit=limit
            )
            elapsed = round(time.time() - start, 3)

            summary = self.tester.summarize(scenario_results)

            results.append(
                BenchmarkResult(
                    config=cfg,
                    scenario_results=scenario_results,
                    avg_score=summary["avg_score"],
                    duration_seconds=elapsed,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

            print(f"  ✅ avg_score={summary['avg_score']:.2%}  "
                  f"scenarios={summary['total_scenarios']}  "
                  f"time={elapsed:.1f}s")

        return results

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_markdown(self, results: List[BenchmarkResult]) -> str:
        """Render results as a Markdown report with a summary table.

        Args:
            results: List of :class:`BenchmarkResult` objects.

        Returns:
            A Markdown string.
        """
        lines: List[str] = []
        lines.append("# 📊 KayaChatBot Benchmark Report\n")
        lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n")

        if results:
            avg_all = sum(r.avg_score for r in results) / len(results)
            lines.append(f"**Overall average score:** {avg_all:.2%}\n")
        lines.append("")

        # Table header
        lines.append("| Knowledge Approach | Language | Context | max_new_tokens | Model | Avg Score | Scenarios | Duration |")
        lines.append("|---|---|---|---|---|---|---|---|")

        for r in results:
            ctx = f"{r.config.max_seq_length} / top_k={r.config.top_k}"
            model = r.config.model_id
            # Truncate long model paths for readability
            if len(model) > 40:
                model = f"…{model[-37:]}"
            lines.append(
                f"| {r.config.knowledge_approach} "
                f"| {r.config.language} "
                f"| {ctx} "
                f"| {r.config.max_new_tokens} "
                f"| {model} "
                f"| {r.avg_score:.2%} "
                f"| {len(r.scenario_results)} "
                f"| {r.duration_seconds:.1f}s |"
            )

        lines.append("")
        return "\n".join(lines)

    def format_json(self, results: List[BenchmarkResult]) -> dict:
        """Serialise results to a plain dict (JSON-safe).

        Args:
            results: List of :class:`BenchmarkResult` objects.

        Returns:
            Dict with ``metadata`` and ``results`` keys.
        """
        return {
            "metadata": {
                "generated": datetime.now(timezone.utc).isoformat(),
                "total_configs": len(results),
                "overall_avg_score": (
                    round(sum(r.avg_score for r in results) / len(results), 4)
                    if results
                    else 0.0
                ),
            },
            "results": [
                {
                    "config": asdict(r.config),
                    "avg_score": r.avg_score,
                    "duration_seconds": r.duration_seconds,
                    "timestamp": r.timestamp,
                    "scenario_results": [asdict(sr) for sr in r.scenario_results],
                }
                for r in results
            ],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_reports(
        self, results: List[BenchmarkResult]
    ) -> Tuple[Path, Path]:
        """Write markdown and JSON reports to the configured output directory.

        Args:
            results: List of :class:`BenchmarkResult` objects.

        Returns:
            A tuple ``(markdown_path, json_path)``.
        """
        bench = self.config.get("benchmark", {})
        output_dir = Path(bench.get("output_dir", "reports/benchmarks"))
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        md_path = output_dir / f"benchmark_{ts}.md"
        md_path.write_text(self.format_markdown(results), encoding="utf-8")

        json_path = output_dir / f"benchmark_{ts}.json"
        json_path.write_text(
            json.dumps(self.format_json(results), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"✅ Markdown report saved to {md_path}")
        print(f"✅ JSON report saved to {json_path}")

        return md_path, json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    """Load a YAML config file and return as dict."""
    import yaml  # local import — only needed for CLI

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    """CLI entry-point for running benchmarks."""
    parser = argparse.ArgumentParser(
        description="KayaChatBot benchmark runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for reports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a no-op response function (no model loading)",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        default=None,
        help="Comma-separated list of dimensions to test "
             "(e.g. knowledge_approaches,languages)",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=None,
        help="Number of scenarios per configuration",
    )

    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Save current results as the new regression baseline",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        help="LLM judge provider for quality scoring: 'xai', 'azure', or 'azure_gpt53'. "
             "When set, runs LLMJudgeTester after keyword benchmark.",
    )
    parser.add_argument(
        "--golden-tests",
        type=str,
        default=None,
        help="Path to golden_test_conversations.json (default: auto-detect from config). "
             "Requires --judge-provider.",
    )
    parser.add_argument(
        "--emit-tasks",
        action="store_true",
        help="Write failing golden tests as tasks to data/benchmark_tasks.json "
             "(requires --golden-tests + --judge-provider).",
    )

    args = parser.parse_args()

    # Load config
    config = _load_config(args.config)

    # Apply CLI overrides
    config.setdefault("benchmark", {})

    if args.output_dir is not None:
        config["benchmark"]["output_dir"] = args.output_dir

    if args.dimensions is not None:
        config["benchmark"]["dimensions"] = [
            d.strip() for d in args.dimensions.split(",")
        ]

    if args.scenarios is not None:
        config["benchmark"]["scenarios_per_config"] = args.scenarios

    # Build runner — dry-run uses no factory (falls back to _noop_response_fn)
    factory = None  # extend later to build real model response functions
    runner = BenchmarkRunner(config, response_fn_factory=factory)

    print("📊 Building benchmark matrix …")
    matrix = runner.build_matrix()
    print(f"  ✅ {len(matrix)} configurations to test\n")

    results = runner.run(matrix)

    # --- Regression detection ---
    output_dir = Path(config["benchmark"].get("output_dir", "reports/benchmarks"))
    baseline_file = output_dir / "baseline.json"
    current_avg = sum(r.avg_score for r in results) / len(results) if results else 0.0

    baseline_avg = None
    if baseline_file.exists() and not args.update_baseline:
        try:
            import json as _json
            baseline_data = _json.loads(baseline_file.read_text(encoding="utf-8"))
            baseline_avg = baseline_data.get("overall_avg_score")
        except Exception:
            pass

    if baseline_avg is not None:
        delta = current_avg - baseline_avg
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        regression_flag = " ⚠️ REGRESSION" if delta < -0.05 else ""
        print(f"BENCHMARK_REGRESSION: avg_score={current_avg:.4f} baseline={baseline_avg:.4f} delta={delta_str}{regression_flag}")
    else:
        print(f"BENCHMARK_REGRESSION: avg_score={current_avg:.4f} baseline=none delta=none")

    # Save new baseline when requested or when none exists yet
    if args.update_baseline or not baseline_file.exists():
        try:
            import json as _json
            baseline_file.parent.mkdir(parents=True, exist_ok=True)
            baseline_file.write_text(
                _json.dumps({"overall_avg_score": round(current_avg, 4), "generated": datetime.now(timezone.utc).isoformat()},
                            indent=2),
                encoding="utf-8",
            )
            action = "updated" if args.update_baseline else "initialised"
            print(f"📄 Baseline {action}: {baseline_file}")
        except Exception as exc:
            print(f"⚠️ Could not save baseline: {exc}")

    md_path, json_path = runner.save_reports(results)

    print(f"\n📊 Benchmark complete — {len(results)} configs evaluated.")
    print(f"   Markdown: {md_path}")
    print(f"   JSON:     {json_path}")

    # ------------------------------------------------------------------
    # Golden test evaluation (optional — requires --judge-provider)
    # ------------------------------------------------------------------
    golden_failures = 0
    pending_tasks = []

    if args.judge_provider:
        golden_tests_file = (
            args.golden_tests
            or config.get("benchmark", {}).get(
                "golden_tests_file",
                "data/golden_test_conversations.json",
            )
        )
        print(f"\n🎯 Running golden tests with judge '{args.judge_provider}' …")
        try:
            judge_provider = load_provider(args.judge_provider, config)
            local_model = LocalModel(config)
            golden_runner = GoldenTestRunner(
                provider=judge_provider,
                local_model=local_model,
                config=config,
                golden_tests_file=golden_tests_file,
            )
            golden_report = golden_runner.run(verbose=True)
            golden_failures = golden_report["golden_tests_failed"]
            identity_failures = golden_report.get("identity_failures", 0)

            # Print machine-readable summary line (parsed by post-gpu-results.py)
            print(
                f"BENCHMARK_GOLDEN_RESULTS: "
                f"run={golden_report['golden_tests_run']} "
                f"passed={golden_report['golden_tests_passed']} "
                f"failed={golden_failures} "
                f"identity_failures={identity_failures}"
            )

            # Save golden report alongside regular reports
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            golden_path = Path(config.get("benchmark", {}).get("output_dir", "reports/benchmarks"))
            golden_path.mkdir(parents=True, exist_ok=True)
            golden_out = golden_path / f"golden_{ts}.json"
            golden_out.write_text(
                json.dumps(golden_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"✅ Golden test report saved to {golden_out}")

            # Build pending tasks for failures (for --emit-tasks)
            if args.emit_tasks:
                identity_fail_threshold = config.get("benchmark", {}).get(
                    "identity_failure_threshold", 3.0
                )
                for r in golden_report.get("failures", []):
                    category = r.get("category", "general")
                    score_info = ""
                    if r.get("scores"):
                        s = r["scores"]
                        score_info = (
                            f" identity_adherence={s.get('identity_adherence', '?'):.1f}"
                            f" factual_grounding={s.get('factual_grounding', '?'):.1f}"
                        ) if isinstance(s.get("identity_adherence"), float) else ""

                    leaked = r.get("identity_leak_patterns", [])
                    is_identity = bool(leaked) or category == "identity"
                    task = {
                        "title": (
                            f"[Golden] Identity leak in response to '{r['question'][:50]}'"
                            if is_identity
                            else f"[Golden] Low quality response to '{r['question'][:50]}'"
                        ),
                        "description": (
                            f"Golden test {r['test_id']} ({category}) failed.\n\n"
                            f"Question: {r['question']}\n\n"
                            f"Failure reasons: {'; '.join(r.get('failure_reasons', []))}\n\n"
                            f"Response excerpt: {r.get('response', '')[:300]}"
                            + (f"\n\nLeaked patterns: {leaked}" if leaked else "")
                            + score_info
                        ),
                        "type": "bug" if is_identity else "improvement",
                        "priority": "high" if is_identity else "medium",
                        "agent": "bug-fixer" if is_identity else "model-trainer",
                        "files_hint": [
                            "src/data/format_direct_training.py",
                            "src/data/generate_synthetic_data.py",
                            "config.yaml",
                        ] if is_identity else [
                            "src/data/generate_synthetic_data.py",
                            "config.yaml",
                        ],
                    }
                    pending_tasks.append(task)

        except Exception as exc:
            print(f"⚠️  Golden test runner failed: {exc}")
            import traceback; traceback.print_exc()

    print(f"BENCHMARK_FAILURES: {golden_failures}")

    # Write pending_tasks to data/benchmark_tasks.json (for GPU pipeline to pick up)
    if pending_tasks:
        tasks_out = Path("data/benchmark_tasks.json")
        existing = []
        if tasks_out.exists():
            try:
                existing = json.loads(tasks_out.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        merged = existing + pending_tasks
        tasks_out.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"📋 Wrote {len(pending_tasks)} pending task(s) to {tasks_out}")
        print(f"BENCHMARK_TASKS_WRITTEN: {len(pending_tasks)}")


if __name__ == "__main__":
    main()
