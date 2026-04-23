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
    scoring_mode: str = "keyword"  # "keyword" or "llm_judge"
    golden_passed: Optional[int] = None   # set in llm_judge mode
    golden_total: Optional[int] = None    # set in llm_judge mode


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
        llm_bench_provider=None,
        golden_tests_file: Optional[str] = None,
    ):
        """Initialise the benchmark runner.

        Args:
            config: The full ``config.yaml`` dict.
            response_fn_factory: Optional factory that, given a
                :class:`BenchmarkConfig`, returns a ``Callable[[str], str]``
                used to generate responses.  When *None* a no-op placeholder
                is used (returns empty string — useful for dry-runs).
            llm_bench_provider: When set, use LLM judge (GoldenTestRunner) for
                per-config scoring instead of keyword matching.  Pass an
                already-loaded provider object.
            golden_tests_file: Override path to golden_test_conversations.json.
                Relevant when ``llm_bench_provider`` is set.
        """
        self.config = config
        self.response_fn_factory = response_fn_factory
        self.llm_bench_provider = llm_bench_provider
        self.golden_tests_file = golden_tests_file
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

        # When llm_bench_provider is set, load a single GoldenTestRunner shared
        # across all configs (model stays in VRAM; provider is reused per call).
        golden_runner = None
        if self.llm_bench_provider is not None:
            gt_file = self.golden_tests_file or str(
                Path(__file__).resolve().parent.parent.parent
                / "data" / "golden_test_conversations.json"
            )
            golden_runner = GoldenTestRunner(
                provider=self.llm_bench_provider,
                config=self.config,
                golden_tests_file=gt_file,
            )

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

            if golden_runner is not None:
                # LLM judge mode: score via GoldenTestRunner per config
                report = golden_runner.run(response_fn=response_fn, verbose=False)
                elapsed = round(time.time() - start, 3)
                passed = report["golden_tests_passed"]
                total = report["golden_tests_run"]
                avg_score = round(passed / total, 4) if total > 0 else 0.0
                results.append(
                    BenchmarkResult(
                        config=cfg,
                        scenario_results=[],
                        avg_score=avg_score,
                        duration_seconds=elapsed,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        scoring_mode="llm_judge",
                        golden_passed=passed,
                        golden_total=total,
                    )
                )
                print(f"  ✅ avg_score={avg_score:.2%}  "
                      f"golden={passed}/{total}  time={elapsed:.1f}s")
            else:
                # Keyword mode (default)
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
            scoring_note = "llm_judge" if any(r.scoring_mode == "llm_judge" for r in results) else "keyword"
            lines.append(f"**Overall average score:** {avg_all:.2%}  (**scoring: {scoring_note}**)\n")
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
            if r.scoring_mode == "llm_judge":
                scenarios_col = f"{r.golden_passed}/{r.golden_total} ★"
            else:
                scenarios_col = str(len(r.scenario_results))
            lines.append(
                f"| {r.config.knowledge_approach} "
                f"| {r.config.language} "
                f"| {ctx} "
                f"| {r.config.max_new_tokens} "
                f"| {model} "
                f"| {r.avg_score:.2%} "
                f"| {scenarios_col} "
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
                "scoring_mode": results[0].scoring_mode if results else "keyword",
            },
            "results": [
                {
                    "config": asdict(r.config),
                    "avg_score": r.avg_score,
                    "duration_seconds": r.duration_seconds,
                    "timestamp": r.timestamp,
                    "scoring_mode": r.scoring_mode,
                    "golden_passed": r.golden_passed,
                    "golden_total": r.golden_total,
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
# RAG-aware response factory
# ---------------------------------------------------------------------------

def build_local_rag_factory(
    config: dict,
) -> Optional[Callable[[BenchmarkConfig], Callable[[str], str]]]:
    """Return a factory that creates RAG-aware response functions backed by the local model.

    The model and retriever are loaded once and shared across all configs.
    Each :class:`BenchmarkConfig` gets its own ``response_fn`` that applies the
    appropriate ``knowledge_approach`` and ``top_k`` to the retriever before
    building the prompt.

    Returns *None* when the model directory does not exist (triggers noop fallback).
    """
    import json as _json
    import torch

    from pathlib import Path as _Path

    model_dir = config.get("training", {}).get("output_dir", "")
    if not model_dir or not _Path(model_dir).exists():
        print(f"⚠️  Model not found at '{model_dir}'; falling back to noop responses.")
        return None

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel  # type: ignore[import]
    except ImportError as exc:
        print(f"⚠️  Cannot import transformers/peft: {exc}; falling back to noop.")
        return None

    # -----------------------------------------------------------------
    # Load model + tokenizer once
    # -----------------------------------------------------------------
    adapter_cfg_path = _Path(model_dir) / "adapter_config.json"
    if not adapter_cfg_path.exists():
        print(f"⚠️  adapter_config.json not found in {model_dir}; falling back to noop.")
        return None

    adapter_cfg = _json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
    base_model_name = adapter_cfg.get("base_model_name_or_path", "")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print(f"📦 Loading base model '{base_model_name}' for benchmark …")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    print("✅ Benchmark model loaded")

    # -----------------------------------------------------------------
    # Build retriever (shared, knowledge_approach varies per response_fn)
    # -----------------------------------------------------------------
    retriever = None
    try:
        from src.chat.retriever import get_retriever as _get_retriever
        retriever = _get_retriever(config)
        print("✅ Benchmark retriever loaded")
    except Exception as exc:
        print(f"⚠️  Retriever unavailable ({exc}); running without RAG context.")

    # -----------------------------------------------------------------
    # Pre-build member-profile lines for JSON injection
    # -----------------------------------------------------------------
    base_system_prompt = config.get("data", {}).get("system_prompt", "")
    members_file = config.get("data", {}).get("group_members_file", "")
    member_lines: List[str] = []
    if members_file:
        mf = _Path(members_file)
        if not mf.is_absolute():
            mf = _Path("config.yaml").resolve().parent / members_file
        if mf.exists():
            members_data = _json.loads(mf.read_text(encoding="utf-8"))
            for m in members_data.get("members", []):
                line: str = m["name"]
                aliases = [a for a in m.get("aliases", []) if a.lower() != m["name"].lower()]
                if aliases:
                    line += f" (também conhecido como: {', '.join(aliases)})"
                notes = m.get("notes", "")
                key_facts = m.get("key_facts", [])
                if key_facts:
                    line += f" — {'. '.join(key_facts)}."
                elif notes:
                    sentences = [s.strip() for s in notes.split(".") if s.strip()]
                    line += f" — {'. '.join(sentences[:3])}."
                member_lines.append(line)

    inf_config = config.get("inference", {})

    def factory(cfg: BenchmarkConfig) -> Callable[[str], str]:
        """Create a response function for the given BenchmarkConfig."""
        knowledge_approach = cfg.knowledge_approach
        top_k = cfg.top_k

        system_prompt = base_system_prompt
        if member_lines and knowledge_approach in ("both", "json_only"):
            system_prompt += f"\n\nMembros do grupo Kaya: {'; '.join(member_lines)}."

        def response_fn(question: str) -> str:
            context = ""
            if retriever and knowledge_approach != "none":
                try:
                    # Override top_k on retriever for this config
                    orig_top_k = config.get("rag", {}).get("top_k", 5)
                    config.setdefault("rag", {})["top_k"] = top_k
                    context = retriever.retrieve_all(question, knowledge_approach=knowledge_approach)
                    config["rag"]["top_k"] = orig_top_k  # restore
                except Exception:
                    context = ""

            message_parts = []
            if context:
                message_parts.append(context)
            message_parts.append(question)
            user_message = "\n\n".join(message_parts)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=inf_config.get("temperature", 0.7),
                    top_p=inf_config.get("top_p", 0.9),
                    repetition_penalty=inf_config.get("repetition_penalty", 1.1),
                    do_sample=True,
                    use_cache=True,
                )
            full = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            # Strip the input portion
            prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            if prompt_text and full.startswith(prompt_text):
                return full[len(prompt_text):].strip()
            return full.strip()

        return response_fn

    return factory


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(path: str, profile: Optional[str] = None) -> dict:
    """Load config.yaml, applying the given model profile override if provided."""
    from src.config_loader import load_config as _lc
    return _lc(path, profile_override=profile)


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
        "--profile",
        type=str,
        default=None,
        help="Model profile to use (e.g. 'gemma4-e4b', 'qwen3-14b'). "
             "Overrides active_model_profile in config.yaml.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Override the model directory (training.output_dir). Useful for benchmarking "
             "a specific checkpoint without editing config.yaml.",
    )
    parser.add_argument(
        "--llm-bench",
        action="store_true",
        help="Use LLM judge (GoldenTestRunner) for per-config scoring instead of keyword "
             "matching. Requires --llm-bench-provider.",
    )
    parser.add_argument(
        "--llm-bench-provider",
        type=str,
        default=None,
        help="LLM provider for --llm-bench mode: 'xai', 'azure', or 'azure_gpt53'.",
    )

    args = parser.parse_args()

    # Load config (with optional profile merge)
    config = _load_config(args.config, profile=args.profile)

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

    # Apply model dir override
    if args.model_dir is not None:
        config.setdefault("training", {})["output_dir"] = args.model_dir

    # Build runner — dry-run uses no factory (falls back to _noop_response_fn)
    if args.dry_run:
        factory = None
    else:
        factory = build_local_rag_factory(config)

    # Optional LLM bench provider
    llm_bench_provider = None
    if args.llm_bench:
        if not args.llm_bench_provider:
            parser.error("--llm-bench requires --llm-bench-provider (e.g. --llm-bench-provider xai)")
        llm_bench_provider = load_provider(args.llm_bench_provider, config)

    runner = BenchmarkRunner(
        config,
        response_fn_factory=factory,
        llm_bench_provider=llm_bench_provider,
        golden_tests_file=args.golden_tests,
    )

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
        scoring_mode_str = results[0].scoring_mode if results else "keyword"
        print(f"BENCHMARK_REGRESSION: avg_score={current_avg:.4f} baseline={baseline_avg:.4f} delta={delta_str}{regression_flag} scoring={scoring_mode_str}")
    else:
        scoring_mode_str = results[0].scoring_mode if results else "keyword"
        print(f"BENCHMARK_REGRESSION: avg_score={current_avg:.4f} baseline=none delta=none scoring={scoring_mode_str}")

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
            # Reuse the already-loaded benchmark model via factory to avoid double GPU load
            golden_response_fn = None
            if factory is not None:
                max_seq = config.get("model", {}).get("max_seq_length", 2048)
                model_id = config.get("training", {}).get("output_dir", "")
                golden_cfg = BenchmarkConfig(
                    knowledge_approach="json_only",
                    language="pt",
                    max_seq_length=max_seq,
                    top_k=5,
                    model_id=model_id,
                    max_new_tokens=256,
                )
                golden_response_fn = factory(golden_cfg)
            golden_runner = GoldenTestRunner(
                provider=judge_provider,
                config=config,
                golden_tests_file=golden_tests_file,
                response_fn=golden_response_fn,
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

        except Exception as exc:
            print(f"⚠️  Golden test runner failed: {exc}")
            import traceback; traceback.print_exc()

    print(f"BENCHMARK_FAILURES: {golden_failures}")


if __name__ == "__main__":
    main()
