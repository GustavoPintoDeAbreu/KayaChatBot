#!/usr/bin/env python3
"""Run the golden regression suite against the current model and print a scorecard.

A lightweight, repeatable alternative to the full `benchmark.py` config sweep: it
loads the live engine once (same model the app serves), sends every case in
data/golden_test_conversations.json, scores via the configured LLM judge, and saves
a timestamped report under reports/benchmarks/. Use it to track whether prompt /
training changes improve or regress the curated cases (incl. the ones derived from
real logged interactions).

    # needs the GPU free (stop prod first) + the judge provider key in .env
    kaya_chatbot_env/bin/python scripts/run_golden.py
    kaya_chatbot_env/bin/python scripts/run_golden.py --judge xai
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.chat.engine import get_engine, build_system_prompt
from src.testing.conversation_tester import GoldenTestRunner, load_provider


def main() -> None:
    ap = argparse.ArgumentParser(description="Run golden regression tests on the current model.")
    ap.add_argument("--judge", default=None, help="Judge provider (default: benchmark.judge_provider in config).")
    ap.add_argument("--golden-tests", default=None, help="Override golden tests file path.")
    ap.add_argument("--speaker", default="Gustavo", help="Speaker label for the model turn.")
    ap.add_argument("--model-dir", default=None, help="Override the model/adapter directory to evaluate.")
    args = ap.parse_args()

    config_path = str(Path(__file__).parent.parent / "config.yaml")
    config = load_config(config_path)

    if args.model_dir:
        config["training"]["output_dir"] = args.model_dir

    judge_name = args.judge or config.get("benchmark", {}).get("judge_provider", "xai")
    golden_file = args.golden_tests or config.get("benchmark", {}).get("golden_tests_file")

    print(f"Loading engine ({config['training']['output_dir']}) …")
    engine = get_engine(config)
    system_prompt = build_system_prompt(
        config, config_path, include_uncensored=config.get("chat", {}).get("uncensored_mode", False)
    )

    def response_fn(question: str, history=None) -> str:
        # ``history`` (optional, from multi-thread test cases) is a list of prior
        # "<who>: <text>" turns fed as recent conversation context.
        return engine.generate_reply(
            question, speaker=args.speaker, recent_lines=history or [], system_prompt=system_prompt
        )

    print(f"Loading judge provider '{judge_name}' …")
    provider = load_provider(judge_name, config)

    runner = GoldenTestRunner(
        provider, config=config, golden_tests_file=golden_file, response_fn=response_fn
    )
    print(f"Running {len(runner.test_cases)} golden tests …\n")
    report = runner.run(verbose=True)

    out_dir = Path(config.get("benchmark", {}).get("output_dir", "reports/benchmarks/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"golden_{stamp}.json"
    report["_meta"] = {
        "timestamp": stamp,
        "model_dir": config["training"]["output_dir"],
        "judge": judge_name,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    run = report.get("golden_tests_run", 0)
    failed = report.get("golden_tests_failed", 0)
    print("\n" + "=" * 56)
    print(f"GOLDEN: {run - failed}/{run} passed  ({failed} failed)  judge={judge_name}")
    print(f"report saved → {out_path}")
    print("=" * 56)


if __name__ == "__main__":
    main()
