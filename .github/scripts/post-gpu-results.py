#!/usr/bin/env python3
"""Parse GPU pipeline output logs and format results as Markdown for PR comments/job summaries."""

import sys
import re
from pathlib import Path


def parse_training_log(log_content: str) -> dict:
    """Extract training metrics from ProgressCallback output."""
    metrics = {
        "started": None,
        "completed": None,
        "duration_minutes": None,
        "final_loss": None,
        "final_eval_loss": None,
        "final_step": None,
        "total_steps": None,
        "errors": [],
    }

    start_match = re.search(
        r"Training Started - (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log_content
    )
    if start_match:
        metrics["started"] = start_match.group(1)

    end_match = re.search(
        r"Training Completed - (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log_content
    )
    if end_match:
        metrics["completed"] = end_match.group(1)

    duration_match = re.search(r"Duration: ([\d.]+) minutes", log_content)
    if duration_match:
        metrics["duration_minutes"] = float(duration_match.group(1))

    # Extract all step entries and take the last one
    step_matches = re.findall(
        r"Step (\d+)/(\d+) \([\d.]+%\).*?Loss: ([\d.]+)", log_content, re.DOTALL
    )
    if step_matches:
        last = step_matches[-1]
        metrics["final_step"] = int(last[0])
        metrics["total_steps"] = int(last[1])
        metrics["final_loss"] = float(last[2])

    eval_matches = re.findall(r"Val Loss: ([\d.]+)", log_content)
    if eval_matches:
        metrics["final_eval_loss"] = float(eval_matches[-1])

    error_lines = re.findall(r"(?:Error|Exception|Traceback \(most recent)[^\n]*", log_content)
    metrics["errors"] = error_lines[:5]

    return metrics


def parse_test_log(log_content: str) -> dict:
    """Extract pytest results from test output."""
    metrics = {"passed": 0, "failed": 0, "errors": 0, "duration": None, "summary": ""}

    summary_match = re.search(
        r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) error(?:s)?)? in ([\d.]+)s",
        log_content,
    )
    if summary_match:
        metrics["passed"] = int(summary_match.group(1))
        metrics["failed"] = int(summary_match.group(2) or 0)
        metrics["errors"] = int(summary_match.group(3) or 0)
        metrics["duration"] = float(summary_match.group(4))
        metrics["summary"] = summary_match.group(0)

    return metrics


def format_training_results(metrics: dict, mode: str) -> str:
    lines = [f"## 🖥️ GPU Pipeline Results — `{mode}`\n"]

    if metrics.get("completed"):
        lines.append("**Status:** ✅ Completed\n")
    elif metrics.get("started"):
        lines.append("**Status:** ⚠️ Started but did not complete\n")
    else:
        lines.append("**Status:** ❌ Could not parse training output\n")

    if metrics.get("started"):
        lines.append(f"- **Started:** {metrics['started']}")
    if metrics.get("completed"):
        lines.append(f"- **Completed:** {metrics['completed']}")
    if metrics.get("duration_minutes") is not None:
        lines.append(f"- **Duration:** {metrics['duration_minutes']:.1f} min")
    if metrics.get("final_step") is not None:
        lines.append(
            f"- **Steps completed:** {metrics['final_step']}/{metrics['total_steps']}"
        )
    if metrics.get("final_loss") is not None:
        lines.append(f"- **Final training loss:** `{metrics['final_loss']:.4f}`")
    if metrics.get("final_eval_loss") is not None:
        lines.append(f"- **Final eval loss:** `{metrics['final_eval_loss']:.4f}`")

    if metrics.get("errors"):
        lines.append("\n**Errors detected:**")
        for err in metrics["errors"]:
            lines.append(f"- `{err[:120]}`")

    return "\n".join(lines)


def format_test_results(metrics: dict, mode: str) -> str:
    lines = [f"## 🖥️ GPU Pipeline Results — `{mode}`\n"]

    if metrics.get("summary"):
        status = "✅" if metrics["failed"] == 0 and metrics["errors"] == 0 else "❌"
        lines.append(f"**Status:** {status} `{metrics['summary']}`\n")
        lines.append(f"- **Passed:** {metrics['passed']}")
        if metrics["failed"]:
            lines.append(f"- **Failed:** {metrics['failed']}")
        if metrics["errors"]:
            lines.append(f"- **Errors:** {metrics['errors']}")
        if metrics["duration"] is not None:
            lines.append(f"- **Duration:** {metrics['duration']:.2f}s")
    else:
        lines.append("**Status:** ⚠️ Could not parse test output\n")

    return "\n".join(lines)


def parse_benchmark_log(log_content: str) -> dict:
    """Extract benchmark metrics from benchmark.py output."""
    metrics = {
        "configs": 0,
        "avg_score": None,
        "baseline_avg": None,
        "delta": None,
        "regression": False,
        "completed": False,
    }

    complete_match = re.search(r"Benchmark complete[^\n]*(\d+) configs", log_content)
    if complete_match:
        metrics["completed"] = True
        metrics["configs"] = int(complete_match.group(1))

    regression_match = re.search(
        r"BENCHMARK_REGRESSION:\s*avg_score=([\d.]+)\s+baseline=([\d.nan]+)\s+delta=([^\s]+)",
        log_content,
    )
    if regression_match:
        metrics["avg_score"] = float(regression_match.group(1))
        baseline_raw = regression_match.group(2)
        metrics["baseline_avg"] = float(baseline_raw) if baseline_raw != "none" else None
        delta_raw = regression_match.group(3).rstrip("⚠️ REGRESSION").strip()
        try:
            metrics["delta"] = float(delta_raw.rstrip("%").replace("+", ""))
        except ValueError:
            pass
        metrics["regression"] = "REGRESSION" in regression_match.group(0)

    return metrics


def format_benchmark_results(metrics: dict, mode: str) -> str:
    lines = [f"## 🖥️ GPU Pipeline Results — `{mode}`\n"]

    if metrics.get("completed"):
        lines.append("**Status:** ✅ Completed\n")
        if metrics.get("configs"):
            lines.append(f"- **Configs evaluated:** {metrics['configs']}")
        if metrics.get("avg_score") is not None:
            pct = f"{metrics['avg_score'] * 100:.2f}%" if metrics["avg_score"] <= 1 else f"{metrics['avg_score']:.2f}%"
            lines.append(f"- **Overall avg score:** `{pct}`")
            if metrics.get("baseline_avg") is not None:
                base_pct = f"{metrics['baseline_avg'] * 100:.2f}%" if metrics["baseline_avg"] <= 1 else f"{metrics['baseline_avg']:.2f}%"
                delta = metrics.get("delta", 0)
                delta_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
                regression_icon = " ⚠️ **REGRESSION**" if metrics.get("regression") else " ✅"
                lines.append(f"- **vs baseline:** `{base_pct}` → delta `{delta_str}`{regression_icon}")
    else:
        lines.append("**Status:** ⚠️ Could not parse benchmark output\n")

    return "\n".join(lines)


def parse_judge_log(log_content: str) -> dict:
    """Extract LLM judge evaluation metrics from conversation_tester.py output."""
    metrics = {
        "completed": False,
        "total_scenarios": None,
        "total_failures": None,
        "overall_avg": None,
        "factual_accuracy": None,
        "relevance": None,
        "language_quality": None,
        "tone": None,
    }

    if "Evaluation complete!" in log_content:
        metrics["completed"] = True

    m = re.search(r"Total scenarios\s*:\s*(\d+)", log_content)
    if m:
        metrics["total_scenarios"] = int(m.group(1))

    m = re.search(r"Total failures\s*:\s*(\d+)", log_content)
    if m:
        metrics["total_failures"] = int(m.group(1))

    m = re.search(r"Overall average\s*:\s*([\d.]+)", log_content)
    if m:
        metrics["overall_avg"] = float(m.group(1))

    m = re.search(r"Factual accuracy\s*:\s*([\d.]+)", log_content)
    if m:
        metrics["factual_accuracy"] = float(m.group(1))

    m = re.search(r"Relevance\s*:\s*([\d.]+)", log_content)
    if m:
        metrics["relevance"] = float(m.group(1))

    m = re.search(r"Language quality\s*:\s*([\d.]+)", log_content)
    if m:
        metrics["language_quality"] = float(m.group(1))

    m = re.search(r"Tone\s*:\s*([\d.]+)", log_content)
    if m:
        metrics["tone"] = float(m.group(1))

    return metrics


def format_judge_results(metrics: dict, mode: str) -> str:
    lines = [f"## 🖥️ GPU Pipeline Results — `{mode}`\n"]

    if metrics.get("completed"):
        failures = metrics.get("total_failures", 0) or 0
        total = metrics.get("total_scenarios", 0) or 0
        status = "✅" if failures == 0 else ("⚠️" if failures < total / 2 else "❌")
        lines.append(f"**Status:** {status} Evaluation complete\n")
        if total:
            lines.append(f"- **Scenarios:** {total}")
        if failures:
            lines.append(f"- **Failures:** {failures}")
        if metrics.get("overall_avg") is not None:
            lines.append(f"- **Overall avg score:** `{metrics['overall_avg']:.2f} / 5`")
        if metrics.get("factual_accuracy") is not None:
            lines.append(f"- **Factual accuracy:** `{metrics['factual_accuracy']:.2f}`")
        if metrics.get("relevance") is not None:
            lines.append(f"- **Relevance:** `{metrics['relevance']:.2f}`")
        if metrics.get("language_quality") is not None:
            lines.append(f"- **Language quality:** `{metrics['language_quality']:.2f}`")
        if metrics.get("tone") is not None:
            lines.append(f"- **Tone:** `{metrics['tone']:.2f}`")
    else:
        lines.append("**Status:** ⚠️ Could not parse judge-eval output\n")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: post-gpu-results.py <log_file> [mode]")
        sys.exit(1)

    log_file = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "finetune"

    if not log_file.exists():
        print(
            f"## 🖥️ GPU Pipeline Results — `{mode}`\n\n"
            f"⚠️ No output log found. The pipeline step may have been skipped or failed before producing output."
        )
        return

    log_content = log_file.read_text(errors="replace")

    if mode in ("evaluate", "inference-test"):
        metrics = parse_test_log(log_content)
        print(format_test_results(metrics, mode))
    elif mode == "benchmark":
        metrics = parse_benchmark_log(log_content)
        print(format_benchmark_results(metrics, mode))
    elif mode == "judge-eval":
        metrics = parse_judge_log(log_content)
        print(format_judge_results(metrics, mode))
    else:
        metrics = parse_training_log(log_content)
        print(format_training_results(metrics, mode))


if __name__ == "__main__":
    main()
