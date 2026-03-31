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
    else:
        metrics = parse_training_log(log_content)
        print(format_training_results(metrics, mode))


if __name__ == "__main__":
    main()
