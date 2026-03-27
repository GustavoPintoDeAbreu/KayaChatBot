"""
Full Pipeline Runner
Orchestrates the complete data processing and training pipeline.

Supports two modes (controlled by pipeline.skip_synthetic in config.yaml):
  skip_synthetic: false  — full pipeline with API-based synthetic generation
  skip_synthetic: true   — direct pipeline, no API calls required
"""

import subprocess
import sys
import os
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
if os.path.exists('/app'):
    BASE_DIR = Path("/app")
    PYTHON = "python"
else:
    BASE_DIR = Path(__file__).parent
    # Prefer the project venv; fall back to whatever python is running this script
    _venv_python = BASE_DIR / "kaya_chatbot_env" / "bin" / "python"
    PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

CONFIG_PATH = BASE_DIR / "config.yaml"


def run_script(script_path: Path, description: str) -> bool:
    """Run a Python script and return True on success."""
    print("\n" + "=" * 60, flush=True)
    print(f"🚀 {description}", flush=True)
    print("=" * 60, flush=True)

    try:
        subprocess.run(
            [PYTHON, str(script_path)],
            cwd=str(BASE_DIR),
            check=True,
            capture_output=False,
            text=True,
        )
        print(f"✅ {description} - Complete!", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!", flush=True)
        print(f"Error: {e}", flush=True)
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} - Interrupted by user", flush=True)
        return False


def main():
    """Run the full pipeline."""

    print("=" * 60, flush=True)
    print("🎯 KAYA CHATBOT - FULL PIPELINE", flush=True)
    print("=" * 60, flush=True)

    # Load config to decide pipeline mode
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    skip_synthetic: bool = config.get('pipeline', {}).get('skip_synthetic', False)
    test_mode: bool = config.get('test_mode', {}).get('enabled', False)

    if test_mode:
        print("\n⚠️  TEST MODE ENABLED — reduced data & steps", flush=True)

    if skip_synthetic:
        print("\nMode: Direct (no API calls)  [pipeline.skip_synthetic = true]", flush=True)
        print("\nSteps:", flush=True)
        print("  1. Extract messages from WhatsApp + Instagram", flush=True)
        print("  2. Format direct training data from messages", flush=True)
        print("  3. Merge datasets and create train/val splits", flush=True)
        print("  4. Fine-tune the model", flush=True)
    else:
        print("\nMode: Full (with API synthetic generation)  [pipeline.skip_synthetic = false]", flush=True)
        print("\nSteps:", flush=True)
        print("  1. Extract messages from WhatsApp + Instagram", flush=True)
        print("  2. Generate Kaya-specific conversations (xAI/Azure API)", flush=True)
        print("  3. Download and prepare Portuguese dataset", flush=True)
        print("  4. Merge datasets and create train/val splits", flush=True)
        print("  5. Fine-tune the model", flush=True)

    print("\n🚀 Starting pipeline...", flush=True)

    # ------------------------------------------------------------------
    # Step 1 — Extract raw messages
    # ------------------------------------------------------------------
    if not run_script(
        BASE_DIR / "src/data/extract_all_messages.py",
        "Step 1: Extract Messages",
    ):
        print("\n❌ Pipeline failed at Step 1", flush=True)
        return

    if skip_synthetic:
        # ------------------------------------------------------------------
        # Step 2 (direct) — Format training data without API
        # ------------------------------------------------------------------
        if not run_script(
            BASE_DIR / "src/data/format_direct_training.py",
            "Step 2: Format Direct Training Data",
        ):
            print("\n❌ Pipeline failed at Step 2 (direct format)", flush=True)
            return

        # Step 3 → merge (no Portuguese data in this mode)
        step_merge = 3
        step_train = 4

    else:
        # ------------------------------------------------------------------
        # Step 2 (synthetic) — Generate Kaya conversations via LLM API
        # ------------------------------------------------------------------
        print("\n💰 Note: This step will call the xAI / Azure API", flush=True)
        if not run_script(
            BASE_DIR / "src/data/generate_synthetic_data.py",
            "Step 2: Generate Kaya Conversations (API)",
        ):
            print("\n❌ Pipeline failed at Step 2", flush=True)
            return

        # ------------------------------------------------------------------
        # Step 3 — Prepare Portuguese dataset
        # ------------------------------------------------------------------
        if not run_script(
            BASE_DIR / "src/data/prepare_portuguese_data.py",
            "Step 3: Prepare Portuguese Dataset",
        ):
            print("\n❌ Pipeline failed at Step 3", flush=True)
            return

        step_merge = 4
        step_train = 5

    # ------------------------------------------------------------------
    # Merge step
    # ------------------------------------------------------------------
    if not run_script(
        BASE_DIR / "src/data/merge_datasets.py",
        f"Step {step_merge}: Merge Datasets",
    ):
        print(f"\n❌ Pipeline failed at Step {step_merge}", flush=True)
        return

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    print("\n🚀 Starting model training...", flush=True)
    if not run_script(
        BASE_DIR / "src/finetuning/train.py",
        f"Step {step_train}: Model Training",
    ):
        print(f"\n❌ Pipeline failed at Step {step_train}", flush=True)
        return

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("🎉 FULL PIPELINE COMPLETE!", flush=True)
    print("=" * 60, flush=True)
    print("\n✅ Pipeline finished successfully!", flush=True)
    print("\nGenerated files:", flush=True)
    print("  📄 data/all_messages_cleaned.jsonl", flush=True)
    print("  📄 data/finetune_chunks.jsonl", flush=True)
    print("  📄 data/synthetic_kaya.jsonl", flush=True)
    print("  📄 data/train_synthetic.jsonl", flush=True)
    print("  📄 data/val_synthetic.jsonl", flush=True)
    print("  🤖 models/kaya_v2_synthetic/ (fine-tuned model)", flush=True)


if __name__ == "__main__":
    main()
