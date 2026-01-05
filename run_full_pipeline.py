"""
Full Pipeline Runner (Docker Version)
Orchestrates the complete synthetic data generation and training pipeline.
"""

import subprocess
import sys
from pathlib import Path

# Python executable (Docker environment)
PYTHON = "python"
BASE_DIR = Path("/app")


def run_script(script_path: Path, description: str):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 60, flush=True)
    print(f"🚀 {description}", flush=True)
    print("=" * 60, flush=True)

    try:
        # Run with explicit stdout/stderr handling
        result = subprocess.run(
            [PYTHON, str(script_path)],
            cwd=str(BASE_DIR),
            check=True,
            capture_output=False,
            text=True
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
    print("🎯 KAYA CHATBOT - FULL SYNTHETIC DATA PIPELINE", flush=True)
    print("=" * 60, flush=True)
    print("\nThis will run the complete pipeline:", flush=True)
    print("  1. Extract messages from WhatsApp + Instagram", flush=True)
    print("  2. Generate Kaya-specific conversations (xAI Grok)", flush=True)
    print("  3. Download and prepare Portuguese dataset", flush=True)
    print("  4. Merge datasets and create train/val splits", flush=True)
    print("  5. Fine-tune the model", flush=True)

    # Check TEST_MODE warning
    print("\n⚠️  Check test_mode setting in config.yaml:", flush=True)
    print("   - test_mode.enabled: true  → Quick test (minimal data)", flush=True)
    print("   - test_mode.enabled: false → Full production run", flush=True)

    print("\n🚀 Starting full pipeline...", flush=True)

    # Step 1: Extract messages
    if not run_script(BASE_DIR / "src/data/extract_all_messages.py", "Step 1: Extract Messages"):
        print("\n❌ Pipeline failed at Step 1", flush=True)
        return

    # Step 2: Generate Kaya conversations
    print("\n💰 Note: This step will use xAI Grok API", flush=True)
    if not run_script(BASE_DIR / "src/data/generate_synthetic_data.py", "Step 2: Generate Kaya Conversations"):
        print("\n❌ Pipeline failed at Step 2", flush=True)
        return

    # Step 3: Prepare Portuguese data
    if not run_script(BASE_DIR / "src/data/prepare_portuguese_data.py", "Step 3: Prepare Portuguese Dataset"):
        print("\n❌ Pipeline failed at Step 3", flush=True)
        return

    # Step 4: Merge datasets
    if not run_script(BASE_DIR / "src/data/merge_datasets.py", "Step 4: Merge Datasets"):
        print("\n❌ Pipeline failed at Step 4", flush=True)
        return

    # Step 5: Train the model
    print("\n🚀 Starting model training...", flush=True)
    if not run_script(BASE_DIR / "src/finetuning/train.py", "Step 5: Model Training"):
        print("\n❌ Pipeline failed at Step 5", flush=True)
        return

    # Success!
    print("\n" + "=" * 60, flush=True)
    print("🎉 FULL PIPELINE COMPLETE!", flush=True)
    print("=" * 60, flush=True)
    print("\n✅ All synthetic data generated, merged, and model trained!", flush=True)
    print("\nGenerated files:", flush=True)
    print("  📄 data/all_messages_cleaned.jsonl", flush=True)
    print("  📄 data/finetune_chunks.jsonl", flush=True)
    print("  📄 data/synthetic_kaya.jsonl", flush=True)
    print("  📄 data/synthetic_portuguese.jsonl", flush=True)
    print("  📄 data/train_synthetic.jsonl", flush=True)
    print("  📄 data/val_synthetic.jsonl", flush=True)
    print("  🤖 models/kaya_v2_synthetic/ (fine-tuned model)", flush=True)


if __name__ == "__main__":
    main()
