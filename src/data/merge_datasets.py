"""
Merge synthetic datasets script.
Combines Kaya-specific and Portuguese datasets, formats, and splits.

Usage:
    python src/data/merge_datasets.py              # default: train_synthetic.jsonl
    python src/data/merge_datasets.py --variant a  # outputs train_synthetic_a.jsonl
    python src/data/merge_datasets.py --variant b  # outputs train_synthetic_b.jsonl
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.readers import SyntheticDatasetMerger
from src.config_loader import load_config


def main():
    """Run the dataset merge pipeline."""
    parser = argparse.ArgumentParser(
        description="Merge and format synthetic training datasets."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["a", "b"],
        help="Dataset variant label (a or b). Appended to output filenames for A/B comparison.",
    )
    args = parser.parse_args()

    # Detect if running in Docker
    if os.path.exists('/app'):
        data_dir = "/app/data"
    else:
        data_dir = str(Path(__file__).parent.parent.parent / "data")

    # Build output filenames — append variant suffix when supplied
    suffix = f"_{args.variant}" if args.variant else ""
    output_train = f"{data_dir}/train_synthetic{suffix}.jsonl"
    output_val = f"{data_dir}/val_synthetic{suffix}.jsonl"

    if args.variant:
        print(f"📦 Building dataset variant '{args.variant}': {output_train}")

    # Load config to get active model's id and chat template
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    config = load_config(str(config_path))
    model_id = config.get("model", {}).get("model_id", None)
    print(f"🤖 Using model_id for tokenizer: {model_id}")

    # Only use RAG-aware Kaya data (no general Portuguese alpaca data)
    # The model should learn from RAG examples only
    merger = SyntheticDatasetMerger(
        kaya_file=f"{data_dir}/synthetic_kaya.jsonl",
        portuguese_file=None,  # Removed: not RAG-aware, would confuse the model
        output_train=output_train,
        output_val=output_val,
        train_split=0.9,
        kaya_ratio=1.0,  # 100% RAG-aware Kaya Q&A data
        model_id=model_id,
        chat_template="gemma-4",
    )
    
    train_count, val_count = merger.merge_and_split()
    
    print(f"\n\n🎉 Ready for training!")
    print(f"   Run: python src/finetuning/train.py")


if __name__ == "__main__":
    main()
