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
    parser.add_argument(
        "--extra-source",
        action="append",
        dest="extra_sources",
        default=[],
        metavar="PATH",
        help="Additional JSONL source file(s) to include. May be repeated.",
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

    # Source file is configurable so a run can use the on-prem generated bank
    # (data/synthetic_local.jsonl) instead of the legacy synthetic_kaya.jsonl.
    _src = config.get("data", {}).get("synthetic_source_file", "./data/synthetic_kaya.jsonl")
    _src_path = _src if os.path.isabs(_src) else f"{data_dir}/{Path(_src).name}"
    print(f"📄 Synthetic source: {_src_path}")

    extra_sources = [
        (s if os.path.isabs(s) else str(Path(data_dir) / Path(s).name))
        for s in args.extra_sources
    ]
    if extra_sources:
        print(f"📎 Extra sources: {extra_sources}")

    # Only use RAG-aware Kaya data (no general Portuguese alpaca data)
    # The model should learn from RAG examples only
    merger = SyntheticDatasetMerger(
        kaya_file=_src_path,
        portuguese_file=None,  # Removed: not RAG-aware, would confuse the model
        output_train=output_train,
        output_val=output_val,
        train_split=config.get("data", {}).get("train_test_split", 0.9),
        kaya_ratio=1.0,  # 100% RAG-aware Kaya Q&A data
        model_id=model_id,
        chat_template=config.get("model", {}).get("chat_template", "gemma-4"),
        # Bake the live persona into training (no train/inference drift) and apply
        # the term blocklist so blocked content never enters the fine-tune.
        kaya_system_prompt=config.get("data", {}).get("system_prompt"),
        blocked_terms=config.get("data", {}).get("blocked_terms", []),
        extra_files=extra_sources,
    )
    
    train_count, val_count = merger.merge_and_split()
    
    print(f"\n\n🎉 Ready for training!")
    print(f"   Run: python src/finetuning/train.py")


if __name__ == "__main__":
    main()
