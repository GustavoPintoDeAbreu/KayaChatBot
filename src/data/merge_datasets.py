"""
Merge synthetic datasets script.
Combines Kaya-specific and Portuguese datasets, formats, and splits.
"""
import sys
import os
from pathlib import Path

# Add project root to Python path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.readers import SyntheticDatasetMerger


def main():
    """Run the dataset merge pipeline."""
    
    # Detect if running in Docker
    if os.path.exists('/app'):
        data_dir = "/app/data"
    else:
        data_dir = str(Path(__file__).parent.parent.parent / "data")
    
    # Only use RAG-aware Kaya data (no general Portuguese alpaca data)
    # The model should learn from RAG examples only
    merger = SyntheticDatasetMerger(
        kaya_file=f"{data_dir}/synthetic_kaya.jsonl",
        portuguese_file=None,  # Removed: not RAG-aware, would confuse the model
        output_train=f"{data_dir}/train_synthetic.jsonl",
        output_val=f"{data_dir}/val_synthetic.jsonl",
        train_split=0.9,
        kaya_ratio=1.0  # 100% RAG-aware Kaya Q&A data
    )
    
    train_count, val_count = merger.merge_and_split()
    
    print(f"\n\n🎉 Ready for training!")
    print(f"   Run: python src/finetuning/train.py")


if __name__ == "__main__":
    main()
