"""
Full Pipeline Runner
Orchestrates the complete synthetic data generation and training pipeline.
"""

import subprocess
import sys
from pathlib import Path

# Python executable
PYTHON = "C:/Users/guga/Desktop/KayaChatBot/kaya_chatbot_env/Scripts/python.exe"
BASE_DIR = Path("C:/Users/guga/Desktop/KayaChatBot")


def run_script(script_path: Path, description: str):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 60)
    print(f"🚀 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [PYTHON, str(script_path)],
            cwd=str(BASE_DIR),
            check=True,
            capture_output=False
        )
        print(f"✅ {description} - Complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} - Interrupted by user")
        return False


def main():
    """Run the full pipeline."""
    
    print("=" * 60)
    print("🎯 KAYA CHATBOT - FULL SYNTHETIC DATA PIPELINE")
    print("=" * 60)
    print("\nThis will run the complete pipeline:")
    print("  1. Extract messages from WhatsApp + Instagram")
    print("  2. Generate Kaya-specific conversations (Azure OpenAI)")
    print("  3. Download and prepare Portuguese dataset")
    print("  4. Merge datasets and create train/val splits")
    print("\nNote: Training is separate - run train.py after this completes")
    
    # Check TEST_MODE warning
    print("\n⚠️  Check test_mode setting in config.yaml:")
    print("   - test_mode.enabled: true  → Quick test (minimal data)")
    print("   - test_mode.enabled: false → Full production run")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Extract messages
    if not run_script(BASE_DIR / "src/data/extract_all_messages.py", "Step 1: Extract Messages"):
        print("\n❌ Pipeline failed at Step 1")
        return
    
    # Step 2: Generate Kaya conversations
    print("\n💰 Note: This step will use Azure OpenAI and incur costs")
    response = input("Continue with synthetic generation? (y/n): ")
    if response.lower() != 'y':
        print("Skipping generation. Run manually later:")
        print("  python src/data/generate_synthetic_data.py")
        return
    
    if not run_script(BASE_DIR / "src/data/generate_synthetic_data.py", "Step 2: Generate Kaya Conversations"):
        print("\n❌ Pipeline failed at Step 2")
        return
    
    # Step 3: Prepare Portuguese data
    if not run_script(BASE_DIR / "src/data/prepare_portuguese_data.py", "Step 3: Prepare Portuguese Dataset"):
        print("\n❌ Pipeline failed at Step 3")
        return
    
    # Step 4: Merge datasets
    if not run_script(BASE_DIR / "src/data/merge_datasets.py", "Step 4: Merge Datasets"):
        print("\n❌ Pipeline failed at Step 4")
        return
    
    # Success!
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n✅ All synthetic data generated and merged!")
    print("\nGenerated files:")
    print("  📄 data/all_messages_cleaned.jsonl")
    print("  📄 data/finetune_chunks.jsonl")
    print("  📄 data/synthetic_kaya.jsonl")
    print("  📄 data/synthetic_portuguese.jsonl")
    print("  📄 data/train_synthetic.jsonl")
    print("  📄 data/val_synthetic.jsonl")
    
    print("\nNext step: Train the model")
    print("  python src/finetuning/train.py")
    
    # Ask if user wants to start training
    response = input("\nStart training now? (y/n): ")
    if response.lower() == 'y':
        print("\n🚀 Starting training...")
        run_script(BASE_DIR / "src/finetuning/train.py", "Model Training")
    else:
        print("\nRun training manually when ready:")
        print("  python src/finetuning/train.py")


if __name__ == "__main__":
    main()
