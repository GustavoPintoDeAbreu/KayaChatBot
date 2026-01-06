"""
Test Pipeline Runner - Non-interactive version for automated testing
"""

import subprocess
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

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
            check=True
        )
        print(f"✅ {description} - Complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"Error: {e}")
        return False


def main():
    print("=" * 60)
    print("🧪 TEST MODE - SYNTHETIC DATA PIPELINE")
    print("=" * 60)
    print("\nRunning test pipeline...")
    print("⚠️  Make sure test_mode.enabled: true in config.yaml\n")
    
    # Step 1: Extract
    if not run_script(BASE_DIR / "src/data/extract_all_messages.py", "Step 1: Extract Messages"):
        return
    
    # Step 2: Generate
    print("\n⚠️  Step 2 will use Azure OpenAI (~$0.20 for test mode)")
    if not run_script(BASE_DIR / "src/data/generate_synthetic_data.py", "Step 2: Generate Kaya Conversations"):
        return
    
    # Step 3: Portuguese
    if not run_script(BASE_DIR / "src/data/prepare_portuguese_data.py", "Step 3: Prepare Portuguese Dataset"):
        return
    
    # Step 4: Merge
    if not run_script(BASE_DIR / "src/data/merge_datasets.py", "Step 4: Merge Datasets"):
        return
    
    print("\n" + "=" * 60)
    print("🎉 TEST PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n✅ Review the outputs in data/ folder")
    print("\nIf everything looks good, disable TEST_MODE and run full pipeline")


if __name__ == "__main__":
    main()
