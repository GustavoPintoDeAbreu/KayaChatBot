"""
Download and prepare NOVA-vision-language/alpaca-portuguese dataset.
Converts to ShareGPT format for mixing with Kaya-specific data.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
import random

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Configuration
TEST_MODE = config['test_mode']['enabled']
TEST_SAMPLE_SIZE = config['test_mode']['portuguese']['examples_limit'] if TEST_MODE else None
FULL_SAMPLE_SIZE = 800  # 800 examples for full generation

# Detect if running in Docker
import os
if os.path.exists('/app'):
    DATA_DIR = Path("/app/data")
else:
    DATA_DIR = Path(__file__).parent.parent.parent / "data"

OUTPUT_FILE = DATA_DIR / "synthetic_portuguese.jsonl"

# System prompt for general Portuguese conversations
PORTUGUESE_SYSTEM_PROMPT = """You a helpful assistant that speaks European Portuguese naturally and conversationally. You are informative and adapt your tone to the conversation."""


def download_dataset():
    """Download Portuguese dataset from Hugging Face."""
    print("📥 Downloading Portuguese conversational dataset from Hugging Face...")
    
    # Use rhaymison/superset - verified working dataset
    dataset_name = "rhaymison/superset"
    
    try:
        print(f"   Loading {dataset_name}...")
        dataset = load_dataset(dataset_name, split="train")
        print(f"✅ Downloaded {len(dataset)} examples from {dataset_name}")
        return dataset
    except Exception as e:
        print(f"❌ Could not download {dataset_name}")
        print(f"   Error: {e}")
        return None


def is_quality_example(example: Dict) -> bool:
    """Filter for quality examples."""
    
    # Must have instruction and output
    if not example.get('instruction') or not example.get('output'):
        return False
    
    # Minimum length requirements
    if len(example['instruction']) < 10 or len(example['output']) < 20:
        return False
    
    # Skip examples that are too long (likely not conversational)
    if len(example['output']) > 1000:
        return False
    
    # Skip English examples (basic check)
    english_words = ['the', 'is', 'are', 'and', 'to', 'of', 'in', 'for', 'with', 'on']
    text = (example['instruction'] + ' ' + example['output']).lower()
    english_count = sum(1 for word in english_words if f' {word} ' in text)
    
    if english_count > 3:  # Too many English words
        return False
    
    return True


def convert_to_sharegpt(example: Dict) -> Dict:
    """Convert alpaca format to ShareGPT multi-turn format."""
    
    instruction = example['instruction']
    output = example['output']
    input_text = example.get('input', '')
    
    # Create conversation
    conversations = []
    
    # User message (instruction + optional input)
    user_message = instruction
    if input_text and input_text.strip():
        user_message += f"\n\n{input_text}"
    
    conversations.append({
        "role": "user",
        "content": user_message.strip()
    })
    
    # Assistant response
    conversations.append({
        "role": "assistant",
        "content": output.strip()
    })
    
    return {
        "conversations": conversations,
        "source": "alpaca-portuguese",
        "system": PORTUGUESE_SYSTEM_PROMPT
    }


def adapt_to_european_portuguese(text: str) -> str:
    """Basic adaptations from Brazilian to European Portuguese."""
    
    # Common Brazilian → European Portuguese replacements
    replacements = {
        'você': 'tu',
        'vocês': 'vocês',  # Same in both
        'está': 'está',     # Same in both
        'estão': 'estão',   # Same in both
    }
    
    # Note: Full adaptation would require more sophisticated NLP
    # For now, we keep it simple and rely on the mix with Kaya data
    
    return text


def main():
    """Main preparation pipeline."""
    print("=" * 60)
    print("📚 PORTUGUESE DATASET PREPARATION")
    print("=" * 60)
    
    if TEST_MODE:
        print(f"\n⚠️  RUNNING IN TEST MODE")
        print(f"   - Only {TEST_SAMPLE_SIZE} examples will be sampled")
        print(f"   - Set TEST_MODE=False for full dataset ({FULL_SAMPLE_SIZE} examples)\n")
    
    # Download dataset
    dataset = download_dataset()
    
    if not dataset:
        print("❌ Failed to download dataset!")
        return
    
    # Filter for quality
    print(f"\n🔍 Filtering for quality examples...")
    quality_examples = [ex for ex in tqdm(dataset, desc="Filtering") if is_quality_example(ex)]
    print(f"✅ {len(quality_examples)} quality examples (from {len(dataset)} total)")
    
    # Sample based on mode
    sample_size = TEST_SAMPLE_SIZE if TEST_MODE else FULL_SAMPLE_SIZE
    
    if len(quality_examples) > sample_size:
        print(f"\n🎲 Randomly sampling {sample_size} examples...")
        sampled_examples = random.sample(quality_examples, sample_size)
    else:
        print(f"\n⚠️  Only {len(quality_examples)} quality examples available (requested {sample_size})")
        sampled_examples = quality_examples
    
    # Convert to ShareGPT format
    print(f"\n🔄 Converting to ShareGPT format...")
    converted_examples = []
    
    for example in tqdm(sampled_examples, desc="Converting"):
        try:
            converted = convert_to_sharegpt(example)
            converted_examples.append(converted)
        except Exception as e:
            print(f"⚠️  Skipped example due to error: {e}")
            continue
    
    print(f"✅ Converted {len(converted_examples)} examples")
    
    # Save to file
    print(f"\n💾 Saving to {OUTPUT_FILE.name}...")
    DATA_DIR.mkdir(exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for example in converted_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(converted_examples)} examples")
    
    # Statistics
    print("\n" + "=" * 60)
    print("📊 PREPARATION STATISTICS")
    print("=" * 60)
    print(f"Total downloaded: {len(dataset)}")
    print(f"Quality filtered: {len(quality_examples)} ({len(quality_examples)/len(dataset)*100:.1f}%)")
    print(f"Sampled: {len(sampled_examples)}")
    print(f"Successfully converted: {len(converted_examples)}")
    
    # Sample preview
    if converted_examples:
        print("\n📝 Sample conversation:")
        sample = converted_examples[0]
        for turn in sample['conversations'][:2]:  # Show first exchange
            role = "User" if turn['role'] == 'user' else "Kaya"
            content = turn['content'][:100] + "..." if len(turn['content']) > 100 else turn['content']
            print(f"  {role}: {content}")
    
    print(f"\n✅ Preparation complete!")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"\nNext steps:")
    print(f"  1. Review {OUTPUT_FILE.name} for quality")
    print(f"  2. Run: python src/data/merge_datasets.py  ← REQUIRED before training!")
    print(f"  3. Run: python src/finetuning/train.py")


if __name__ == "__main__":
    main()
