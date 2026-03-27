"""
Quick validation script to check pipeline outputs without regenerating
"""

from pathlib import Path
import json
import re

DATA_DIR = Path("/app/data") if Path("/app").exists() else Path(__file__).parent.parent.parent / "data"

def check_file(filepath, description):
    """Check if file exists and show stats"""
    if not filepath.exists():
        print(f"❌ {description}: NOT FOUND")
        return 0
    
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
    
    print(f"✅ {description}: {count} lines")
    
    # Show first example
    if count > 0:
        with open(filepath, 'r', encoding='utf-8') as f:
            first = json.loads(f.readline())
            print(f"   Sample keys: {list(first.keys())[:5]}")
    
    return count

def analyze_filler_words(filepath, description):
    """Analyze filler word usage in dataset"""
    if not filepath.exists():
        print(f"\n⚠️  {description} not found - skipping filler analysis")
        return
    
    print(f"\n🔍 Analyzing filler words in {description}...")
    
    fillers = ['ahahah', 'ahah', 'ahahha', 'wtf', 'lmao', 'lol']
    filler_counts = {f: 0 for f in fillers}
    total_responses = 0
    responses_with_fillers = 0
    responses_starting_with_fillers = 0
    responses_with_multiple_fillers = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # Extract assistant responses from formatted_text
                formatted_text = data.get('formatted_text', '')
                
                # Find all assistant responses
                # Pattern: <|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>
                assistant_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>'
                responses = re.findall(assistant_pattern, formatted_text, re.DOTALL)
                
                for response in responses:
                    total_responses += 1
                    response_lower = response.lower()
                    
                    # Count fillers
                    found_fillers = []
                    for filler in fillers:
                        count = len(re.findall(rf'\b{filler}\b', response_lower))
                        if count > 0:
                            filler_counts[filler] += count
                            found_fillers.append((filler, count))
                    
                    # Check if response has any fillers
                    if found_fillers:
                        responses_with_fillers += 1
                        
                        # Check if starts with filler
                        if any(response_lower.strip().startswith(f) for f in fillers):
                            responses_starting_with_fillers += 1
                        
                        # Check if has multiple filler occurrences
                        total_filler_count = sum(cnt for _, cnt in found_fillers)
                        if total_filler_count > 1:
                            responses_with_multiple_fillers += 1
                            
            except Exception as e:
                continue
    
    # Print statistics
    print(f"\n📊 Filler Word Statistics for {description}:")
    print(f"   Total assistant responses: {total_responses}")
    print(f"   Responses with fillers: {responses_with_fillers} ({responses_with_fillers/max(total_responses,1)*100:.1f}%)")
    print(f"   Starting with fillers: {responses_starting_with_fillers} ({responses_starting_with_fillers/max(total_responses,1)*100:.1f}%)")
    print(f"   With multiple fillers: {responses_with_multiple_fillers} ({responses_with_multiple_fillers/max(total_responses,1)*100:.1f}%)")
    
    print(f"\n   Filler word counts:")
    for filler, count in sorted(filler_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"      {filler}: {count}")
    
    # Quality assessment
    print(f"\n   Quality Assessment:")
    filler_ratio = responses_with_fillers / max(total_responses, 1)
    start_ratio = responses_starting_with_fillers / max(total_responses, 1)
    
    if filler_ratio > 0.7:
        print(f"      ⚠️  HIGH filler usage ({filler_ratio*100:.0f}%) - may sound repetitive")
    elif filler_ratio > 0.4:
        print(f"      ✅ MODERATE filler usage ({filler_ratio*100:.0f}%) - acceptable")
    else:
        print(f"      ✅ LOW filler usage ({filler_ratio*100:.0f}%) - natural variation")
    
    if start_ratio > 0.2:
        print(f"      ⚠️  TOO MANY responses start with fillers ({start_ratio*100:.0f}%)")
    else:
        print(f"      ✅ Good response opening variation ({start_ratio*100:.0f}% start with fillers)")

def main():
    print("=" * 60)
    print("📊 PIPELINE DATA VALIDATION")
    print("=" * 60)
    
    print("\nExtraction outputs:")
    check_file(DATA_DIR / "all_messages_cleaned.jsonl", "Cleaned messages")
    check_file(DATA_DIR / "finetune_chunks.jsonl", "Finetune chunks")
    
    print("\nSynthetic generation outputs:")
    kaya_count = check_file(DATA_DIR / "synthetic_kaya.jsonl", "Kaya conversations")
    port_count = check_file(DATA_DIR / "synthetic_portuguese.jsonl", "Portuguese data")
    
    print("\nMerged outputs:")
    train_count = check_file(DATA_DIR / "train_synthetic.jsonl", "Training set")
    val_count = check_file(DATA_DIR / "val_synthetic.jsonl", "Validation set")
    
    # Analyze filler words in training data
    if train_count > 0:
        analyze_filler_words(DATA_DIR / "train_synthetic.jsonl", "Training Set")
    
    if val_count > 0:
        analyze_filler_words(DATA_DIR / "val_synthetic.jsonl", "Validation Set")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if kaya_count == 0:
        print("⚠️  No Kaya conversations yet - need to run generation")
        print("   Note: Azure rate limits may require waiting between runs")
    
    if port_count == 0:
        print("⚠️  No Portuguese data yet - need to run prepare_portuguese_data.py")
    
    if train_count > 0 and val_count > 0:
        total = train_count + val_count
        print(f"✅ Dataset ready for training: {total} total examples")
        print(f"   Train: {train_count} | Val: {val_count}")
        print(f"\n🚀 Ready to train! Run: python src/finetuning/train.py")
    else:
        print("⏳ Dataset not complete - need to run merge step")
    
    print()

if __name__ == "__main__":
    main()
