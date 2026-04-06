"""
Inference Script
Tests the fine-tuned model with sample prompts.
"""
import argparse
import os
import torch
from unsloth import FastLanguageModel

from src.config_loader import load_config


def main():
    print("=" * 60)
    print("Inference Pipeline")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="KayaChatBot inference script.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Model profile name (overrides active_model_profile in config.yaml).",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    print(f"\n1. Loading configuration...")
    config = load_config(config_path, profile_override=args.profile)
    
    model_dir = config['training']['output_dir']
    max_seq_length = config['model']['max_seq_length']
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"\n❌ Error: Model not found at {model_dir}")
        print(f"   Please run training first: python src/train.py")
        return
    
    print(f"   ✓ Model directory: {model_dir}")
    
    # Load model
    print(f"\n2. Loading fine-tuned model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"   ✓ Model loaded")
    
    # Test prompts
    test_prompts = [
        "Gil João: O que acham desta música?\nPeter:",
        "What did Gil João say about music?\nKaya:",
        "Tell me about the group chat history.\nKaya:",
    ]
    
    print(f"\n3. Running inference tests...")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        print(f"\nResponse:")
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            use_cache=True
        )
        response = tokenizer.batch_decode(outputs)[0]
        print(response)
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("✅ Inference tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
