"""
Inference Script
Tests the fine-tuned model with sample prompts.
"""
import argparse
import json
import os
import torch
from pathlib import Path

from src.config_loader import load_config


def _load_model_unsloth(model_dir, max_seq_length, model_id):
    """Load model via Unsloth FastModel (Gemma 4) or FastLanguageModel (others)."""
    is_gemma4 = "gemma-4" in model_id.lower() or "gemma4" in model_id.lower()
    if is_gemma4:
        from unsloth import FastModel
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastModel.for_inference(model)
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    return model, tokenizer


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
    
    # Detect base model from adapter config
    adapter_config_path = Path(model_dir) / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_cfg = json.loads(adapter_config_path.read_text(encoding='utf-8'))
        base_model_id = adapter_cfg.get('base_model_name_or_path', config['model']['model_id'])
    else:
        base_model_id = config['model']['model_id']
    
    # Load model
    print(f"\n2. Loading fine-tuned model ({base_model_id})...")
    model, tokenizer = _load_model_unsloth(model_dir, max_seq_length, base_model_id)
    print(f"   ✓ Model loaded")
    
    # Test prompts — formatted as chat messages for apply_chat_template
    system_prompt = config.get('data', {}).get('system_prompt', '')

    # Prepend uncensored preamble when uncensored_mode is enabled (runtime only, not training)
    chat_cfg = config.get('chat', {})
    if chat_cfg.get('uncensored_mode', False):
        uncensored_preamble = chat_cfg.get('uncensored_system_prompt', '')
        if uncensored_preamble:
            system_prompt = uncensored_preamble + "\n\n" + system_prompt

    test_messages = [
        [
            {"role": "user", "content": "Gil João: O que acham desta música?\nPeter: "},
        ],
        [
            {"role": "user", "content": "What did Gil João say about music?"},
        ],
        [
            {"role": "user", "content": "Tell me about the group chat history."},
        ],
    ]
    if system_prompt:
        for msgs in test_messages:
            msgs.insert(0, {"role": "system", "content": system_prompt})
    
    print(f"\n3. Running inference tests...")
    print("=" * 60)
    
    for i, messages in enumerate(test_messages, 1):
        user_content = messages[-1]["content"]
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {user_content}")
        print(f"\nResponse:")
        
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text=input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            repetition_penalty=1.0,
            use_cache=True
        )
        # Decode only the new tokens (skip the input)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(response)
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("✅ Inference tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
