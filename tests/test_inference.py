"""
Quick inference test for the fine-tuned Kaya model (LoRA adapter on Qwen3-14B).
Uses standard transformers + peft + bitsandbytes (no Unsloth needed for inference).

Run locally:
  kaya_chatbot_env/bin/python tests/test_inference.py
"""

import os
import sys
import torch
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Point HF to the model cache that was populated during Docker training
ADAPTER_PATH = str(Path(__file__).parent.parent / "models" / "kaya_v2_synthetic")

# Resolve base model from local snapshot cache (avoids permission issues with root-owned Docker volume)
_cache_hub = Path(__file__).parent.parent / "models" / ".cache" / "hub"
_model_cache = _cache_hub / "models--unsloth--qwen3-14b-bnb-4bit" / "snapshots"
if _model_cache.exists():
    BASE_MODEL = str(next(_model_cache.iterdir()))  # local path, no download needed
else:
    BASE_MODEL = "unsloth/qwen3-14b-bnb-4bit"
MAX_NEW_TOKENS = 256

# Load system prompt from config to stay in sync with production
import yaml as _yaml
try:
    _cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        _cfg = _yaml.safe_load(_f)
    SYSTEM_PROMPT = _cfg['data']['system_prompt']
except Exception:
    # Fallback — mirrors config.yaml system_prompt exactly
    SYSTEM_PROMPT = (
        "És o bot assistente do grupo de amigos 'Kaya'. "
        "Tens memória de factos, eventos e pessoas que aprendeste através das conversas passadas do grupo. "
        "Não és um membro do grupo — és um bot com acesso à memória coletiva do grupo. "
        "Nunca fales na primeira pessoa sobre experiências pessoais com membros do grupo. "
        "Refere-te sempre aos membros na terceira pessoa."
    )

# A few representative test conversations
TESTS = [
    {
        "label": "Simple greeting (PT)",
        "messages": [{"role": "user", "content": "Oi Kaya, tudo bem?"}],
    },
    {
        "label": "Ask about the group",
        "messages": [
            {"role": "user", "content": "Kaya, quem é que normalmente aparece nas conversas do grupo?"}
        ],
    },
    {
        "label": "Multi-turn conversation (EN)",
        "messages": [
            {"role": "user", "content": "Hey Kaya, what's up?"},
            {"role": "assistant", "content": "Not much, just hanging around the group chat. What's going on?"},
            {"role": "user", "content": "We're thinking of doing something this weekend. Any ideas?"},
        ],
    },
    {
        "label": "Casual question (PT)",
        "messages": [
            {"role": "user", "content": "O que é que fazias nos fins de semana com o grupo?"}
        ],
    },
]


def load_model():
    import warnings
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # Suppress redundant quantization warning: the model is pre-quantized (bnb-4bit),
    # so transformers uses its embedded config and ignores any explicit one we pass.
    warnings.filterwarnings("ignore", message="You passed `quantization_config`")

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    print(f"Applying adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, messages):
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    prompt = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            max_length=None,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def print_separator(label=""):
    width = 60
    print("\n" + "=" * width)
    if label:
        print(f"  {label}")
        print("=" * width)


def main():
    print_separator("KAYA INFERENCE TEST")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (WARNING)'}")
    print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model, tokenizer = load_model()

    vram_used = torch.cuda.memory_allocated(0) / 1024**3
    print(f"VRAM after loading: {vram_used:.1f} GB\n")

    for i, test in enumerate(TESTS, 1):
        print_separator(f"Test {i}: {test['label']}")
        print("Conversation:")
        for msg in test["messages"]:
            role_label = "User" if msg["role"] == "user" else "Kaya"
            print(f"  [{role_label}] {msg['content']}")
        print()
        print("Kaya's response:")
        response = run_inference(model, tokenizer, test["messages"])
        print(f"  {response}")

    print_separator("ALL TESTS COMPLETE")


if __name__ == "__main__":
    main()
