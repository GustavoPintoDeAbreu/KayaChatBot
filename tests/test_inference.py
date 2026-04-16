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

# Resolve model paths from config (supports any active profile)
import json as _json
import yaml as _yaml
from src.config_loader import load_config as _load_config

_cfg_path = Path(__file__).parent.parent / "config.yaml"
try:
    _cfg = _load_config(str(_cfg_path))
    ADAPTER_PATH = str(Path(__file__).parent.parent / _cfg['training']['output_dir'])
    _adapter_config = Path(ADAPTER_PATH) / "adapter_config.json"
    if _adapter_config.exists():
        _acfg = _json.loads(_adapter_config.read_text(encoding='utf-8'))
        BASE_MODEL = _acfg.get('base_model_name_or_path', _cfg['model']['model_id'])
        _BASE_MODEL_CLASS = _acfg.get('auto_mapping', {}).get('base_model_class', '')
    else:
        BASE_MODEL = _cfg['model']['model_id']
        _BASE_MODEL_CLASS = ''
    SYSTEM_PROMPT = _cfg['data']['system_prompt']
except Exception:
    ADAPTER_PATH = str(Path(__file__).parent.parent / "models" / "kaya_v2_synthetic")
    BASE_MODEL = "unsloth/Qwen3-14B-bnb-4bit"
    _BASE_MODEL_CLASS = ''
    SYSTEM_PROMPT = (
        "És o bot assistente do grupo de amigos 'Kaya'. "
        "Tens memória de factos, eventos e pessoas que aprendeste através das conversas passadas do grupo. "
        "Não és um membro do grupo — és um bot com acesso à memória coletiva do grupo. "
        "Nunca fales na primeira pessoa sobre experiências pessoais com membros do grupo. "
        "Refere-te sempre aos membros na terceira pessoa."
    )
MAX_NEW_TOKENS = 256

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

    # Gemma 4 uses Gemma4ForConditionalGeneration (not registered with AutoModelForCausalLM)
    if 'Gemma4' in _BASE_MODEL_CLASS:
        from transformers import Gemma4ForConditionalGeneration
        base = Gemma4ForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
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
