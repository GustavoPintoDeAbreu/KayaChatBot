"""On-prem teacher model shared by local generation pipelines.

``TeacherModel`` loads a local instruct model in 4-bit and generates answers —
no group data leaves the box. It is used by generate_local_synthetic.py
(synthetic training data) and generate_knowledge_base.py (biographical fact
extraction, via ``LocalTeacherProvider``).

The heavy import/load happens in ``__init__`` only, so callers can inject a
stub in tests without any model download or GPU.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.synthetic_filters import strip_thinking


class TeacherModel:
    """Loads a local instruct model in 4-bit and generates answers.

    Kept thin and lazy: the heavy import/load happens in __init__, only when a
    real run (or --smoke) constructs it. Orchestration code never constructs it
    itself, so tests inject a stub instead.
    """

    def __init__(self, model_id: str, sampling: Optional[Dict[str, Any]] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        sampling = sampling or {}
        self.max_new_tokens = int(sampling.get("max_new_tokens", 400))
        self.temperature = float(sampling.get("temperature", 0.7))
        self.top_p = float(sampling.get("top_p", 0.8))
        self.top_k = int(sampling.get("top_k", 20))

        print(f"🤖 Loading teacher model: {model_id} (4-bit)…", flush=True)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="cuda", trust_remote_code=True
        )
        self.model.eval()
        self._torch = torch
        print("✓ Teacher model loaded", flush=True)

    def generate(self, system_prompt: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        # We want direct synthesized answers, not chain-of-thought. Disable
        # thinking when the template supports it (Qwen3/Qwen3.5); fall back
        # gracefully for templates that don't accept the kwarg. strip_thinking()
        # downstream is the safety net either way.
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(text=[prompt], return_tensors="pt").to("cuda")
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                use_cache=True,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


class LocalTeacherProvider:
    """Adapter exposing the cloud-provider ``generate_text`` interface on top of
    the local teacher, so provider-agnostic call sites (e.g. the knowledge
    extraction loop) work unchanged with an on-prem backend."""

    def __init__(self, teacher):
        self.teacher = teacher

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return strip_thinking(self.teacher.generate(system_prompt, user_prompt)).strip()
