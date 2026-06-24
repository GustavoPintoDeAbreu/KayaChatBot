"""Shared inference core for KayaChatBot.

Every chat entry point — the Gradio web UI (``web_app.py``), the CLI
(``chat.py``) and the WhatsApp bridge (``whatsapp_adapter.py``) — must run on the
*same* loaded model. The box has a single GPU and the model takes ~11 GB, so it
can only be loaded once per process. This module owns that single load
(``get_engine`` is a process-wide singleton, mirroring ``get_retriever`` and
``get_gpu_lock``) and exposes a non-streaming ``generate_reply`` used by the
WhatsApp path. The web UI keeps its own token-streaming loop but sources the
model, tokenizer and retriever from the same engine so nothing is loaded twice.

System-prompt construction lives here too (``build_system_prompt``) so the CLI,
web UI and WhatsApp bridge can each pick their own policy (e.g. the uncensored
preamble) without duplicating the member-profile / date assembly.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chat.gpu_lock import gpu_section
from src.chat.response_utils import (
    build_member_prompt_suffix,
    clean_response,
    truncate_history_line,
    wants_long_answer,
)


def build_system_prompt(
    config: Dict[str, Any],
    config_path: str,
    include_uncensored: bool = False,
) -> str:
    """Assemble the runtime system prompt.

    Mirrors the assembly previously inlined in ``web_app.py``/``chat.py``: the
    base persona, an optional uncensored preamble, the group-member profile
    suffix (when ``knowledge_approach`` injects JSON), and a "today is …" line so
    the model can reason about recency. ``include_uncensored`` is a per-caller
    choice — the web UI historically omitted it; the CLI and WhatsApp bridge
    enable it via ``chat.uncensored_mode``.
    """
    base = config["data"]["system_prompt"]
    system_prompt = base

    if include_uncensored:
        preamble = config.get("chat", {}).get("uncensored_system_prompt", "")
        if preamble:
            system_prompt = preamble + "\n\n" + system_prompt

    knowledge_approach = config.get("rag", {}).get("knowledge_approach", "both")
    members_file = config.get("data", {}).get("group_members_file")
    if members_file and knowledge_approach in ("both", "json_only"):
        members_path = Path(members_file)
        if not members_path.is_absolute():
            members_path = Path(config_path).parent / members_file
        if members_path.exists():
            members_data = json.loads(members_path.read_text(encoding="utf-8"))
            system_prompt += build_member_prompt_suffix(members_data, shuffle=True)

    system_prompt += f"\n\nHoje é {datetime.now().strftime('%Y-%m-%d')}."
    return system_prompt


def _load_model(config: Dict[str, Any]):
    """Load the fine-tuned model + tokenizer once.

    Uses Unsloth ``FastModel`` for Gemma 4 (detected from ``adapter_config.json``)
    and the standard PEFT path for Qwen3 — identical to the logic that lived in
    ``web_app.py`` so behaviour is unchanged.
    """
    model_dir = config["training"]["output_dir"]
    adapter_cfg_path = Path(model_dir) / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {model_dir}")

    base_model_name = json.loads(adapter_cfg_path.read_text())["base_model_name_or_path"]
    is_gemma4 = "gemma-4" in base_model_name.lower() or "gemma4" in base_model_name.lower()

    print(f"Loading model from {model_dir} …")
    if is_gemma4:
        from unsloth import FastModel

        model, tokenizer = FastModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=config["model"]["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )
        FastModel.for_inference(model)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, quantization_config=bnb, device_map="cuda", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_dir)
        model.eval()
    print("✓ Model loaded")
    return model, tokenizer


class KayaEngine:
    """Holds the single loaded model + RAG retriever and runs generation.

    Deliberately stateless w.r.t. system prompt and conversation history: callers
    pass those in. This keeps the heavy, shared resources (model, tokenizer,
    retriever) decoupled from per-surface policy (which system prompt, whose
    history), so the web UI and the WhatsApp bridge can share one instance.
    """

    def __init__(self, model, tokenizer, retriever, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.config = config
        rag_cfg = config.get("rag", {})
        self.rag_enabled = bool(rag_cfg.get("enabled", False)) and retriever is not None
        self.knowledge_approach = rag_cfg.get("knowledge_approach", "both")
        self._inf = config.get("inference", {})

    def build_user_turn(
        self, message: str, recent_lines: Optional[List[str]] = None, speaker_label: str = "User"
    ) -> tuple:
        """Return ``(user_message_full, context, web_result)`` for one turn.

        ``recent_lines`` is a list of already-formatted ``"<who>: <text>"`` lines.
        The RAG context is retrieved fresh for every turn (RAG is always-on).
        ``web_result`` is a ``WebSearchResult`` (``used=False`` when no web search ran).
        """
        from src.chat.web_search import maybe_web_search, WebSearchResult

        context = ""
        if self.rag_enabled and self.retriever:
            try:
                context = self.retriever.retrieve_all(
                    message, knowledge_approach=self.knowledge_approach
                )
            except Exception as exc:  # noqa: BLE001 — never let RAG failure drop a reply
                print(f"⚠️  RAG retrieval failed: {exc}")

        parts = []
        if context:
            parts.append(context)
        # Out-of-group / general-knowledge questions: augment with live web results
        # (no-op unless web_search is enabled + a key is set + the query is off-topic).
        web_result = WebSearchResult()
        if self.retriever:
            web_result = maybe_web_search(message, self.retriever, self.config)
            if web_result.used:
                parts.append(web_result.context)
        if recent_lines:
            # Truncate prior turns to a gist so the model can't copy its own long
            # previous answers back verbatim (the repetition / "stuck" bug).
            trimmed = [truncate_history_line(line) for line in recent_lines]
            parts.append("Conversa recente:\n" + "\n".join(trimmed))
        parts.append(f"{speaker_label}: {message}")
        return "\n\n".join(parts), context, web_result

    def generate_reply(
        self,
        message: str,
        speaker: str,
        recent_lines: Optional[List[str]],
        system_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Non-streaming generation for one message. Serialized on the GPU lock.

        Used by the WhatsApp bridge (the web UI streams instead). ``speaker`` is
        the display name of who is talking, used both as the prompt label and to
        trim any hallucinated continuation in ``clean_response``.
        """
        # Dynamic length: short & chatty by default, raised to the elaboration
        # ceiling only when the question actually asks for detail. An explicit
        # caller-supplied cap always wins.
        wants_long = wants_long_answer(message)
        if max_new_tokens is None:
            if wants_long:
                max_new_tokens = self._inf.get("max_new_tokens", 512)
            else:
                max_new_tokens = self._inf.get(
                    "max_new_tokens_default", min(256, self._inf.get("max_new_tokens", 512))
                )

        with gpu_section(self.config):
            user_turn, _context, web_result = self.build_user_turn(
                message, recent_lines, speaker_label=speaker
            )
            # A token cap alone won't make replies feel chatty — the model writes full
            # paragraphs well under it. Steer brevity explicitly unless detail was asked.
            brevity_hint = self._inf.get("brevity_hint", "")
            if brevity_hint and not wants_long:
                user_turn += f"\n\n({brevity_hint})"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_turn},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text=[prompt], return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self._inf.get("temperature", 1.0),
                do_sample=True,
                top_p=self._inf.get("top_p", 0.95),
                top_k=self._inf.get("top_k", 64),
                repetition_penalty=self._inf.get("repetition_penalty", 1.0),
                no_repeat_ngram_size=self._inf.get("no_repeat_ngram_size", 0),
                use_cache=True,
            )
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        reply = clean_response(raw, user_name=speaker, bot_name="Kaya Bot")
        # Append a source line so the user can see the answer is web-grounded.
        if web_result.used:
            citation = web_result.citation_line()
            if citation:
                reply = f"{reply}\n\n{citation}"
        return reply


_engine_instance: Optional[KayaEngine] = None
_engine_guard = threading.Lock()


def get_engine(config: Dict[str, Any]) -> KayaEngine:
    """Return the process-wide engine, loading the model on first use.

    Double-checked locking singleton (same pattern as ``get_retriever`` /
    ``get_gpu_lock``) so importing the web UI and the WhatsApp server in one
    process loads the model exactly once.
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_guard:
            if _engine_instance is None:
                model, tokenizer = _load_model(config)
                retriever = None
                if config.get("rag", {}).get("enabled", False):
                    try:
                        from src.chat.retriever import get_retriever

                        retriever = get_retriever(config)
                        print("✓ RAG retriever initialized")
                    except Exception as exc:  # noqa: BLE001
                        print(f"⚠️  RAG initialization failed: {exc}")
                _engine_instance = KayaEngine(model, tokenizer, retriever, config)
    return _engine_instance
