"""
Gradio web UI for KayaChatBot.
Loads the fine-tuned model + RAG retriever once at startup and serves a
streaming chat interface. Every turn is logged to
data/feedback/live_interactions.jsonl via the same logger used by chat.py.
"""
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread

import torch
import gradio as gr
from transformers import TextIteratorStreamer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config_loader import load_config

# ── Config ──────────────────────────────────────────────────────────────────
_docker_cfg = "/app/config.yaml"
_local_cfg = str(Path(__file__).parent.parent.parent / "config.yaml")
config_path = _docker_cfg if os.path.exists(_docker_cfg) else _local_cfg
config = load_config(config_path)

# ── Model ───────────────────────────────────────────────────────────────────
model_dir = config["training"]["output_dir"]
_adapter_cfg_path = Path(model_dir) / "adapter_config.json"
if not _adapter_cfg_path.exists():
    raise FileNotFoundError(f"adapter_config.json not found in {model_dir}")

base_model_name = json.loads(_adapter_cfg_path.read_text())["base_model_name_or_path"]
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    _bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _base = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=_bnb, device_map="cuda", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(_base, model_dir)
    model.eval()
print("✓ Model loaded")

# ── System prompt ────────────────────────────────────────────────────────────
rag_config = config.get("rag", {})
knowledge_approach = rag_config.get("knowledge_approach", "both")
system_prompt = config["data"]["system_prompt"]

_members_file = config.get("data", {}).get("group_members_file")
if _members_file:
    _mf = Path(_members_file) if Path(_members_file).is_absolute() else Path(config_path).parent / _members_file
    if _mf.exists() and knowledge_approach in ("both", "json_only"):
        _members_data = json.loads(_mf.read_text(encoding="utf-8"))
        _lines = []
        for m in _members_data.get("members", []):
            line = m["name"]
            aliases = [a for a in m.get("aliases", []) if a.lower() != m["name"].lower()]
            if aliases:
                line += f" (também conhecido como: {', '.join(aliases)})"
            notes = m.get("notes", "")
            if notes:
                sentences = [s.strip() for s in notes.split(".") if s.strip()]
                line += f" — {'. '.join(sentences[:2])}."
            _lines.append(line)
        if _lines:
            system_prompt += f"\n\nMembros do grupo Kaya: {'; '.join(_lines)}."

# ── RAG retriever ────────────────────────────────────────────────────────────
retriever = None
rag_enabled = rag_config.get("enabled", False)
if rag_enabled:
    try:
        from src.chat.retriever import get_retriever
        retriever = get_retriever(config)
        print("✓ RAG retriever initialized")
    except Exception as exc:
        print(f"⚠️  RAG initialization failed: {exc}")

# ── Interaction logger ────────────────────────────────────────────────────────
_log_dir = Path(config_path).parent / "data" / "feedback"
_log_dir.mkdir(parents=True, exist_ok=True)
_log_file = _log_dir / "live_interactions.jsonl"


def _log_interaction(user_message: str, assistant_response: str) -> None:
    entry = {
        "interaction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_message": user_message,
        "assistant_response": assistant_response,
    }
    with open(_log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Inference ────────────────────────────────────────────────────────────────
_inf = config.get("inference", {})


def respond(message: str, history: list):
    """Stream a response token-by-token and log the completed turn."""
    # RAG context
    context = ""
    if rag_enabled and retriever:
        try:
            context = retriever.retrieve_all(message, knowledge_approach=knowledge_approach)
        except Exception as exc:
            print(f"⚠️  RAG retrieval failed: {exc}")

    # Build user turn: context + recent chat history + current message
    # Gradio 6.x passes history as list[dict] with "role"/"content" keys.
    parts = []
    if context:
        parts.append(context)
    if history:
        recent_lines = []
        for item in history[-3:]:
            if isinstance(item, dict):
                role = "User" if item.get("role") == "user" else "Kaya Bot"
                content = item.get("content") or ""
                if content:
                    recent_lines.append(f"{role}: {content}")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, bot_msg = item
                if user_msg:
                    recent_lines.append(f"User: {user_msg}")
                if bot_msg:
                    recent_lines.append(f"Kaya Bot: {bot_msg}")
        if recent_lines:
            parts.append(f"Conversa recente:\n" + "\n".join(recent_lines))
    parts.append(f"User: {message}")
    user_message_full = "\n\n".join(parts)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_full},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[prompt], return_tensors="pt").to("cuda")

    # timeout=60s prevents the iterator from hanging if the generation thread crashes
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=_inf.get("max_new_tokens", 512),
        temperature=_inf.get("temperature", 1.0),
        do_sample=True,
        top_p=_inf.get("top_p", 0.95),
        top_k=_inf.get("top_k", 64),
        repetition_penalty=_inf.get("repetition_penalty", 1.0),
        use_cache=True,
    )

    # Run generation in a background thread; iterate tokens on the main thread
    Thread(target=model.generate, kwargs=gen_kwargs, daemon=True).start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial

    _log_interaction(message, partial.strip())


# ── Gradio UI ────────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=respond,
    title="Kaya Bot 🤖",
    description=(
        "Chat com o bot do grupo Kaya. "
        "Tem acesso à memória das conversas e aos perfis dos membros do grupo."
    ),
    submit_btn="Enviar",
    stop_btn="Parar",
    examples=[
        "Quem é o Gil?",
        "O que é que o grupo costuma fazer ao fim de semana?",
        "Tell me about Gustavo",
        "Onde é que o grupo Kaya costuma sair?",
    ],
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
