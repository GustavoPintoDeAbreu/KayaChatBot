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
from src.chat.response_utils import clean_response, build_member_prompt_suffix
from src.chat.suggestions import generate_suggestions
from src.chat.gpu_lock import gpu_section, GpuBusyError

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
if _members_file and knowledge_approach in ("both", "json_only"):
    _mf = Path(_members_file) if Path(_members_file).is_absolute() else Path(config_path).parent / _members_file
    if _mf.exists():
        _members_data = json.loads(_mf.read_text(encoding="utf-8"))
        system_prompt += build_member_prompt_suffix(_members_data)

# Give the model a notion of "today" so it can reason about recency when the user
# asks about timing (runtime-only, never part of training data).
system_prompt += f"\n\nHoje é {datetime.now().strftime('%Y-%m-%d')}."

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
_suggestions_cfg = config.get("chat", {}).get("suggestions", {})
SUGGESTIONS_ENABLED = _suggestions_cfg.get("enabled", True)
SUGGESTION_COUNT = int(_suggestions_cfg.get("count", 3))

# Concurrency: the box has one GPU, so generation + RAG run one job at a time.
# Gradio's queue (set up at launch) provides fairness/queue position; the GPU lock
# (src/chat/gpu_lock.py) is the hard guarantee that serializes CUDA work.
_concurrency_cfg = config.get("chat", {}).get("concurrency", {})
CONCURRENCY_ENABLED = _concurrency_cfg.get("enabled", True)
MAX_CONCURRENT = int(_concurrency_cfg.get("max_concurrent", 1))
MAX_QUEUE_SIZE = int(_concurrency_cfg.get("max_queue_size", 32))

_BUSY_MESSAGE = (
    "⏳ Estou ocupado a responder a outras mensagens neste momento. "
    "Tenta novamente daqui a pouco. / I'm busy with other messages right now — "
    "please try again in a moment."
)


def _build_user_turn(message: str, history: list) -> tuple:
    """Return (user_message_full, context) for a turn.

    ``history`` is the prior chat (list of {"role","content"} dicts) excluding the
    current message.
    """
    context = ""
    if rag_enabled and retriever:
        try:
            context = retriever.retrieve_all(message, knowledge_approach=knowledge_approach)
        except Exception as exc:
            print(f"⚠️  RAG retrieval failed: {exc}")

    parts = []
    if context:
        parts.append(context)
    if history:
        recent_lines = []
        for item in history[-4:]:
            if isinstance(item, dict):
                role = "User" if item.get("role") == "user" else "Kaya Bot"
                content = item.get("content") or ""
                if content:
                    recent_lines.append(f"{role}: {content}")
        if recent_lines:
            parts.append("Conversa recente:\n" + "\n".join(recent_lines))
    parts.append(f"User: {message}")
    return "\n\n".join(parts), context


def bot_stream(history: list):
    """Stream the assistant reply for the last user message in ``history``.

    Yields (history, context) so the suggestion step can reuse the RAG context.
    """
    message = history[-1]["content"]
    prior = history[:-1]

    # Serialize all GPU work (RAG retrieval + streaming generation) behind the
    # shared lock so concurrent users don't race on CUDA. Held for the whole
    # stream; released when this generator is exhausted. If the GPU stays busy
    # past the timeout, degrade to a friendly busy message instead of hanging.
    try:
        with gpu_section(config):
            user_message_full, context = _build_user_turn(message, prior)

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

            history = history + [{"role": "assistant", "content": ""}]
            partial = ""
            for token in streamer:
                partial += token
                history[-1]["content"] = partial
                yield history, context

            cleaned = clean_response(partial, user_name="User", bot_name="Kaya Bot")
            history[-1]["content"] = cleaned
            yield history, context
        _log_interaction(message, cleaned)
    except GpuBusyError:
        history = history + [{"role": "assistant", "content": _BUSY_MESSAGE}]
        yield history, ""


def make_suggestions(history: list, context: str):
    """Produce gr.update()s for the suggestion chip buttons after a turn."""
    if not SUGGESTIONS_ENABLED or len(history) < 2:
        return [gr.update(visible=False, value="") for _ in range(SUGGESTION_COUNT)]

    user_msg = history[-2].get("content", "")
    bot_msg = history[-1].get("content", "")
    try:
        with gpu_section(config):
            suggestions = generate_suggestions(
                model, tokenizer, config, user_msg, bot_msg, context,
                count=SUGGESTION_COUNT,
            )
    except Exception as exc:  # noqa: BLE001 — chips are best-effort (incl. GpuBusyError)
        print(f"⚠️  Suggestion generation failed: {exc}")
        suggestions = []

    updates = []
    for i in range(SUGGESTION_COUNT):
        if i < len(suggestions):
            updates.append(gr.update(value=suggestions[i], visible=True))
        else:
            updates.append(gr.update(value="", visible=False))
    return updates


def add_user_message(message: str, history: list):
    """Append a user message and clear the input box."""
    message = (message or "").strip()
    if not message:
        return "", history
    return "", history + [{"role": "user", "content": message}]


def _hide_chips():
    return [gr.update(visible=False) for _ in range(SUGGESTION_COUNT)]


# ── Gradio UI ────────────────────────────────────────────────────────────────
_env_label = os.environ.get("KAYA_ENV", "")
_version = os.environ.get("KAYA_VERSION", "unknown")
_version_line = f"\n\n<sub>{_env_label + ' · ' if _env_label else ''}commit `{_version}`</sub>"

with gr.Blocks(title="Kaya Bot 🤖") as demo:
    gr.Markdown(
        "# Kaya Bot 🤖\n"
        "Chat com o bot do grupo Kaya. "
        "Tem acesso à memória das conversas e aos perfis dos membros do grupo."
        + _version_line
    )

    # Gradio 6.x uses the OpenAI-style messages format ({"role","content"}) by
    # default and removed the `type` kwarg, which is the format this app produces.
    chatbot = gr.Chatbot(height=480, label="Kaya Bot")
    last_context = gr.State("")

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Escreve a tua mensagem…",
            scale=8,
            show_label=False,
            container=False,
        )
        send_btn = gr.Button("Enviar", variant="primary", scale=1)

    with gr.Row():
        chip_buttons = [
            gr.Button(visible=False, size="sm", variant="secondary")
            for _ in range(SUGGESTION_COUNT)
        ]

    # Submit (textbox enter or send button): add user msg → hide chips →
    # stream answer → regenerate chips.
    for trigger in (msg.submit, send_btn.click):
        trigger(add_user_message, [msg, chatbot], [msg, chatbot]).then(
            _hide_chips, None, chip_buttons
        ).then(
            bot_stream, chatbot, [chatbot, last_context]
        ).then(
            make_suggestions, [chatbot, last_context], chip_buttons
        )

    # Clicking a chip submits its text as the next user message.
    def _click_chip(chip_value: str, history: list):
        return add_user_message(chip_value, history)[1]

    for chip in chip_buttons:
        chip.click(_click_chip, [chip, chatbot], chatbot).then(
            _hide_chips, None, chip_buttons
        ).then(
            bot_stream, chatbot, [chatbot, last_context]
        ).then(
            make_suggestions, [chatbot, last_context], chip_buttons
        )

if __name__ == "__main__":
    # Localhost-only by default: the bot serves private group memory, so it must
    # not bind to all interfaces without auth. To expose on the LAN, set
    # chat.web_server_name: "0.0.0.0" AND chat.web_auth: ["user", "password"]
    # in config.yaml.
    _chat_cfg = config.get("chat", {})
    # GRADIO_SERVER_NAME (set by the kaya-dev/kaya-prod container) overrides the
    # config default so the UI is reachable from the host / Cloudflare Tunnel;
    # production keeps the localhost-only config default otherwise.
    _server_name = os.environ.get("GRADIO_SERVER_NAME") or _chat_cfg.get("web_server_name", "127.0.0.1")
    _server_port = int(os.environ.get("KAYA_WEB_PORT") or _chat_cfg.get("web_server_port", 7860))

    # App login layer: prefer credentials from the environment (deployed via
    # GitHub/host secrets) over plaintext config. Falls back to chat.web_auth.
    _env_user = os.environ.get("KAYA_WEB_USER")
    _env_pass = os.environ.get("KAYA_WEB_PASS")
    if _env_user and _env_pass:
        _auth = (_env_user, _env_pass)
    else:
        _cfg_auth = _chat_cfg.get("web_auth")  # list [user, pass] or null
        _auth = tuple(_cfg_auth) if _cfg_auth else None

    if _auth is None and _server_name != "127.0.0.1":
        print(
            "⚠️  Serving on a non-localhost interface WITHOUT app auth. Set "
            "KAYA_WEB_USER/KAYA_WEB_PASS (or chat.web_auth) — the bot exposes "
            "private group memory."
        )

    # Queue requests so concurrent users wait their turn on the single GPU.
    # default_concurrency_limit caps how many generation events run at once;
    # max_size bounds the backlog (extra users get a busy message). The app-level
    # GPU lock in bot_stream/make_suggestions is the hard serialization guarantee.
    if CONCURRENCY_ENABLED:
        demo.queue(default_concurrency_limit=MAX_CONCURRENT, max_size=MAX_QUEUE_SIZE)

    demo.launch(
        server_name=_server_name,
        server_port=_server_port,
        auth=_auth,
        share=False,
        show_error=True,
    )
