"""
Gradio web UI for KayaChatBot.
Loads the fine-tuned model + RAG retriever once at startup and serves a
streaming chat interface. Every turn is logged to
data/feedback/live_interactions.jsonl via the same logger used by chat.py.
"""
import os
import sys
import time
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config_loader import load_config
from src.chat.response_utils import clean_response, coerce_text as _coerce_text, detect_language
from src.chat.suggestions import generate_suggestions
from src.chat.gpu_lock import gpu_section, GpuBusyError
from src.chat.engine import get_engine, build_system_prompt
from src.chat import metrics
from src.chat import feedback
from src.chat.web_search import maybe_web_search, WebSearchResult

# ── Config ──────────────────────────────────────────────────────────────────
_docker_cfg = "/app/config.yaml"
_local_cfg = str(Path(__file__).parent.parent.parent / "config.yaml")
config_path = _docker_cfg if os.path.exists(_docker_cfg) else _local_cfg
config = load_config(config_path)

# ── Engine (single model load, shared with the WhatsApp bridge) ──────────────
# The model is loaded once per process by get_engine(); importing this module
# from whatsapp_server.py reuses the same instance instead of loading twice.
_engine = get_engine(config)
model = _engine.model
tokenizer = _engine.tokenizer
backend = _engine.backend
retriever = _engine.retriever
rag_enabled = _engine.rag_enabled
rag_config = config.get("rag", {})
knowledge_approach = _engine.knowledge_approach

# ── System prompt ────────────────────────────────────────────────────────────
# The web UI keeps its historical behaviour (no uncensored preamble); the
# WhatsApp bridge builds its own prompt via build_system_prompt() at startup.
system_prompt = build_system_prompt(config, config_path, include_uncensored=False)

# Interaction logging now lives in src/chat/metrics.py (shared with the WhatsApp
# bridge), which writes the same data/feedback/live_interactions.jsonl plus
# latency/length/source for the Estatísticas dashboard.

# ── Inference ────────────────────────────────────────────────────────────────
_inf = config.get("inference", {})
_suggestions_cfg = config.get("chat", {}).get("suggestions", {})
SUGGESTIONS_ENABLED = _suggestions_cfg.get("enabled", True)
SUGGESTION_COUNT = int(_suggestions_cfg.get("count", 3))

# User feedback (thumbs up/down + optional reason) and the bug-report channel.
FEEDBACK_ENABLED = bool(config.get("chat", {}).get("feedback", {}).get("enabled", True))
BUG_REPORT_ENABLED = bool(config.get("chat", {}).get("bug_report", {}).get("enabled", True))

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
    """Return (user_message_full, context) for one local-model turn.

    ``history`` is the prior chat (list of {"role","content"} dicts) excluding the
    current message. Web search is handled separately in ``bot_stream`` (it answers
    directly via Grok and bypasses the local model), so it is not injected here.
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
                content = _coerce_text(item.get("content"))
                if content:
                    recent_lines.append(f"{role}: {content}")
        if recent_lines:
            parts.append("Conversa recente:\n" + "\n".join(recent_lines))
    parts.append(f"User: {message}")
    # Steer the reply language (English message → English answer; else reinforce EU-PT).
    if detect_language(message) == "en":
        parts.append("(Reply in English.)")
    else:
        parts.append("(Responde em português europeu.)")
    return "\n\n".join(parts), context


def bot_stream(history: list):
    """Stream the assistant reply for the last user message in ``history``.

    Yields (history, context, interaction_id) so the suggestion step can reuse the
    RAG context and a thumbs rating on this answer can reference the logged interaction.
    """
    message = _coerce_text(history[-1]["content"])
    prior = history[:-1]
    _t0 = time.perf_counter()

    # Off-topic / current-events questions are answered directly by Grok's web search
    # (factually grounded, EU-PT) and bypass the local model + the GPU lock entirely.
    web_result = WebSearchResult()
    if retriever:
        web_result = maybe_web_search(message, retriever, config)
    if web_result.used and web_result.answer:
        cleaned = web_result.answer
        citation = web_result.citation_line()
        if citation:
            cleaned = f"{cleaned}\n\n{citation}"
        history = history + [{"role": "assistant", "content": cleaned}]
        yield history, "", ""
        interaction_id = metrics.log_interaction(
            source="web", user_message=message, assistant_response=cleaned,
            latency_ms=(time.perf_counter() - _t0) * 1000.0, web_search_used=True,
        )
        yield history, "", interaction_id
        return

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
            # Backend-agnostic streaming: hf streams from the in-process model,
            # gguf streams SSE deltas from the llama.cpp server. Same token loop.
            history = history + [{"role": "assistant", "content": ""}]
            partial = ""
            for token in backend.generate_stream(
                messages, max_new_tokens=_inf.get("max_new_tokens", 512), sampling=_inf
            ):
                partial += token
                history[-1]["content"] = partial
                yield history, context, ""

            cleaned = clean_response(partial, user_name="User", bot_name="Kaya Bot")
            history[-1]["content"] = cleaned
            yield history, context, ""
        interaction_id = metrics.log_interaction(
            source="web",
            user_message=message,
            assistant_response=cleaned,
            latency_ms=(time.perf_counter() - _t0) * 1000.0,
            web_search_used=False,
        )
        # Final yield carries the interaction_id so a 👍/👎 on this answer links to it.
        yield history, context, interaction_id
    except GpuBusyError:
        history = history + [{"role": "assistant", "content": _BUSY_MESSAGE}]
        yield history, "", ""


def make_suggestions(history: list, context: str):
    """Produce gr.update()s for the suggestion chip buttons after a turn."""
    if not SUGGESTIONS_ENABLED or len(history) < 2:
        return [gr.update(visible=False, value="") for _ in range(SUGGESTION_COUNT)]

    user_msg = _coerce_text(history[-2].get("content"))
    bot_msg = _coerce_text(history[-1].get("content"))
    try:
        with gpu_section(config):
            suggestions = generate_suggestions(
                backend, config, user_msg, bot_msg, context,
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
    message = _coerce_text(message).strip()
    if not message:
        return "", history
    return "", history + [{"role": "user", "content": message}]


def _hide_chips():
    return [gr.update(visible=False) for _ in range(SUGGESTION_COUNT)]


# ── Gradio UI ────────────────────────────────────────────────────────────────
_env_label = os.environ.get("KAYA_ENV", "")
_version = os.environ.get("KAYA_VERSION", "unknown")
_version_line = f"\n\n<sub>{_env_label + ' · ' if _env_label else ''}commit `{_version}`</sub>"

def _load_stats():
    """Build the dashboard markdown + per-day volume table from the metrics log."""
    agg = metrics.aggregate()
    by_src = ", ".join(f"{k}: {v}" for k, v in agg["by_source"].items()) or "—"
    md = (
        "### 📊 Estatísticas do Kaya Bot\n"
        f"- **Total de interações:** {agg['total']}\n"
        f"- **Por origem:** {by_src}\n"
        f"- **Comprimento médio da resposta:** {agg['avg_response_words']} palavras "
        f"({agg['avg_response_chars']} caracteres)\n"
        f"- **Tempo médio de resposta:** {agg['avg_latency_ms']} ms\n"
        f"- **Taxa de pesquisa web:** {agg['web_search_rate'] * 100:.1f}%\n"
    )
    per_day = agg["per_day"]
    rows = [[day, count] for day, count in per_day.items()]
    return md, rows


# ── Feedback handlers (thumbs up/down, reason box, bug report) ────────────────
def _resolve_rated_pair(history: list, index) -> tuple:
    """Return (user_message, assistant_response) for the rated message ``index``."""
    try:
        idx = index[0] if isinstance(index, (tuple, list)) else int(index)
    except (TypeError, ValueError):
        return "", ""
    if not (0 <= idx < len(history)):
        return "", ""
    item = history[idx]
    assistant_text = _coerce_text(item.get("content")) if isinstance(item, dict) else ""
    user_text = ""
    for prior in reversed(history[:idx]):
        if isinstance(prior, dict) and prior.get("role") == "user":
            user_text = _coerce_text(prior.get("content"))
            break
    return user_text, assistant_text


def on_like(data: gr.LikeData, history: list, last_iid: str):
    """Log a 👍/👎 on an answer; on 👎 reveal the optional reason box.

    Returns (reason_row update, last_feedback_id) — the id lets a follow-up reason
    attach to this same rating.
    """
    if not FEEDBACK_ENABLED:
        return gr.update(visible=False), ""
    user_text, assistant_text = _resolve_rated_pair(history, data.index)
    liked = data.liked if isinstance(data.liked, bool) else str(data.liked).lower() in ("true", "like", "liked")
    rating = "up" if liked else "down"
    # Only the most recent answer can be safely linked to its interaction_id.
    is_last = isinstance(data.index, int) and data.index == len(history) - 1
    feedback_id = feedback.log_rating(
        source="web",
        rating=rating,
        user_message=user_text,
        assistant_response=assistant_text,
        interaction_id=last_iid if is_last else None,
    )
    if rating == "down":
        return gr.update(visible=True), feedback_id
    return gr.update(visible=False), ""


def submit_reason(reason: str, feedback_id: str):
    """Attach a written reason to the prior 👎, then clear + hide the box."""
    feedback.log_comment(feedback_id=feedback_id, source="web", comment=reason or "")
    return gr.update(visible=False), "", ""


def submit_bug(description: str, contact: str, history: list):
    """Record a web bug report (separate from disliking an answer)."""
    if not (description or "").strip():
        return "⚠️ Descreve o problema antes de enviar."
    recent = []
    for item in (history or [])[-4:]:
        if isinstance(item, dict):
            role = "User" if item.get("role") == "user" else "Kaya Bot"
            content = _coerce_text(item.get("content"))
            if content:
                recent.append(f"{role}: {content}")
    feedback.log_bug_report(
        source="web",
        description=description,
        contact=contact,
        env=_env_label,
        version=_version,
        recent_turns=recent,
    )
    return "✅ Bug reportado. Obrigado! / Bug reported, thanks!"


def _load_feedback_stats():
    """Build the Feedback tab: rating counts + recent 👎 reasons + recent bug reports."""
    agg = feedback.aggregate_feedback()
    by_src = ", ".join(f"{k}: {v}" for k, v in agg["by_source"].items()) or "—"
    md = (
        "### 👍👎 Feedback dos utilizadores\n"
        f"- **Avaliações totais:** {agg['total_ratings']} "
        f"(👍 {agg['up']} · 👎 {agg['down']})\n"
        f"- **Por origem:** {by_src}\n"
        f"- **Bugs reportados:** {agg['bug_total']}\n"
    )
    down_rows = [
        [d["timestamp"][:19], d["source"], d["user_message"], d["reason"]]
        for d in agg["recent_down"]
    ]
    bug_rows = [
        [b["timestamp"][:19], b["description"], b["contact"], b["version"]]
        for b in agg["recent_bugs"]
    ]
    return md, down_rows, bug_rows


with gr.Blocks(title="Kaya Bot 🤖") as demo:
    gr.Markdown(
        "# Kaya Bot 🤖\n"
        "Chat com o bot do grupo Kaya. "
        "Tem acesso à memória das conversas e aos perfis dos membros do grupo."
        + _version_line
    )

    with gr.Tabs():
        with gr.Tab("Chat"):
            # Gradio 6.x uses the OpenAI-style messages format ({"role","content"}) by
            # default and removed the `type` kwarg, which is the format this app produces.
            chatbot = gr.Chatbot(height=480, label="Kaya Bot")
            last_context = gr.State("")
            # Carries the interaction_id of the latest answer so a 👍/👎 on it links
            # to the logged interaction; and the feedback_id of the latest 👎 so a
            # written reason attaches to that rating.
            last_interaction_id = gr.State("")
            last_feedback_id = gr.State("")

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Escreve a tua mensagem…",
                    scale=8,
                    show_label=False,
                    container=False,
                )
                send_btn = gr.Button("Enviar", variant="primary", scale=1)

            # Optional reason box, revealed only after a 👎 (thumbs-down).
            with gr.Row(visible=False) as reason_row:
                reason_box = gr.Textbox(
                    placeholder="O que correu mal nesta resposta? (opcional) / What went wrong?",
                    scale=8,
                    show_label=False,
                    container=False,
                )
                reason_btn = gr.Button("Enviar motivo", variant="secondary", scale=1)

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
                    bot_stream, chatbot, [chatbot, last_context, last_interaction_id]
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
                    bot_stream, chatbot, [chatbot, last_context, last_interaction_id]
                ).then(
                    make_suggestions, [chatbot, last_context], chip_buttons
                )

            # Thumbs up/down on a message: log the rating; a 👎 opens the reason box.
            chatbot.like(on_like, [chatbot, last_interaction_id], [reason_row, last_feedback_id])
            reason_btn.click(
                submit_reason, [reason_box, last_feedback_id], [reason_row, reason_box, last_feedback_id]
            )

        if BUG_REPORT_ENABLED:
            with gr.Tab("Reportar bug"):
                gr.Markdown(
                    "### 🐞 Reportar um bug\n"
                    "Encontraste um problema na app (erro, página partida, algo que não "
                    "funciona)? Descreve-o aqui. Isto é diferente de não gostares de uma "
                    "resposta — para isso usa o 👍/👎 no chat."
                )
                bug_desc = gr.Textbox(
                    label="Descrição do problema",
                    placeholder="O que aconteceu? O que esperavas que acontecesse?",
                    lines=4,
                )
                bug_contact = gr.Textbox(
                    label="Contacto (opcional)",
                    placeholder="Nome ou email, caso queiras seguimento",
                )
                bug_btn = gr.Button("Enviar relatório", variant="primary")
                bug_status = gr.Markdown("")
                bug_btn.click(
                    submit_bug, [bug_desc, bug_contact, chatbot], bug_status
                ).then(lambda: "", None, bug_desc)

        with gr.Tab("Estatísticas"):
            stats_md = gr.Markdown("Carrega para ver as estatísticas.")
            stats_table = gr.Dataframe(
                headers=["dia", "interações"],
                datatype=["str", "number"],
                label="Volume por dia",
                interactive=False,
            )
            refresh_btn = gr.Button("Atualizar", variant="secondary")
            refresh_btn.click(_load_stats, None, [stats_md, stats_table])
            demo.load(_load_stats, None, [stats_md, stats_table])

        with gr.Tab("Feedback"):
            fb_md = gr.Markdown("Carrega para ver o feedback.")
            fb_down_table = gr.Dataframe(
                headers=["quando", "origem", "pergunta", "motivo"],
                datatype=["str", "str", "str", "str"],
                label="👎 recentes (com motivo)",
                interactive=False,
            )
            fb_bug_table = gr.Dataframe(
                headers=["quando", "descrição", "contacto", "versão"],
                datatype=["str", "str", "str", "str"],
                label="🐞 bugs reportados",
                interactive=False,
            )
            fb_refresh = gr.Button("Atualizar", variant="secondary")
            fb_refresh.click(_load_feedback_stats, None, [fb_md, fb_down_table, fb_bug_table])
            demo.load(_load_feedback_stats, None, [fb_md, fb_down_table, fb_bug_table])

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
