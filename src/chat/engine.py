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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chat import router
from src.chat.gpu_lock import gpu_section
from src.chat.response_utils import (
    build_member_prompt_suffix,
    clean_response,
    detect_language,
    truncate_history_line,
    wants_long_answer,
)


@dataclass
class Reply:
    """One answered turn: the text plus how the message was classified.

    The WhatsApp bridge needs ``route`` to act on commands (switch to voice
    replies, clear context) that are executed in code rather than generated —
    those come back with an empty ``text``.
    """

    text: str
    route: Optional["router.Route"] = None


def build_mode_system_prompt(config: Dict[str, Any], mode_prompt: str) -> str:
    """System prompt for a non-factual mode.

    Deliberately does NOT append the group-member profiles that
    ``build_system_prompt`` adds. Those profiles are exactly what made the model
    answer "😂" with an analysis of a randomly chosen member — given a pile of
    profiles and told to elaborate, it finds someone to talk about. The date line
    is kept so the model can still reason about "hoje"/"ontem".
    """
    prompt = mode_prompt
    if config.get("chat", {}).get("uncensored_mode", False):
        preamble = config.get("chat", {}).get("uncensored_system_prompt", "")
        if preamble:
            prompt = preamble + "\n\n" + prompt
    return prompt + f"\n\nHoje é {datetime.now().strftime('%Y-%m-%d')}."


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

    # With the llama.cpp (gguf) backend the heavy model lives in the llama-server
    # sidecar; this process only needs the tokenizer (for chat templating).
    from src.chat.inference_backend import resolve_backend

    if resolve_backend(config) == "gguf":
        from transformers import AutoTokenizer

        print(f"Backend=gguf — loading tokenizer only from {model_dir} (generation via llama.cpp) …")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print("✓ Tokenizer loaded")
        return None, tokenizer

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

    def __init__(self, model, tokenizer, retriever, config: Dict[str, Any], backend=None):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.config = config
        rag_cfg = config.get("rag", {})
        self.rag_enabled = bool(rag_cfg.get("enabled", False)) and retriever is not None
        self.knowledge_approach = rag_cfg.get("knowledge_approach", "both")
        self._inf = config.get("inference", {})
        if backend is None:
            from src.chat.inference_backend import build_backend

            backend = build_backend(config, model, tokenizer)
        self.backend = backend

    def build_user_turn(
        self,
        message: str,
        recent_lines: Optional[List[str]] = None,
        speaker_label: str = "User",
        retrieval: bool = True,
        top_k: Optional[int] = None,
    ) -> tuple:
        """Return ``(user_message_full, context)`` for one local-model turn.

        ``recent_lines`` is a list of already-formatted ``"<who>: <text>"`` lines.

        RAG is retrieved fresh per turn for factual and mixed intent, but is
        deliberately SKIPPED for banter (``retrieval=False``) — injecting member
        profiles into a reply to "😂" is what made the bot answer laughter with an
        essay about someone chosen at random. ``top_k`` narrows retrieval for the
        lighter `mixed` mode. Web search is handled separately in ``respond``.
        """
        context = ""
        if retrieval and self.rag_enabled and self.retriever:
            try:
                context = self.retriever.retrieve_all(
                    message, knowledge_approach=self.knowledge_approach, top_k=top_k
                )
            except Exception as exc:  # noqa: BLE001 — never let RAG failure drop a reply
                print(f"⚠️  RAG retrieval failed: {exc}")

        parts = []
        if context:
            parts.append(context)
        if recent_lines:
            # Truncate prior turns to a gist so the model can't copy its own long
            # previous answers back verbatim (the repetition / "stuck" bug).
            trimmed = [truncate_history_line(line) for line in recent_lines]
            parts.append("Conversa recente:\n" + "\n".join(trimmed))
        parts.append(f"{speaker_label}: {message}")
        return "\n\n".join(parts), context

    def generate_reply(
        self,
        message: str,
        speaker: str,
        recent_lines: Optional[List[str]],
        system_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Non-streaming generation for one message. Serialized on the GPU lock.

        Thin wrapper over ``respond`` that returns just the text, kept because
        several callers (benchmarks, the agent simulator, the probes) expect a
        plain string.
        """
        return self.respond(
            message, speaker, recent_lines, system_prompt, max_new_tokens
        ).text

    def respond(
        self,
        message: str,
        speaker: str,
        recent_lines: Optional[List[str]],
        system_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> "Reply":
        """Route, then answer. Returns the text plus the routing decision.

        The WhatsApp bridge uses the ``route`` to act on commands (switch to voice
        replies, clear context) that are handled in code rather than generated.

        Routing and generation run inside ONE ``gpu_section``. Taking the lock
        twice would double the contention, and ``whatsapp_server._process`` drops
        a message when the lock is contended rather than queueing it.
        """
        # Dynamic length: short & chatty by default, raised to the elaboration
        # ceiling only when the question actually asks for detail. An explicit
        # caller-supplied cap always wins.
        wants_long = wants_long_answer(message)
        explicit_cap = max_new_tokens is not None

        with gpu_section(self.config):
            # 1. What kind of message is this? Inside the lock, so the whole turn
            #    costs one acquisition. Never raises; falls back to `factual`.
            route = router.classify(self.backend, self.config, message, recent_lines)
            mcfg = router.mode_config(self.config, route.mode)

            # A pure command ("responde só em áudio") is executed by the caller,
            # not generated — return immediately without spending a generation.
            if route.command:
                return Reply(text="", route=route)

            # Off-topic / current-events questions are answered directly by Grok's
            # web search (factually grounded, EU-PT), bypassing the local model,
            # which garbles facts.
            #
            # This runs AFTER routing, and only for factual intent. It used to run
            # first, which meant conversational messages could trigger a web
            # lookup — "isto é creepy, estás a responder mais naturalmente" came
            # back as "não há informação clara na web sobre alterações no meu
            # comportamento". Banter must never hit the network.
            #
            # The cost is holding the GPU lock across the search call. That is
            # deliberate: the alternative is a second lock acquisition per turn,
            # and a contended lock DROPS the message rather than queueing it.
            # Search fires on a small minority of messages, so the trade is cheap.
            if route.mode == router.FACTUAL and self.retriever:
                from src.chat.web_search import maybe_web_search

                web_result = maybe_web_search(message, self.retriever, self.config)
                if web_result.used and web_result.answer:
                    citation = web_result.citation_line()
                    answer = (
                        f"{web_result.answer}\n\n{citation}" if citation else web_result.answer
                    )
                    return Reply(text=answer, route=route)

            # 2. Mode picks the length budget, unless the caller forced one.
            if not explicit_cap:
                if wants_long and route.mode == router.FACTUAL:
                    max_new_tokens = self._inf.get("max_new_tokens", 512)
                elif "max_new_tokens" in mcfg:
                    max_new_tokens = int(mcfg["max_new_tokens"])
                else:
                    max_new_tokens = self._inf.get(
                        "max_new_tokens_default", min(256, self._inf.get("max_new_tokens", 512))
                    )

            # 3. Mode picks the prompt. `banter` deliberately drops the member
            #    profiles that made the model riff about a random person.
            mode_prompt = mcfg.get("system_prompt")
            if mode_prompt:
                system_prompt = build_mode_system_prompt(self.config, mode_prompt)

            # 4. Mode picks retrieval: off for banter, reduced for mixed.
            user_turn, _context = self.build_user_turn(
                message,
                recent_lines,
                speaker_label=speaker,
                retrieval=mcfg.get("retrieval", True),
                top_k=mcfg.get("top_k"),
            )
            # A token cap alone won't make replies feel chatty — the model writes full
            # paragraphs well under it. Steer brevity explicitly unless detail was asked.
            brevity_hint = mcfg.get("brevity_hint") or self._inf.get("brevity_hint", "")
            if brevity_hint and not (wants_long and route.mode == router.FACTUAL):
                user_turn += f"\n\n({brevity_hint})"
            # Steer the reply language so an English message isn't answered in Portuguese
            # (and reinforce European-PT otherwise, against Brazilian-PT drift).
            if detect_language(message) == "en":
                user_turn += "\n\n(Reply in English.)"
            else:
                user_turn += "\n\n(Responde em português europeu.)"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_turn},
            ]
            raw = self.backend.generate(
                messages, max_new_tokens=max_new_tokens, sampling=self._inf
            )

        return Reply(
            text=clean_response(raw, user_name=speaker, bot_name="Kaya Bot"),
            route=route,
        )


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
