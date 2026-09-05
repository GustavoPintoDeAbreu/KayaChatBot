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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.chat import router, sources, variety
from src.chat.gpu_lock import gpu_section
from src.chat.response_utils import (
    build_member_prompt_suffix,
    clean_response,
    detect_language,
    is_near_duplicate,
    previous_bot_replies,
    truncate_history_line,
    wants_long_answer,
    wants_reasoning,
)


# Modes allowed to spend the elaboration budget when the question asks for
# detail. `roast` is here because "justifica with everything you've got" is a
# request for detail whatever it is aimed at; banter and mixed are not, since
# being asked to go on at length is exactly what they exist to refuse.
_CAN_ELABORATE = (router.FACTUAL, router.GENERAL, router.ROAST, router.DEBATE)

# A message that hands the bot a position to defend, rather than asking it to
# judge one. Drawn from how the group actually phrases it: "I'll defend communism
# u capitalism!", "defende o contrário", "tu ficas com o outro lado".
_ADVOCATE_CUES = (
    "defende", "defende-te", "argumenta a favor", "faz de advogado",
    "advogado do diabo", "fica com o lado", "ficas com", "tu ficas",
    "u defend", "you defend", "defend the", "argue for", "argue in favour",
    "argue in favor", "take the side", "you take", "i'll defend", "ill defend",
    "eu defendo", "convence-me", "convence me", "persuade me",
)

# The planner's request for a lookup, on its own line.
_NEED_RE = re.compile(r"^\s*(?:[-*]\s*)?NEED\s*:\s*(.+?)\s*$", re.IGNORECASE)
_NEED_NONE = {"none", "nenhuma", "nenhum", "nada", "no", "n/a", "-", "nenhumas"}

_DEBATE_PLAN_INSTRUCTION = (
    "Antes de responderes, planeia o argumento. Escreve 2 a 4 tópicos curtos com "
    "os pontos que vais defender ou avaliar.\n\n"
    "Depois decide se precisas mesmo de factos externos para sustentar isto. "
    "Precisas quando o argumento depende de um número, de uma data, de uma "
    "estatística ou de um facto verificável que não tens. NÃO precisas quando o "
    "argumento é de lógica, de princípio, de definição ou de opinião, nem quando "
    "já tens a informação no contexto acima.\n\n"
    "Na última linha escreve 'NEED: none' se não precisas de procurar nada. "
    "Se precisares, escreve até três linhas 'NEED: <pergunta de pesquisa>', uma "
    "por facto, cada uma como uma pergunta que se pesquisa sozinha e sem nomes de "
    "pessoas do grupo. Não escrevas a resposta final."
)


@dataclass
class Reply:
    """One answered turn: the text plus how the message was classified.

    The WhatsApp bridge needs ``route`` to act on commands (switch to voice
    replies, clear context) that are executed in code rather than generated —
    those come back with an empty ``text``.

    ``citation`` is kept OUT of ``text`` on purpose. When a web-grounded answer is
    delivered as a voice note, gluing "🌐 Fontes: espn.com, record.pt" onto the
    reply means Piper reads two bare domains aloud at the end of the message. The
    caller decides where the sources go: appended for text, sent separately (or
    dropped) for speech.

    ``telemetry`` is how the turn was produced — which route fired, whether
    retrieval ran, which members ended up being talked about. The interaction log
    recorded latency and delivery medium but never the route, so a bad answer
    could not be attributed to a bad classification rather than a bad generation.
    Kept as a dict because it is written straight into the log's ``**extra``.
    """

    text: str
    route: Optional["router.Route"] = None
    citation: str = ""
    telemetry: Dict[str, Any] = field(default_factory=dict)

    @property
    def text_with_citation(self) -> str:
        """The reply as it should appear in a *written* message."""
        return f"{self.text}\n\n{self.citation}" if self.citation else self.text


def render_maintainer_clause(config: Dict[str, Any], to_maintainer: bool = False) -> str:
    """The technical-complaint clause, addressed to the group or to its author."""
    chat_cfg = config.get("chat", {}) or {}
    maintainer = str(chat_cfg.get("maintainer") or "").strip()
    key = "maintainer_self_clause" if to_maintainer else "maintainer_clause"
    return str(chat_cfg.get(key) or "").replace("{maintainer}", maintainer)


def fill_prompt_defaults(config: Dict[str, Any], prompt: str) -> str:
    """Resolve the speaker-independent form of every templated clause.

    Called by both prompt builders so that any surface which does not know who is
    writing — the web UI, the CLI, the probes — gets a complete prompt rather
    than a leaked placeholder. ``apply_speaker_rules`` upgrades it per turn.
    """
    return prompt.replace("{maintainer_clause}", render_maintainer_clause(config))


def apply_speaker_rules(config: Dict[str, Any], prompt: str, speaker: str) -> str:
    """Swap in the parts of the system prompt that depend on who is writing.

    Today that is one clause. The bot cannot change its own code, so a technical
    complaint is answered by naming the person who maintains it — and that person
    is a member of the group, which the clause never allowed for. Gustavo asked
    why image generation was so bad and was told that Gustavo has to deal with
    it, then had to reply "Yah eu sou o Gustavo".

    Applied per turn to whichever prompt the mode chose, rather than in config:
    the detailed prompt is built once at import and shared by every chat, so it
    cannot carry a speaker.
    """
    maintainer = str((config.get("chat", {}) or {}).get("maintainer") or "").strip()
    if not maintainer or (speaker or "").strip().lower() != maintainer.lower():
        return fill_prompt_defaults(config, prompt)
    self_clause = render_maintainer_clause(config, to_maintainer=True)
    return (prompt
            .replace("{maintainer_clause}", self_clause)
            .replace(render_maintainer_clause(config), self_clause))


def build_mode_system_prompt(config: Dict[str, Any], mode_prompt: str) -> str:
    """System prompt for a non-factual mode.

    Deliberately does NOT append the group-member profiles that
    ``build_system_prompt`` adds. Those profiles are exactly what made the model
    answer "😂" with an analysis of a randomly chosen member — given a pile of
    profiles and told to elaborate, it finds someone to talk about. The date line
    is kept so the model can still reason about "hoje"/"ontem".
    """
    prompt = fill_prompt_defaults(config, mode_prompt)
    if config.get("chat", {}).get("uncensored_mode", False):
        preamble = config.get("chat", {}).get("uncensored_system_prompt", "")
        if preamble:
            prompt = preamble + "\n\n" + prompt
    return prompt + f"\n\nHoje é {datetime.now().strftime('%Y-%m-%d')}."


def build_system_prompt(
    config: Dict[str, Any],
    config_path: str,
    include_uncensored: bool = False,
    max_facts: Optional[int] = None,
    sample_facts: bool = False,
) -> str:
    """Assemble the runtime system prompt.

    Mirrors the assembly previously inlined in ``web_app.py``/``chat.py``: the
    base persona, an optional uncensored preamble, the group-member profile
    suffix (when ``knowledge_approach`` injects JSON), and a "today is …" line so
    the model can reason about recency. ``include_uncensored`` is a per-caller
    choice — the web UI historically omitted it; the CLI and WhatsApp bridge
    enable it via ``chat.uncensored_mode``.
    """
    base = fill_prompt_defaults(config, config["data"]["system_prompt"])
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
            if max_facts is None:
                max_facts = int((config.get("rag", {}) or {}).get("max_facts_per_member", 0))
            system_prompt += build_member_prompt_suffix(
                members_data, shuffle=True,
                max_facts=max_facts, sample_facts=sample_facts)

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
        # How many of its own recent replies the model is shown and checked
        # against. 0 disables the check entirely.
        self._repeat_window = int(self._inf.get("no_repeat_last_replies", 4))
        # The same idea across chats and across days, keyed on WHO a reply was
        # about rather than which chat it was in — how many interactions back to
        # look, and how many previous lines about one person to show.
        self._variety_window = int(self._inf.get("variety_scan_interactions", 400))
        self._variety_recall = int(self._inf.get("variety_lines_per_member", 3))
        # How many recent openings a short reply is told not to reuse. 0 disables
        # it. Separate from the per-member recall above: this one is about the
        # shape of a sentence, not about who it is aimed at.
        self._opener_window = int(self._inf.get("variety_recent_openers", 6))
        # Set by the surface (the WhatsApp bridge) so an open-ended turn can be
        # given a freshly drawn handful of member facts. Left None by the CLI,
        # the benchmarks and the tests, which must stay reproducible.
        self.system_prompt_factory: Optional[Any] = None
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
        scope: Optional[str] = None,
        exclude_from: Optional[str] = None,
        extra_context: str = "",
        include_documents: bool = True,
        summary: str = "",
        retrieval_query: str = "",
    ) -> tuple:
        """Return ``(user_message_full, context)`` for one local-model turn.

        ``recent_lines`` is a list of already-formatted ``"<who>: <text>"`` lines.

        RAG is retrieved fresh per turn for factual and mixed intent, but is
        deliberately SKIPPED for banter (``retrieval=False``) — injecting member
        profiles into a reply to "😂" is what made the bot answer laughter with an
        essay about someone chosen at random. ``top_k`` narrows retrieval for the
        lighter `mixed` mode. ``extra_context`` carries a block the caller already
        has in hand (today: the web-search result), prepended ahead of RAG.
        ``summary`` is this chat's rolling summary of what has already scrolled
        out of the verbatim window (see ``src/chat/summary.py``).

        ``retrieval_query`` is the router's standalone rewrite of the message.
        It is used for the vector search ONLY: "E de bater na mãe?" embeds to
        nothing useful, while "quem do grupo tinha maior probabilidade de bater
        na mãe?" embeds to the question actually being asked. What the model
        reads is still the message the person wrote.
        """
        context = ""
        if retrieval and self.rag_enabled and self.retriever:
            try:
                context = self.retriever.retrieve_all(
                    retrieval_query or message,
                    knowledge_approach=self.knowledge_approach,
                    top_k=top_k,
                    scope=scope,
                    exclude_from=exclude_from,
                    include_documents=include_documents,
                )
            except Exception as exc:  # noqa: BLE001 — never let RAG failure drop a reply
                print(f"⚠️  RAG retrieval failed: {exc}")

        parts = []
        if extra_context:
            parts.append(extra_context)
        if context:
            parts.append(context)
        if summary:
            # What has already fallen out of the verbatim window, condensed. Sits
            # above the recent lines so the model reads it as older background
            # rather than as part of the live exchange.
            parts.append(f"Resumo da conversa até agora:\n{summary}")
        if recent_lines:
            # Truncate prior turns to a gist so the model can't copy its own long
            # previous answers back verbatim (the repetition / "stuck" bug).
            max_words = int(self._inf.get("history_max_words", 40))
            trimmed = [truncate_history_line(line, max_words) for line in recent_lines]
            # Said outright, because these lines are no longer a transcript of
            # turns the bot took part in: it now reads everything said in the
            # chat, and most of it was people talking to each other.
            parts.append(
                "Conversa recente no grupo (a maior parte destas mensagens não foi "
                "dirigida a ti, é o grupo a falar; lê-as para teres contexto e "
                "responde só à última):\n" + "\n".join(trimmed))
        # Who is writing, said outright. In a group every history line looks like
        # "Nome: texto", so a final line in the same shape is a weak signal — and
        # it failed: asked "why he roasting ME in my iq guess", the bot carried on
        # working through the member list and answered about somebody else
        # entirely instead of resolving "me" to the person who had just written.
        if speaker_label and speaker_label != "User":
            parts.append(
                f"Quem está a escrever agora é o {speaker_label}. "
                f'"eu", "me", "mim" e "meu" nesta mensagem referem-se ao '
                f"{speaker_label}."
            )
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
        plain string. Those are all written surfaces, so the citation is appended
        exactly as it used to be; only spoken delivery splits it off.
        """
        return self.respond(
            message, speaker, recent_lines, system_prompt, max_new_tokens
        ).text_with_citation

    def respond(
        self,
        message: str,
        speaker: str,
        recent_lines: Optional[List[str]],
        system_prompt: str,
        max_new_tokens: Optional[int] = None,
        scope: Optional[str] = None,
        exclude_from: Optional[str] = None,
        summary: str = "",
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
        # Explicit request only: a reasoning pass is a second generation held
        # inside the GPU lock, and a false positive doubles the latency of a path
        # already at 8-16s in a busy group.
        reasoning = (
            wants_reasoning(message)
            and (self.config.get("chat", {}) or {}).get("reasoning", {}).get("enabled", True)
        )

        with gpu_section(self.config):
            # 1. What kind of message is this? Inside the lock, so the whole turn
            #    costs one acquisition. Never raises; falls back to `factual`.
            route = router.classify(self.backend, self.config, message, recent_lines)
            # A GENERAL that names somebody the conversation is already about is
            # a follow-up that lost its thread, not a question about the world.
            # Corrected deterministically rather than by asking the model again.
            route = self._reconcile(route, message, recent_lines)
            # Everything downstream that asks "who is this turn about" reads the
            # rewrite too, so an elliptical follow-up resolves to a person.
            subject_text = f"{message} {route.query}".strip()
            mcfg = router.mode_config(self.config, route.mode)

            # A pure command ("responde só em áudio") is executed by the caller,
            # not generated — return immediately without spending a generation.
            # A one-off "explain this in audio" still needs a real answer — only
            # the delivery medium changes. Other commands are pure state changes
            # and are executed by the caller without generating anything.
            # CMD_COUNT is a command that still needs an answer, like
            # CMD_AUDIO_ONCE: the numbers are counted here rather than
            # remembered, but a person still has to be told them in the bot's
            # own voice.
            if route.command and route.command not in (router.CMD_AUDIO_ONCE,
                                                       router.CMD_COUNT):
                return Reply(text="", route=route,
                             telemetry=self._telemetry(route, "", subject_text, ""))

            # Counting is not retrieval. Top-k semantic search returns the chunks
            # nearest the question, which cannot answer "how many times" — asked
            # for a per-member tally the model wrote a confident table that was
            # out by 8x with the ranking inverted. The count is done over the log
            # and handed in as fact; the model only phrases it.
            count_context = ""
            if route.command == router.CMD_COUNT:
                count_context = self._count_context(message, scope)
                mcfg = router.mode_config(self.config, router.FACTUAL)

            # Off-topic / current-events questions get live facts from Grok's web
            # search. This runs AFTER routing, and only for the two informational
            # modes. It used to run first, which meant conversational messages
            # could trigger a web lookup — "isto é creepy, estás a responder mais
            # naturalmente" came back as "não há informação clara na web sobre
            # alterações no meu comportamento". Banter must never hit the network.
            #
            # Grok's answer is now CONTEXT, not the reply. Returning it verbatim
            # bypassed the persona, the uncensored preamble and clean_response, and
            # the logs show what that cost: "manda-o para o caralho? Já agora quem
            # é melhor, Cristiano ou Messi?" came back as a comparison plus "A
            # primeira parte da pergunta não se enquadra em resposta factual
            # baseada na web." Grok answers the half it can and disclaims the rest
            # in a register this group does not want. One voice answers the whole
            # message; the search only supplies the facts.
            #
            # The cost is holding the GPU lock across the search call. That is
            # deliberate: the alternative is a second lock acquisition per turn,
            # and a contended lock DROPS the message rather than queueing it.
            # Search fires on a small minority of messages, so the trade is cheap.
            web_context = ""
            citation = ""
            if route.mode in (router.FACTUAL, router.GENERAL) and self.retriever:
                from src.chat.web_search import maybe_web_search

                web_result = maybe_web_search(message, self.retriever, self.config)
                if web_result.used and web_result.answer:
                    citation = web_result.citation_line()
                    if self._synthesize_web_locally():
                        web_context = (
                            "Resultados de pesquisa web (informação atual e fiável):\n"
                            f"{web_result.answer}"
                        )
                    else:
                        return Reply(
                            text=web_result.answer, route=route, citation=citation,
                            telemetry=self._telemetry(
                                route, "", message, web_result.answer),
                        )

            # 2. Mode picks the length budget, unless the caller forced one.
            if not explicit_cap:
                if wants_long and route.mode in _CAN_ELABORATE:
                    max_new_tokens = self._inf.get("max_new_tokens", 512)
                elif "max_new_tokens" in mcfg:
                    max_new_tokens = int(mcfg["max_new_tokens"])
                else:
                    max_new_tokens = self._inf.get(
                        "max_new_tokens_default", min(256, self._inf.get("max_new_tokens", 512))
                    )

            # 3. Mode picks the prompt. `banter` deliberately drops the member
            #    profiles that made the model riff about a random person.
            # An opinion may vary; a fact may not. Only an open-ended turn gets a
            # freshly drawn handful of each member's facts — the WhatsApp prompt
            # is built once at import, so until now every roast for the whole
            # uptime saw byte-identical profiles with every fact present, and
            # reached for the same two. A factual answer keeps the full set:
            # "o que faz o Gil?" cannot depend on whether his job survived a draw.
            open_ended = variety.is_open_ended(route.mode, route.command)
            mode_prompt = mcfg.get("system_prompt")
            if open_ended and mode_prompt is None and self.system_prompt_factory is not None:
                # `mode_prompt is None` is the point: a mode that brings its own
                # prompt overwrites this two lines below, so banter and mixed were
                # paying for a full 15-profile rebuild every turn and throwing it
                # away. Roast and the null-prompt modes are the ones that actually
                # read it.
                try:
                    system_prompt = self.system_prompt_factory(sample_facts=True)
                except Exception as exc:  # noqa: BLE001 — fall back to the fixed prompt
                    print(f"⚠️  could not rebuild the system prompt: {exc}")

            if mode_prompt:
                system_prompt = build_mode_system_prompt(self.config, mode_prompt)
            # Whoever ends up holding the prompt, the clauses that depend on who
            # is writing are filled in here — the detailed prompt is built once
            # at import and shared by every chat, so it cannot carry a speaker.
            system_prompt = apply_speaker_rules(self.config, system_prompt, speaker)

            # 4. Mode picks retrieval: off for banter, reduced for mixed.
            user_turn, context = self.build_user_turn(
                message,
                recent_lines,
                speaker_label=speaker,
                retrieval=mcfg.get("retrieval", True),
                top_k=mcfg.get("top_k"),
                # A debate retrieves documents itself, numbered so they can be
                # cited and the citations checked. Leaving them in here too would
                # send the same pages twice.
                include_documents=route.mode != router.DEBATE,
                scope=scope,
                exclude_from=exclude_from,
                extra_context="\n\n".join(
                    part for part in (count_context, web_context) if part),
                # Banter gets no summary: it retrieves nothing by design, and a
                # paragraph of background would undo exactly what that mode is for.
                summary="" if route.mode == router.BANTER else summary,
                retrieval_query=route.query,
            )
            # A token cap alone won't make replies feel chatty — the model writes full
            # paragraphs well under it. Steer brevity explicitly unless detail was asked.
            brevity_hint = mcfg.get("brevity_hint") or self._inf.get("brevity_hint", "")
            if brevity_hint and not (wants_long and route.mode in _CAN_ELABORATE):
                user_turn += f"\n\n({brevity_hint})"
            # Unlike brevity_hint, this survives a request for detail: it says how
            # to answer, not how long to be.
            if mcfg.get("mode_hint"):
                user_turn += f"\n\n({mcfg['mode_hint']})"
            if route.mode == router.ROAST:
                user_turn += self._roast_hint(subject_text, recent_lines)
            if route.mode == router.DEBATE:
                user_turn += self._debate_hint(message)
            # `_roast_hint` keeps the bot off the same PERSON; this keeps it off
            # the same material about them. Peter asked to be roasted four times
            # over three days and got Rotterdam, editing other people's videos
            # and Five Guys every time — the per-chat repetition guard could not
            # see it, being per-chat and per-session.
            if open_ended:
                user_turn += self._variety_hint(subject_text, speaker, route.mode)
            # A debate plans its own argument and decides, in that same pass,
            # whether it needs facts it does not have. This replaces the generic
            # reasoning pass rather than adding to it: two planning generations
            # in one turn would double the latency for the same answer.
            debate_docs: List[Dict[str, Any]] = []
            debate_web: List[Dict[str, Any]] = []
            web_urls: List[str] = []
            allowed_markers: set = set()
            allowed_pages: set = set()
            if route.mode == router.DEBATE and self._debate_config().get("enabled", True):
                reasoning = True   # a debate always thinks first; logged as such
                plan, needs = self._debate_plan(system_prompt, user_turn)
                debate_docs, debate_web, web_urls = self._gather_evidence(
                    [route.query or message, *needs], scope, needs)
                lookups = len(debate_web)
                print(f"⚖️  debate: {len(needs)} lookup(s) requested, "
                      f"{lookups} answered, {len(debate_docs)} document chunk(s)")
                if plan:
                    user_turn += (
                        "\n\nNotas que tiraste antes de responder (usa-as, não as "
                        f"cites nem as mostres):\n{plan}"
                    )
                block, allowed_markers, allowed_pages = sources.build_source_block(
                    debate_docs, debate_web)
                if block:
                    user_turn += f"\n\n{block}"
                else:
                    # Nothing was retrieved. Saying so beats letting the model
                    # infer that it may cite from memory.
                    user_turn += ("\n\n(Não tens nenhuma fonte para isto. Argumenta "
                                  "pela lógica e diz claramente quando uma "
                                  "afirmação é tua e não vem de uma fonte. Não "
                                  "inventes números, estudos, páginas nem links.)")
                if not explicit_cap:
                    max_new_tokens = max(int(max_new_tokens or 0),
                                         int(mcfg.get("max_new_tokens", 400)))

            # Asked to think it through, the bot plans first and then answers
            # from the plan. Explicit request only — see wants_reasoning.
            if reasoning and route.mode != router.DEBATE:
                plan = self._plan(system_prompt, user_turn)
                if plan:
                    user_turn += (
                        "\n\nNotas que tiraste antes de responder (usa-as, não as "
                        f"cites nem as mostres):\n{plan}"
                    )
                if not explicit_cap:
                    max_new_tokens = self._inf.get("max_new_tokens", 512)
            # Steer the reply language so an English message isn't answered in Portuguese
            # (and reinforce European-PT otherwise, against Brazilian-PT drift).
            if detect_language(message) == "en":
                user_turn += "\n\n(Reply in English.)"
            else:
                user_turn += "\n\n(Responde em português europeu.)"
            # What it already said, so it does not say it again. `repetition_penalty`
            # and `no_repeat_ngram_size` act only WITHIN one generation and cannot
            # see the previous turn at all — which is how "So attack" and "Godamn"
            # got the byte-identical reply one turn apart.
            said_before = previous_bot_replies(recent_lines, limit=self._repeat_window)
            if said_before:
                user_turn += (
                    "\n\n(Já disseste isto há pouco — não repitas nem estas frases nem "
                    "esta construção: " + " | ".join(
                        truncate_history_line(line, 20) for line in said_before) + ")"
                )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_turn},
            ]
            raw = self.backend.generate(
                messages, max_new_tokens=max_new_tokens, sampling=self._inf
            )
            text = clean_response(raw, user_name=speaker, bot_name="Kaya Bot")

            # One retry, hotter. Only one: this is inside the GPU lock, and
            # whatsapp_server DROPS a message on a contended lock rather than
            # queueing it, so a second retry would be paid for by somebody else's
            # reply going missing.
            if said_before and is_near_duplicate(text, said_before):
                print("↻ reply repeated a recent one; regenerating once")
                hotter = {**self._inf,
                          "temperature": min(1.3, float(self._inf.get("temperature", 0.8)) + 0.25)}
                retry = self.backend.generate(
                    messages, max_new_tokens=max_new_tokens, sampling=hotter
                )
                retry_text = clean_response(retry, user_name=speaker, bot_name="Kaya Bot")
                # Keep the retry only if it is actually different; a second
                # duplicate is not an improvement over the first.
                if retry_text and not is_near_duplicate(retry_text, said_before):
                    text = retry_text

            # Cite or concede. The prompt asks the model to cite only what it was
            # handed; this is what makes that true. The group is currently taking
            # a member apart for AI-generated references that did not say what he
            # claimed, and a bot caught doing the same once is finished.
            stripped: List[str] = []
            if route.mode == router.DEBATE:
                text, stripped = sources.verify_citations(
                    text, allowed_markers, allowed_pages)
                if stripped:
                    print(f"✂️  removed {len(stripped)} invented citation(s): "
                          f"{', '.join(stripped[:5])}")
                citation = sources.citation_line(
                    debate_docs, web_urls, sources.cited_markers(text))

        telemetry = self._telemetry(route, context, subject_text, text,
                                    reasoning=reasoning)
        if route.mode == router.DEBATE:
            telemetry["debate_doc_chunks"] = len(debate_docs)
            telemetry["debate_web_lookups"] = len(debate_web)
            telemetry["citations_used"] = sorted(sources.cited_markers(text))
            telemetry["citations_stripped"] = stripped

        return Reply(
            text=text,
            route=route,
            citation=citation,
            telemetry=telemetry,
        )

    def _reconcile(self, route: "router.Route", message: str,
                   recent_lines: Optional[List[str]]) -> "router.Route":
        """Apply the GENERAL→MIXED correction, if a retriever can name members.

        Without a retriever there is no name detection, so the route is returned
        untouched — the CLI and the benchmarks run that way and must keep the
        router's own decision.
        """
        if not self.retriever:
            return route
        try:
            named = self.retriever.named_members(f"{message} {route.query}")
            if not named:
                return route
            window = int((self.config.get("chat", {}) or {}).get(
                "router", {}).get("context_lines", 6))
            recent = "\n".join((recent_lines or [])[-window:])
            return router.reconcile(route, named, self.retriever.named_members(recent))
        except Exception as exc:  # noqa: BLE001 — a correction is never worth a failure
            print(f"⚠️  could not reconcile the route: {exc}")
            return route

    def _variety_hint(self, message: str, speaker: str, mode: str = "") -> str:
        """What the bot has already said, and how it has already started saying it.

        The subjects are the members named in the message plus the speaker, so
        "roast me" resolves to the person asking — which is exactly the case that
        produced four near-identical roasts of Peter.

        The opener half needs no subject at all, and is the reason this now runs
        even when nobody is named: a banter reply is usually about nothing, and
        it is banter that recycles the same three sentence shapes.
        """
        if not self.retriever:
            return ""
        try:
            subjects = list(self.retriever.named_members(message))
            if speaker and speaker not in subjects and self.retriever.named_members(speaker):
                subjects.append(speaker)
            from src.chat import metrics

            rows = metrics.load_interactions(
                metrics.log_path(self.config), limit=self._variety_window)
            hint = variety.hint_for(subjects, rows, limit=self._variety_recall) \
                if subjects else ""
            if mode in (router.BANTER, router.MIXED) and self._opener_window:
                hint += variety.opener_hint_for(rows, mode, limit=self._opener_window)
            return hint
        except Exception as exc:  # noqa: BLE001 — a hint is never worth a failure
            print(f"⚠️  could not build the variety hint: {exc}")
            return ""

    def _count_context(self, message: str, scope: Optional[str]) -> str:
        """The counted table for a "how many times" question. "" if unanswerable.

        Deliberately returns nothing when the term cannot be identified: the
        failure this replaces was not a refusal, it was a confident wrong answer,
        so a tally that cannot be computed must not be improvised either. The
        prompt's own "diz que não sabes" rule then applies.
        """
        from src.chat import tally

        term = tally.extract_term(message)
        if not term:
            return ("Não foi possível identificar que palavra ou expressão contar. "
                    "Pergunta qual é, em vez de dares um número.")
        scope = scope or "shared"
        try:
            rows, total = tally.count_term(
                term,
                scope=scope,
                resolver=tally.load_resolver(self.config),
                # Only the shared scope reaches back into the pre-bot export; a
                # DM's history is its own file and nothing else.
                archive=tally.archive_path(self.config) if scope == "shared" else None,
            )
        except Exception as exc:  # noqa: BLE001 — a failed count must not drop the reply
            print(f"⚠️  tally failed: {exc}")
            return ("Não foi possível contar isso agora. Diz que não conseguiste "
                    "contar, em vez de dares um número.")
        return tally.format_table(term, rows, total)

    def _plan(self, system_prompt: str, user_turn: str) -> str:
        """One private pass of notes before the answer. "" on any failure.

        The plan is never sent. It goes back into the user turn as notes, so the
        reply is still written once, in one voice, by the same prompt as every
        other reply — the alternative, showing the reasoning, would put a
        different register in the group chat than the persona everywhere else.

        Runs inside the caller's ``gpu_section``: routing, planning and answering
        are one lock acquisition, because ``whatsapp_server._process`` DROPS a
        message on a contended lock rather than queueing it.
        """
        rcfg = (self.config.get("chat", {}) or {}).get("reasoning", {}) or {}
        try:
            raw = self.backend.generate(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_turn + "\n\n" + rcfg.get(
                        "plan_instruction",
                        "Antes de responderes, escreve 2 a 4 tópicos curtos com o "
                        "que é mesmo relevante para responder bem a isto. Só os "
                        "tópicos, sem introdução e sem a resposta final.")},
                ],
                max_new_tokens=int(rcfg.get("max_new_tokens", 200)),
                sampling={**self._inf, "temperature": float(rcfg.get("temperature", 0.4))},
            )
        except Exception as exc:  # noqa: BLE001 — a failed plan answers without one
            print(f"⚠️  reasoning pass failed ({type(exc).__name__}); answering directly")
            return ""
        return (raw or "").strip()

    def _debate_config(self) -> Dict[str, Any]:
        return (self.config.get("chat", {}) or {}).get("debate", {}) or {}

    def _debate_hint(self, message: str) -> str:
        """Whether this turn takes a side or judges one, and the rule for each.

        Both shapes appear in the live log within minutes of each other: Rafa's
        "I'll defend communism u capitalism!" assigns the bot a side, and
        Frederico's "dá a tua opinião para saber quem tem razão, sê analítico"
        asks it to referee. They need opposite instructions — an advocate that
        both-sides its own position is useless, and a referee that picks a team
        before weighing the claims is worse than useless.

        Deterministic, like ``_roast_hint``: which of the two this is depends on
        whether the message hands the bot a position, and that is visible in the
        words. Never raises.
        """
        lowered = (message or "").lower()
        assigned = any(cue in lowered for cue in _ADVOCATE_CUES)
        if assigned:
            return ("\n\n(Foste posto de um lado da discussão. Defende essa posição "
                    "e comprometete-te com ela: dá os melhores argumentos que existem "
                    "a favor dela, não faças o 'por um lado, por outro lado' e não "
                    "mudes de lado a meio. Podes reconhecer o ponto mais forte do "
                    "outro lado, mas só para lhe responderes.)")
        return ("\n\n(Estás a arbitrar, não estás a escolher uma equipa. Vai "
                "afirmação a afirmação e diz quem tem razão em cada uma, com a "
                "fonte quando tens uma. Podes dar razão a pessoas diferentes em "
                "pontos diferentes, e se ninguém tiver razão, diz isso. Não digas "
                "que ambos têm razão para não chatear ninguém.)")

    def _debate_plan(self, system_prompt: str, user_turn: str) -> Tuple[str, List[str]]:
        """Notes for the argument, and what it needs looked up. ``(plan, needs)``.

        This is the "knows when it needs facts" decision, and it is made by the
        model rather than by keywords, because the question is not what words the
        message contains but whether the case being made rests on a number. "Quem
        tem razão sobre o custo da comida desde 1970" needs a statistic; "defende
        que a habitação pública é a solução" is an argument from principle and a
        web lookup would add latency and nothing else.

        The plan itself is never sent — same contract as ``_plan``. Only the
        ``NEED:`` lines have an effect the user can see.

        Returns ``("", [])`` on any failure, which answers without evidence
        rather than not answering.
        """
        dcfg = self._debate_config()
        instruction = dcfg.get("plan_instruction", _DEBATE_PLAN_INSTRUCTION)
        try:
            raw = self.backend.generate(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_turn}\n\n{instruction}"},
                ],
                max_new_tokens=int(dcfg.get("plan_max_new_tokens", 300)),
                sampling={**self._inf, "temperature": float(dcfg.get("temperature", 0.3))},
            )
        except Exception as exc:  # noqa: BLE001 — a failed plan argues without one
            print(f"⚠️  debate plan failed ({type(exc).__name__}); arguing directly")
            return "", []

        text = (raw or "").strip()
        needs: List[str] = []
        notes: List[str] = []
        for line in text.splitlines():
            match = _NEED_RE.match(line.strip())
            if not match:
                if line.strip():
                    notes.append(line.rstrip())
                continue
            want = match.group(1).strip().strip('"').strip()
            if not want or want.lower().rstrip(".") in _NEED_NONE:
                continue
            if want.lower() not in {n.lower() for n in needs}:
                needs.append(want)
        limit = max(0, int(dcfg.get("max_lookups", 3)))
        return "\n".join(notes).strip(), needs[:limit]

    def _gather_evidence(self, queries: Sequence[str], scope: Optional[str],
                         web_queries: Sequence[str]) -> Tuple[List[Dict[str, Any]],
                                                              List[Dict[str, Any]],
                                                              List[str]]:
        """Documents and web results for a debate. ``(docs, web, source_urls)``.

        Documents are searched for every query including the question itself, so
        a debate about a paper somebody shared works even when the planner asks
        for no lookups. The web is only searched for what the planner explicitly
        asked for — that is the whole point of the gate.
        """
        docs: List[Dict[str, Any]] = []
        seen_docs: set = set()
        if self.retriever is not None:
            per_query = int(self._debate_config().get("doc_chunks_per_query", 3))
            for query in queries:
                if not query:
                    continue
                try:
                    hits = self.retriever.retrieve_documents(
                        query, top_k=per_query, scope=scope)
                except Exception as exc:  # noqa: BLE001
                    print(f"⚠️  document retrieval failed: {exc}")
                    continue
                for hit in hits:
                    key = (hit.get("doc_id"), hit.get("page_start"), hit.get("page_end"))
                    if key in seen_docs:
                        continue
                    seen_docs.add(key)
                    docs.append(hit)

        web: List[Dict[str, Any]] = []
        urls: List[str] = []
        if web_queries:
            from src.chat.web_search import search_for

            for query in web_queries:
                result = search_for(query, self.retriever, self.config)
                if not result.used or not result.answer:
                    continue
                web.append({"answer": result.answer, "sources": result.sources,
                            "query": query})
                urls.extend(result.sources or [])
        return docs, web, urls

    def _roast_hint(self, message: str, recent_lines: Optional[List[str]]) -> str:
        """Keep an unaimed roast off the member it just hit.

        A roast with no named target used to land on whoever had the most
        material in the prompt, every time. Measured over the first group
        session: one member took 29.4% of all the mentions the bot made, against
        an even split of 8%, and 37.5% of turns named somebody nobody had asked
        about. The group's reading was that the bot had it in for him personally.

        When the message names a target, that is the target — "roast the Gil"
        must roast the Gil — and what gets added instead is an instruction to
        stay on them. Only an *unaimed* request gets steered towards someone
        fresh, and only away from the members the last few replies already went
        after, which are read back out of the history rather than tracked in new
        state.

        That split is the fix for the drift (2026-09-04). "Escolhe alguém que não
        tenha sido gozado ... e varia" used to live in the roast `mode_hint`, so
        it was appended to EVERY roast, aimed or not — a standing order to find a
        fresh victim even when the message had already named one. Two of the
        three roasts in the fortnight to 2026-09-03 answered the question and
        then appended an unrelated paragraph about somebody who was not in the
        conversation, and the group called both out.
        """
        if not self.retriever:
            return ""
        try:
            named = self.retriever.named_members(message)
            if named:
                who = ", ".join(named)
                return (f"\n\n(O roast é sobre {who}. Fala só dess"
                        f"{'es' if len(named) > 1 else 'a pessoa'} e de mais ninguém.)")
            recent = []
            for reply in previous_bot_replies(recent_lines, limit=self._repeat_window):
                for name in self.retriever.named_members(reply):
                    if name not in recent:
                        recent.append(name)
            if not recent:
                return ""
            return ("\n\n(Ninguém foi nomeado. Não escolhas outra vez " +
                    ", ".join(recent) + ": já falaste deles agora mesmo. Escolhe "
                    "UMA outra pessoa do grupo e fala só dela.)")
        except Exception as exc:  # noqa: BLE001 — a hint is never worth a failure
            print(f"⚠️  could not build the roast hint: {exc}")
            return ""

    def _telemetry(self, route: "router.Route", context: str, message: str,
                   reply: str, reasoning: bool = False) -> Dict[str, Any]:
        """What the interaction log records about how this turn was produced.

        ``reply_members`` is the one that matters and the reason this exists: the
        live logs showed the same two people being talked about turn after turn,
        and there was no way to show it short of reading the thread. Counting who
        the bot actually names — not who the question named — is what makes that
        measurable, and what a later change has to move.

        Never raises: telemetry must not be able to cost somebody their reply.
        """
        info: Dict[str, Any] = {
            "route_mode": route.mode,
            "route_command": route.command or "",
            "route_fallback": route.fallback,
            "route_raw": route.raw,
            "route_query": route.query,
            "route_reconciled_from": route.reconciled_from,
            "retrieval_enabled": route.retrieval_enabled,
            "retrieved_chars": len(context or ""),
            "reasoning_used": reasoning,
        }
        try:
            if self.retriever:
                info["query_members"] = self.retriever.named_members(message)
                info["reply_members"] = self.retriever.named_members(reply)
        except Exception as exc:  # noqa: BLE001 — a log field is never worth a failure
            print(f"⚠️  telemetry member scan failed: {exc}")
        return info

    def _synthesize_web_locally(self) -> bool:
        """Whether a web result is rewritten by the local model or sent verbatim.

        Verbatim was the original design because the retired fine-tuned E4B
        garbled raw web snippets. Grok now hands over a finished answer rather
        than snippets, and the model serving this is a capable stock 12B, so the
        answer is used as context. Flip ``web_search.synthesize_locally`` to false
        to get the old behaviour back in one edit.
        """
        return bool(
            (self.config.get("web_search", {}) or {}).get("synthesize_locally", True)
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
