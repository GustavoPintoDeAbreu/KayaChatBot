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
from typing import Any, Dict, List, Optional

from src.chat import router
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
_CAN_ELABORATE = (router.FACTUAL, router.GENERAL, router.ROAST)


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
            system_prompt += build_member_prompt_suffix(
                members_data, shuffle=True,
                max_facts=int((config.get("rag", {}) or {}).get("max_facts_per_member", 0)))

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
        summary: str = "",
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
        """
        context = ""
        if retrieval and self.rag_enabled and self.retriever:
            try:
                context = self.retriever.retrieve_all(
                    message,
                    knowledge_approach=self.knowledge_approach,
                    top_k=top_k,
                    scope=scope,
                    exclude_from=exclude_from,
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
            parts.append("Conversa recente:\n" + "\n".join(trimmed))
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
                             telemetry=self._telemetry(route, "", message, ""))

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
            mode_prompt = mcfg.get("system_prompt")
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
                scope=scope,
                exclude_from=exclude_from,
                extra_context="\n\n".join(
                    part for part in (count_context, web_context) if part),
                # Banter gets no summary: it retrieves nothing by design, and a
                # paragraph of background would undo exactly what that mode is for.
                summary="" if route.mode == router.BANTER else summary,
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
                user_turn += self._roast_hint(message, recent_lines)
            # Asked to think it through, the bot plans first and then answers
            # from the plan. Explicit request only — see wants_reasoning.
            if reasoning:
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

        return Reply(
            text=text,
            route=route,
            citation=citation,
            telemetry=self._telemetry(route, context, message, text,
                                      reasoning=reasoning),
        )

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

    def _roast_hint(self, message: str, recent_lines: Optional[List[str]]) -> str:
        """Keep an unaimed roast off the member it just hit.

        A roast with no named target used to land on whoever had the most
        material in the prompt, every time. Measured over the first group
        session: one member took 29.4% of all the mentions the bot made, against
        an even split of 8%, and 37.5% of turns named somebody nobody had asked
        about. The group's reading was that the bot had it in for him personally.

        When the message names a target, that is the target and nothing is added
        — "roast the Gil" must roast the Gil. Only an *unaimed* request gets
        steered, and only away from the members the last few replies already
        went after, which are read back out of the history rather than tracked in
        new state.
        """
        if not self.retriever:
            return ""
        try:
            if self.retriever.named_members(message):
                return ""
            recent = []
            for reply in previous_bot_replies(recent_lines, limit=self._repeat_window):
                for name in self.retriever.named_members(reply):
                    if name not in recent:
                        recent.append(name)
            if not recent:
                return ""
            return ("\n\n(Não escolhas outra vez " + ", ".join(recent) +
                    ": já falaste deles agora mesmo. Escolhe outra pessoa do grupo.)")
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
