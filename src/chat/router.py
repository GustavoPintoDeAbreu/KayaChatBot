"""Intent routing: decide what KIND of message this is before answering it.

The bot used to treat every message as a factual question — retrieve group
context, then elaborate. The live logs show what that produced: `Ahahhha` got a
77-word analysis of group dynamics, `hey` got a roast aimed at the wrong person.
Handed a pile of member profiles and told to elaborate, the model picks someone
at random and riffs.

So each message is classified first, and the mode selects three things together:

    mode      retrieval        prompt                    length
    banter    none             short, in-voice           ~48 tokens
    mixed     reduced top-k    conversational            ~128 tokens
    general   none             ordinary assistant        256 tokens
    factual   full (as today)  today's elaborate prompt  256 tokens

`general` exists because `factual` used to swallow every question that was not
banter, including the ones with nothing to do with the group. "quem é melhor,
Ronaldo ou Messi?" retrieved group context and every member profile, and came
back as a sourced report about the Kaya group. A question about the world is
answered like any assistant would answer it, with no group memory attached.

`command` is not a conversational mode — it is an explicit instruction
("responde só em áudio", "/clear") handled in code rather than generated.

Classification is one small generation against the already-loaded model. It
deliberately does NOT take the GPU lock: the caller holds it for the whole turn,
so routing plus answering costs one lock acquisition, not two. That matters
because `whatsapp_server._process` DROPS a message when the lock is contended
rather than queueing it — taking the lock twice would double the drop rate in a
busy group.

Any failure falls back to `factual`, i.e. exactly the previous behaviour.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

BANTER = "banter"
MIXED = "mixed"
GENERAL = "general"
FACTUAL = "factual"
ROAST = "roast"
MODES = (BANTER, MIXED, GENERAL, FACTUAL, ROAST)

CMD_AUDIO = "audio"
CMD_AUDIO_ONCE = "audio_once"
CMD_TEXT = "text"
CMD_CLEAR = "clear"
CMD_IMAGE = "image"

# The router answers with one of these bare tokens.
_LABELS = {
    "BANTER": (BANTER, None),
    "MIXED": (MIXED, None),
    "GENERAL": (GENERAL, None),
    "FACTUAL": (FACTUAL, None),
    "ROAST": (ROAST, None),
    "CMD_AUDIO": (None, CMD_AUDIO),
    "CMD_AUDIO_ONCE": (None, CMD_AUDIO_ONCE),
    "CMD_TEXT": (None, CMD_TEXT),
    "CMD_CLEAR": (None, CMD_CLEAR),
    "CMD_IMAGE": (None, CMD_IMAGE),
}

_ROUTER_SYSTEM = """You classify messages sent to a friend-group chatbot. Answer with EXACTLY ONE of these tokens and nothing else:

BANTER — social noise with no question in it: laughter, emoji, greetings, reactions, agreement, insults or jokes aimed at the bot or the group. Examples: "Ahahhha", "😂😂😂", "hey", "lol", "boa noite", "és burro", "roast me".
MIXED — chat that references a person or event but is not really asking to be informed. Examples: "o Rafa outra vez a fazer disso", "ainda me lembro daquele jantar".
FACTUAL — a request for information, memory or detail about THE GROUP: its members, its history, what was said or shared in it. Examples: "Quem é o Peter?", "quando foi o jantar?", "what does Gil do for work?", "quem mandou aquela foto do barco?".
ROAST — asking the bot to judge, rank, mock or pick on someone in the group. The answer is aimed AT a member rather than being information about one. Examples: "quem é o mais burro?", "roast the Gil", "quem tem o search history mais sus?", "diz mal do Pedro", "quem é que ganha uma luta aqui?", "who's the biggest loser here?".
GENERAL — a question, task or opinion about anything OUTSIDE the group: world knowledge, current events, football, advice, cooking, writing, code, maths. Nobody from the group needs to be looked up to answer it. Examples: "quem é melhor, Ronaldo ou Messi?", "explica-me a inflação", "escreve-me um poema sobre o Porto", "o que faço para o jantar?", "who won the Champions League?", "como é que se muda um pneu?".
CMD_AUDIO — a STANDING instruction to change how the bot replies from now on, to voice. Examples: "responde-me só em áudio", "a partir de agora fala comigo por voz", "manda sempre áudio".
CMD_TEXT — a STANDING instruction to go back to replying in text. Examples: "volta a responder por texto", "chega de áudios, escreve".
CMD_AUDIO_ONCE — asking for THIS one answer as a voice note, without changing the default. Examples: "explica isso num áudio", "manda um áudio a explicar", "responde a esta por voz".
CMD_CLEAR — asking the bot to forget or reset the recent conversation.
CMD_IMAGE — asking the bot to MAKE or ALTER a picture. Examples: "faz uma imagem de um gato astronauta", "põe o Rafa vestido de rei", "edita esta foto e mete-lhe uma coroa", "gera uma foto disto", "photoshop this".

FACTUAL and GENERAL differ only in whether the group is the subject. If answering needs the group's own memory it is FACTUAL; otherwise it is GENERAL, even when a group member is mentioned in passing:
  "o Gil também acha que o Ronaldo é melhor, e tu?" -> GENERAL (the question is about Ronaldo)
  "o Gil joga à bola?" -> FACTUAL (the question is about Gil)
  "manda o Gil para o caralho, e já agora quem ganhou a Champions?" -> GENERAL (nothing has to be looked up about Gil)

FACTUAL and ROAST differ in what the answer is FOR. Information about a member is FACTUAL; a verdict aimed at one is ROAST:
  "o que faz o Gil?" -> FACTUAL (asking to be informed)
  "porque é que o Gil é tão paneleiro?" -> ROAST (asking for a verdict)
  "quem é o mais engraçado?" -> ROAST (ranking the members against each other)
  "quantos membros tem o grupo?" -> FACTUAL

A command must be an instruction about how the bot should reply FROM NOW ON. Merely mentioning audio, voice or text is NOT a command — classify those as BANTER, MIXED or FACTUAL:
  "o áudio estava mau" -> BANTER (an opinion about a recording)
  "não gosto de áudios" -> BANTER (a preference, not an instruction)
  "ouvi o teu áudio ontem" -> MIXED (talking about a past message)
  "manda um áudio a explicar isso" -> CMD_AUDIO_ONCE (answer THIS one by voice, without changing the default)
  "audio" -> BANTER (a bare word, not an instruction)

The same rule applies to pictures. Only an actual request to produce or alter one is CMD_IMAGE:
  "que foto marada" -> BANTER (a reaction to an image)
  "quem está nesta foto?" -> FACTUAL (a question ABOUT an image, not a request to change it)
  "manda a foto do jantar" -> FACTUAL (asking for an existing photo, not a new one)
  "põe-me a andar de camelo" -> CMD_IMAGE (asking for an edit)

Reply with the token only."""


@dataclass
class Route:
    """The routing decision for one inbound message."""

    mode: str
    command: Optional[str] = None
    raw: str = ""
    fallback: bool = False

    @property
    def retrieval_enabled(self) -> bool:
        return self.mode not in (BANTER, GENERAL)


def _router_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return (config.get("chat", {}) or {}).get("router", {}) or {}


def mode_config(config: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Per-mode settings (retrieval / length / prompt override) from config.yaml."""
    modes = (config.get("chat", {}) or {}).get("modes", {}) or {}
    return modes.get(mode, {}) or {}


def _parse(text: str) -> Optional[Route]:
    """Pull a known label out of the model's output, tolerating stray tokens."""
    if not text:
        return None
    upper = text.upper()
    # Longest labels first so CMD_TEXT is not shadowed by a bare TEXT match.
    for label in sorted(_LABELS, key=len, reverse=True):
        if re.search(rf"\b{label}\b", upper):
            mode, command = _LABELS[label]
            return Route(mode=mode or FACTUAL, command=command, raw=text.strip())
    return None


def classify(
    backend: Any,
    config: Dict[str, Any],
    message: str,
    recent_lines: Optional[List[str]] = None,
) -> Route:
    """Classify one message. Never raises — falls back to FACTUAL.

    Does not acquire the GPU lock; the caller is expected to already hold it.
    """
    rcfg = _router_config(config)
    fallback_mode = rcfg.get("fallback_mode", FACTUAL)

    if not rcfg.get("enabled", True):
        return Route(mode=fallback_mode, raw="(router disabled)", fallback=True)

    text = (message or "").strip()
    if not text:
        return Route(mode=BANTER, raw="(empty)", fallback=True)

    # A couple of lines of context so "e o Rafa?" is read as a follow-up rather
    # than as noise. Kept tiny — this call must stay cheap.
    context = ""
    if recent_lines:
        context = "Recent conversation:\n" + "\n".join(recent_lines[-2:]) + "\n\n"

    messages = [
        {"role": "system", "content": _ROUTER_SYSTEM},
        {"role": "user", "content": f"{context}Message to classify:\n{text}"},
    ]
    try:
        raw = backend.generate(
            messages,
            max_new_tokens=int(rcfg.get("max_new_tokens", 8)),
            sampling={
                "temperature": float(rcfg.get("temperature", 0.0)),
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
            },
        )
    except Exception as exc:  # noqa: BLE001 — routing must never drop a reply
        print(f"⚠️  intent router failed ({type(exc).__name__}); falling back to {fallback_mode}")
        return Route(mode=fallback_mode, raw=f"error: {exc}", fallback=True)

    route = _parse(raw)
    if route is None:
        return Route(mode=fallback_mode, raw=(raw or "").strip(), fallback=True)
    return route
