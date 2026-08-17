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

The same call also rewrites the message as a **standalone question** (`Q:`),
because a follow-up is unroutable and unretrievable on its own words. The live
logs show both halves failing at once: mid-thread about Bernardo, "Muda a tua
opinião, agora que entendeste que é o bana?" was classified GENERAL — the one
mode whose prompt forbids naming a member — and "E de bater na mãe?" was embedded
into the vector store as that literal string. The rewrite is for the router and
the embedder; the model still reads the message the person actually wrote.

Any failure falls back to `factual`, i.e. exactly the previous behaviour, and a
missing `Q:` line simply leaves the raw message as the retrieval query.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence

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
CMD_COUNT = "count"

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
    "CMD_COUNT": (None, CMD_COUNT),
}

_ROUTER_SYSTEM = """You classify messages sent to a friend-group chatbot. Answer with EXACTLY ONE of these tokens on the first line:

BANTER — social noise with no question in it: laughter, emoji, greetings, reactions, agreement, insults or jokes aimed at the bot or the group. Naming a member does not change that when nothing about them has to be looked up. Examples: "Ahahhha", "😂😂😂", "hey", "lol", "boa noite", "és burro", "roast me", "manda o Gil para o caralho", "diz mal deste aqui".
MIXED — chat that references a person or event but is not really asking to be informed. Examples: "o Rafa outra vez a fazer disso", "ainda me lembro daquele jantar".
FACTUAL — a request for information, memory or detail about THE GROUP: its members, its history, what was said or shared in it. Examples: "Quem é o Peter?", "quando foi o jantar?", "what does Gil do for work?", "quem mandou aquela foto do barco?".
ROAST — asking the bot to judge, rank, mock or pick on someone in the group. The answer is aimed AT a member rather than being information about one. Examples: "quem é o mais burro?", "roast the Gil", "quem tem o search history mais sus?", "diz mal do Pedro", "quem é que ganha uma luta aqui?", "who's the biggest loser here?".
GENERAL — a question, task or opinion about anything OUTSIDE the group: world knowledge, current events, football, advice, cooking, writing, code, maths. Nobody from the group needs to be looked up to answer it. Examples: "quem é melhor, Ronaldo ou Messi?", "explica-me a inflação", "escreve-me um poema sobre o Porto", "o que faço para o jantar?", "who won the Champions League?", "como é que se muda um pneu?".
CMD_AUDIO — a STANDING instruction to change how the bot replies from now on, to voice. Examples: "responde-me só em áudio", "a partir de agora fala comigo por voz", "manda sempre áudio".
CMD_TEXT — a STANDING instruction to go back to replying in text. Examples: "volta a responder por texto", "chega de áudios, escreve".
CMD_AUDIO_ONCE — asking for THIS one answer as a voice note, without changing the default. Examples: "explica isso num áudio", "manda um áudio a explicar", "responde a esta por voz".
CMD_CLEAR — asking the bot to forget or reset the recent conversation.
CMD_IMAGE — asking the bot to MAKE or ALTER a picture. Examples: "faz uma imagem de um gato astronauta", "põe o Rafa vestido de rei", "edita esta foto e mete-lhe uma coroa", "gera uma foto disto", "photoshop this".
CMD_COUNT — asking HOW MANY TIMES a word or expression was used in the chat, or who used it most. It needs every message counted, not a few remembered. Examples: "quantas vezes é que o Rafa disse isso?", "quem é que diz mais palavrões?", "conta quantas vezes dissemos X", "dá-me a lista de todos e quantas vezes cada um disse Y", "how many times did we say that?".

FACTUAL and GENERAL differ only in whether the group is the subject. If answering needs the group's own memory it is FACTUAL; otherwise it is GENERAL, even when a group member is mentioned in passing:
  "o Gil também acha que o Ronaldo é melhor, e tu?" -> GENERAL (the question is about Ronaldo)
  "o Gil joga à bola?" -> FACTUAL (the question is about Gil)
  "manda o Gil para o caralho, e já agora quem ganhou a Champions?" -> GENERAL (nothing has to be looked up about Gil)

A CORRECTION about a member — their name, who they are, what they do, what they
said — is FACTUAL even though it is not phrased as a question, because answering
it means checking it against what is known about that member, and only FACTUAL is
given that to check against:
  "esse não é o Gil, é o Peter" -> FACTUAL
  "este bernardo é o bana, não é o benny pereira" -> FACTUAL
  "o Romano nunca trabalhou na Glovo" -> FACTUAL
  "estás enganado" -> the mode the thread is in
  "escreveste isso mal" -> BANTER (about the message, not about a member)

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

CMD_COUNT is only for questions that need TALLYING every message. A question about the group answerable from memory is FACTUAL:
  "quantos membros tem o grupo?" -> FACTUAL (a fact, not a tally of messages)
  "quantas vezes é que o Gil falou de correr?" -> CMD_COUNT (every message has to be counted)
  "o Rafa diz muito isso" -> MIXED (an observation, not a request for a number)

A message is often a CONTINUATION of the recent conversation rather than a new
subject: a pronoun with no antecedent, an ellipsis, a challenge, a bare "e o X?".
Classify those by what the recent conversation is ABOUT, not by the isolated
sentence. Given a recent thread about the Bernardo:
  "e ele?" -> the mode the thread is in, not BANTER
  "muda a tua opinião, agora que percebeste?" -> FACTUAL (still about Bernardo)
  "porquê?" -> the mode the thread is in
  "e de bater na mãe?" after "quem tinha mais probabilidade de virar monge?" -> ROAST
A message that starts a genuinely new subject is classified on its own, even if
the recent conversation was about something else.

But most of a group chat is people reacting to each other, and a reaction stays
BANTER however much the thread around it is about somebody. Agreement, laughter,
a jab, a protest and a throwaway aside are social noise even mid-conversation, and
nothing has to be looked up to answer them. In a thread about the Gustavo being a
bad programmer:
  "É isso mesmo. É o chamado fala barato" -> BANTER (agreeing, not asking)
  "Conheço uns quantos ya" -> BANTER (an aside about nobody in particular)
  "Calma crl. Tava a elogiar te" -> BANTER (protesting at the bot)
  "ja sao amiguinhos e o crl" -> BANTER (a jab at the exchange itself)
  "o Gustavo até programava bem no outro projeto" -> MIXED (an actual claim about him)
MIXED needs the message to say something ABOUT a person or an episode. If the
subject only exists in the lines above it, it is BANTER.

Then, on a SECOND line, write "Q: " followed by the message rewritten as a
standalone question or request, with every pronoun and ellipsis resolved from the
recent conversation, naming the people it is about. This is used to search the
group's memory, so it must stand on its own without the conversation:
  "e ele?" -> Q: o Bernardo também faz isso?
  "e de bater na mãe?" -> Q: quem do grupo tinha maior probabilidade de bater na mãe?
  "quem é o Peter?" -> Q: quem é o Peter?
  "do que se trata a minha start up?" -> Q: do que se trata a startup do Pedro?
Omit the Q: line entirely when there is nothing to look up — for BANTER and for
pure commands. Do not invent a question that was not asked:
  "Conheço uns quantos ya" -> no Q: line (nobody asked anything)
  "Ahahhha" -> no Q: line

Reply with the token, and the Q: line when there is one. Nothing else."""


@dataclass
class Route:
    """The routing decision for one inbound message.

    ``query`` is the message rewritten to stand on its own, used for retrieval
    and for working out who the turn is about. Empty when the router did not
    supply one, in which case every caller falls back to the raw message.
    """

    mode: str
    command: Optional[str] = None
    raw: str = ""
    fallback: bool = False
    query: str = ""
    reconciled_from: str = ""

    @property
    def retrieval_enabled(self) -> bool:
        return self.mode not in (BANTER, GENERAL)


def _router_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return (config.get("chat", {}) or {}).get("router", {}) or {}


def mode_config(config: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Per-mode settings (retrieval / length / prompt override) from config.yaml."""
    modes = (config.get("chat", {}) or {}).get("modes", {}) or {}
    return modes.get(mode, {}) or {}


_QUERY_RE = re.compile(r"^\s*Q\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# A rewrite longer than this is the model answering the question instead of
# restating it, and feeding a paragraph to the embedder is worse than feeding the
# original message.
_MAX_QUERY_WORDS = 40


def _parse_query(text: str) -> str:
    """The standalone rewrite, or "" when the router did not give a usable one."""
    match = _QUERY_RE.search(text or "")
    if not match:
        return ""
    query = " ".join(match.group(1).split())
    if not query or len(query.split()) > _MAX_QUERY_WORDS:
        return ""
    return query


def _parse(text: str) -> Optional[Route]:
    """Pull a known label out of the model's output, tolerating stray tokens.

    The label scan runs over the first line only. It used to see the whole
    output, which was safe when the output WAS the label; now that a ``Q:`` line
    follows it, a rewrite like "quem é o mais burro do grupo?" would otherwise
    let a stray word in the restated question outvote the label the model chose.
    """
    if not text:
        return None
    head = text.strip().splitlines()[0] if text.strip() else ""
    upper = head.upper()
    # Longest labels first so CMD_TEXT is not shadowed by a bare TEXT match.
    for label in sorted(_LABELS, key=len, reverse=True):
        if re.search(rf"\b{label}\b", upper):
            mode, command = _LABELS[label]
            return Route(mode=mode or FACTUAL, command=command, raw=text.strip(),
                         query=_parse_query(text))
    return None


def reconcile(route: Route, named_in_message: Sequence[str],
              named_in_recent: Sequence[str]) -> Route:
    """Correct a GENERAL that is really the middle of a thread about a member.

    `general` exists to answer the world with no group retrieval and no member
    profiles, and its prompt says so outright: "não menciones membros do grupo".
    The live logs show it firing 11 times on turns that named one anyway. The
    clearest case is a follow-up: a thread about the Bernardo, then "Muda a tua
    opinião, agora que entendeste que é o bana?" — classified GENERAL, answered
    entirely about Bernardo, with retrieval switched off.

    Deliberately narrow and stateless. A member named in the message is not
    enough on its own: "o Gil também acha que o Ronaldo é melhor, e tu?" is a
    question about Ronaldo and must stay GENERAL. It only downgrades when that
    same member is ALREADY in the recent conversation, which is what makes the
    message a continuation rather than a new subject.

    `mixed` rather than `factual`: the turn is still chat, so it retrieves and
    keeps the member profiles but answers short.
    """
    if route.mode != GENERAL or route.command:
        return route
    in_recent = {str(name).lower() for name in named_in_recent}
    if not any(str(name).lower() in in_recent for name in named_in_message):
        return route
    return replace(route, mode=MIXED, reconciled_from=GENERAL)


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

    # Enough context that a follow-up is read as one. Two lines was not: a
    # thread about the Bernardo, then "Muda a tua opinião, agora que entendeste
    # que é o bana?", and the two lines in front of the router held the bot's own
    # last answer and nothing that said who the thread was about. Still bounded —
    # this call runs on every message and re-prefills from scratch.
    context = ""
    if recent_lines:
        window = max(1, int(rcfg.get("context_lines", 6)))
        context = "Recent conversation:\n" + "\n".join(recent_lines[-window:]) + "\n\n"

    messages = [
        {"role": "system", "content": _ROUTER_SYSTEM},
        {"role": "user", "content": f"{context}Message to classify:\n{text}"},
    ]
    try:
        raw = backend.generate(
            messages,
            max_new_tokens=int(rcfg.get("max_new_tokens", 48)),
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
