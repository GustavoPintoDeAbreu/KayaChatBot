"""Post-processing helpers for generated chat responses.

Kept dependency-free so it can be imported and unit-tested without loading the
model stack, and reused by every chat entry point (chat.py, web_app.py).
"""

import random
import re
from typing import Optional

# Cues that a question is asking for an elaborate answer rather than a quick reply.
# Used to raise the generation length budget only when warranted (see
# ``wants_long_answer``). Kept conservative so normal chit-chat stays short.
_LONG_ANSWER_CUES = (
    "explica",
    "explicar",
    "descreve",
    "descrever",
    "conta",
    "detalhe",
    "detalhada",
    "pormenor",
    "lista",
    "enumera",
    "resume",
    "resumo",
    "porque",
    "porquê",
    "explain",
    "describe",
    "detail",
    "list",
    "summar",
    "why",
    "elaborate",
)


def wants_long_answer(text: str, long_word_threshold: int = 30) -> bool:
    """Heuristic: does this message ask for an elaborate/long answer?

    True when the message contains an elaboration cue (``explica``, ``descreve``,
    ``lista``, ``why`` …) or is itself long (a detailed question tends to want a
    detailed answer). Otherwise False → the caller keeps replies short and chatty.
    Mirrors the lightweight keyword approach used by ``_has_temporal_intent`` in
    the retriever.
    """
    if not text:
        return False
    lowered = text.lower()
    if any(cue in lowered for cue in _LONG_ANSWER_CUES):
        return True
    return len(text.split()) >= long_word_threshold


# Asking the bot to actually think about it, rather than answer off the cuff.
# Deliberately explicit phrases only: this buys a second generation, and a false
# positive doubles the latency of a path already at 8-16s in a busy group.
_REASONING_CUES = (
    "pensa bem",
    "pensa melhor",
    "pensa nisso",
    "pensa lá bem",
    "com calma",
    "a sério agora",
    "justifica",
    "fundamenta",
    "argumenta",
    "com detalhe",
    "em detalhe",
    "think hard",
    "think carefully",
    "think about it",
    "really hard",
    "justify",
    "in detail",
    "make a case",
)


def wants_reasoning(text: str) -> bool:
    """Whether the message explicitly asks the bot to think the answer through.

    Asked for twice: once as a /feedback note ("reasoning router, if requested by
    the user use further intent solver to think question through"), and once in
    the group, where "before you go, try one last time. Really hard and detailed"
    got a single throwaway line back.

    A trigger, not a judgement. The router does not decide this on its own —
    every firing costs an extra generation while the GPU lock is held, and
    whatsapp_server drops an inbound message on a contended lock rather than
    queueing it.
    """
    if not text:
        return False
    lowered = text.lower()
    return any(cue in lowered for cue in _REASONING_CUES)


def truncate_history_line(line: str, max_words: int = 40) -> str:
    """Shorten one ``"<who>: <text>"`` history line to its first ``max_words``.

    Prior bot turns can be ~200-word paragraphs; pasted back verbatim as
    ``"Conversa recente:"`` context they invite the model to copy them wholesale
    (the observed "stuck" / repetition bug). Truncating to a gist keeps the
    speaker label and enough context for continuity without handing the model a
    block to regurgitate. The ``"<who>: "`` prefix is preserved and not counted.
    """
    if not line:
        return line
    who, sep, body = line.partition(": ")
    if not sep:  # no label — treat the whole line as body
        who, body = "", line
    words = body.split()
    if len(words) <= max_words:
        return line
    snippet = " ".join(words[:max_words]) + " …"
    return f"{who}{sep}{snippet}" if sep else snippet


def previous_bot_replies(recent_lines, bot_name: str = "Kaya Bot", limit: int = 4):
    """The bot's own last replies, newest first, pulled out of the history lines.

    ``recent_lines`` is the same ``"<who>: <text>"`` list the prompt is built
    from, so this needs no extra plumbing — the replies are already there.
    """
    prefix = f"{bot_name}: "
    replies = [line[len(prefix):].strip() for line in (recent_lines or [])
               if line.startswith(prefix)]
    return [reply for reply in reversed(replies) if reply][:limit]


def _normalize_for_comparison(text: str) -> str:
    return re.sub(r"[^\w\s]", "", (text or "").lower()).strip()


def is_near_duplicate(text: str, previous, threshold: float = 0.9) -> bool:
    """Whether ``text`` repeats something the bot already said.

    ``repetition_penalty`` and ``no_repeat_ngram_size`` only act WITHIN one
    generation, so they cannot see the previous turn at all. The live log shows
    what that costs: "So attack" and, one turn later, "Godamn" both received the
    byte-identical "Escolhe um alvo e diz quem eu tenho de partir primeiro."

    Similarity rather than equality, because the near-misses read just as badly
    as the exact ones.
    """
    from difflib import SequenceMatcher

    candidate = _normalize_for_comparison(text)
    if not candidate:
        return False
    for earlier in previous or []:
        other = _normalize_for_comparison(earlier)
        if not other:
            continue
        if candidate == other:
            return True
        if SequenceMatcher(None, candidate, other).ratio() >= threshold:
            return True
    return False


def coerce_text(content) -> str:
    """Flatten a chat message ``content`` into a plain string.

    Gradio 6.x uses a multimodal message format whose ``content`` can be a string,
    a dict like ``{"type": "text", "text": "…"}``, or a list of such parts. When a
    suggestion chip is clicked the value round-trips as one of those structured
    forms, and reading it directly rendered the raw ``[{'text': …, 'type': 'text'}]``
    to the user (and into the interaction log). Normalize every shape to text here.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", "") or "")
    if isinstance(content, (list, tuple)):
        return " ".join(coerce_text(part) for part in content).strip()
    return str(content)


# European-Portuguese vs English detection markers. Deliberately curated to be
# language-distinctive (ambiguous tokens like "a"/"do"/"me" are excluded) so a short
# WhatsApp line still classifies. Any Portuguese diacritic short-circuits to "pt".
_PT_WORDS = frozenset((
    "que", "não", "nao", "é", "de", "da", "quem", "com", "para", "uma", "um", "no", "na",
    "sou", "és", "tu", "você", "está", "muito", "obrigado", "olá", "ola", "porque", "como",
    "onde", "quando", "sim", "isto", "isso", "ele", "ela", "meu", "minha", "tens", "tem", "épá",
))
_EN_WORDS = frozenset((
    "the", "is", "are", "you", "what", "who", "can", "does", "and", "of", "tell", "about",
    "your", "hey", "how", "where", "when", "why", "please", "thanks", "thank", "my", "we",
    "they", "he", "she", "it", "this", "that", "hello", "there", "only", "speak", "english",
))


def language_signal(text: str) -> Optional[str]:
    """Language of a text *only where there is evidence*: ``"en"``, ``"pt"`` or None.

    Lightweight + dependency-free: any Portuguese diacritic ⇒ "pt"; otherwise a
    distinctive-stopword count decides. None means the text carries no marker
    either way ("Absolutely brutal.") — the caller decides what to assume, which
    matters for per-sentence voice selection, where guessing Portuguese would read
    an English sentence with Portuguese phonetics.
    """
    if not text:
        return None
    lowered = text.lower()
    if any(ch in lowered for ch in "ãõçáéíóúâêà"):
        return "pt"
    tokens = re.findall(r"[a-zà-ÿ']+", lowered)
    pt = sum(tok in _PT_WORDS for tok in tokens)
    en = sum(tok in _EN_WORDS for tok in tokens)
    if en > pt:
        return "en"
    if pt > en:
        return "pt"
    return None


def detect_language(text: str) -> str:
    """Best-effort language of an incoming message: ``"en"`` or ``"pt"`` (default).

    Used to steer the reply language so an English message isn't answered in PT.
    """
    return language_signal(text) or "pt"


# Meta-narration / 4th-wall leaks the model occasionally emits as a leading sentence
# (e.g. "A Sofia está confusa porque o bot…", "previsão de IA", "o assistente mencionou
# erroneamente…"). Targeted narrowly at the observed phrasings to avoid eating real facts.
_META_SELF_RE = re.compile(
    r"\bo bot\b"
    r"|previs[ãa]o de ia\b"
    r"|\bmodelo de (?:linguagem|ia)\b"
    r"|\bo assistente\b.{0,40}\b(?:mencion|comet|disse|err|baralh)"
    r"|\benquanto (?:ia|assistente)\b",
    re.IGNORECASE,
)


def _strip_meta_narration(text: str, user_name: str) -> str:
    """Drop a *leading* meta-narration sentence (bot self-reference, or third-person
    narration of the asker like "A <user> está a tentar…"), keeping the rest of the
    reply verbatim (newlines intact). Conservative: only the first sentence, only when
    real content follows it — never blanks a reply, never reflows the text.
    """
    if not text:
        return text
    asker = (user_name or "").strip()
    # Only conversation-meta verbs (insiste/está confusa/está a tentar…), NOT factual ones
    # like "está a trabalhar" — so "O Gustavo está a trabalhar" isn't stripped when the asker
    # happens to be Gustavo.
    asker_re = (
        re.compile(
            rf"^[AO]\s+{re.escape(asker)}\b.*\b(?:insiste|pergunta|baralh\w*|quer\s+saber"
            rf"|est[áa]\s+(?:confus\w*|a\s+tentar|a\s+perguntar))",
            re.IGNORECASE,
        )
        if asker
        else None
    )
    body = text.lstrip()
    # First-sentence boundary: the earlier of the first sentence terminator or a newline.
    terminator = re.search(r"[.!?](?=\s|$)", body)
    end = terminator.end() if terminator else len(body)
    newline = body.find("\n")
    if newline != -1 and newline < end:
        end = newline
    first, rest = body[:end].strip(), body[end:].lstrip()
    is_leak = bool(_META_SELF_RE.search(first)) or (
        asker_re is not None and asker_re.search(first) is not None
    )
    return rest if (is_leak and rest) else text


# A dash used as punctuation: surrounded by whitespace, or trailing at end of line.
# Intra-word hyphens must survive untouched ("dá-me", "pt_PT-tugão", "Kontext-dev"),
# and a line-leading "- " is a list bullet, not a clause separator.
_CLAUSE_DASH_RE = re.compile(r"(?<=\S)[ \t]+[—–]|(?<=\S)[ \t]+-{1,2}(?=[ \t])")
# A dash left dangling at the end of a line separates nothing; drop it outright
# rather than leaving a trailing comma behind.
_TRAILING_DASH_RE = re.compile(r"[ \t]+[—–-]{1,2}[ \t]*$")


def strip_clause_dashes(text: str) -> str:
    """Replace dashes used to separate clauses with a comma.

    The group asked for this explicitly: no em dashes, commas at most. Prompting
    alone does not hold over a few hundred tokens, and it cannot reach the parts
    of a reply that were never generated (the canned acknowledgements), so the
    rule is enforced here as well.

    Only whitespace-delimited dashes are touched. ``dá-me``, ``pt_PT-tugão`` and
    a ``- `` bullet at the start of a line are left exactly as they are.
    """
    if not text:
        return text
    out = []
    for line in text.split("\n"):
        line = _TRAILING_DASH_RE.sub("", line)
        stripped = line.lstrip()
        # A bullet line: protect the marker, clean the rest.
        if stripped.startswith(("- ", "-\t", "— ", "– ")):
            lead = len(line) - len(stripped)
            out.append(line[: lead + 2] + _CLAUSE_DASH_RE.sub(",", line[lead + 2 :]))
            continue
        out.append(_CLAUSE_DASH_RE.sub(",", line))
    # "palavra , outra" can only come from the substitution above.
    return re.sub(r"\s+,", ",", "\n".join(out))


# Retrieval scaffolding, as the corpus and the prompt render it. Anchored to the
# start: a bracketed aside mid-sentence is the model's own writing, not a leak.
_RETRIEVAL_SCAFFOLD = re.compile(
    r"^\s*\[\s*(?:(?:á|a)udio|imagem|foto|v[íi]deo|a\s+responder\s+a)\b[^\]]*\]\s*",
    re.IGNORECASE,
)


def clean_response(text: str, user_name: str, bot_name: str = "Kaya Bot") -> str:
    """Clean a raw generated response.

    The model is trained on short third-person observations but can run past its
    own turn and start speaking as another participant. Trim that hallucinated
    continuation *without* discarding legitimate multi-line answers — the old
    ``text.split("\\n")[0]`` truncation threw away everything after the first
    newline, so any multi-sentence answer was silently lost from history and the
    interaction log.

    Behaviour:
      1. Strip an echoed leading speaker label on the first line (e.g. the model
         prefixing its answer with ``"Kaya Bot:"`` or ``"<user>:"``).
      2. Cut at the first line where the model starts a *new user turn*
         (``"<user>:"``, ``"User:"``, ``"Utilizador:"``) — a hallucinated
         continuation — while preserving every line before it.
      3. Replace clause-separating dashes with commas (see
         ``strip_clause_dashes``).
    """
    if not text:
        return ""

    cleaned = text.strip()

    # 1. Drop an echoed leading "<name>:" label if the model prefixed its answer.
    for label in (f"{bot_name}:", f"{user_name}:"):
        if cleaned.lower().startswith(label.lower()):
            cleaned = cleaned[len(label):].lstrip()
            break

    # 1b. Drop a leading stage-direction the model sometimes echoes from the prompt,
    #     e.g. "[reply as Gustavo] …" / "[responde como Gustavo] …".
    cleaned = re.sub(
        r"^\[\s*(?:reply as|responde como|respond as)\b[^\]]*\]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 1d. Drop a retrieval wrapper the model copied out of its own context.
    #     The corpus renders past media as "[Áudio enviado por X em DATA] …"
    #     (see scripts/ingest_media.py) and replies as "[a responder a X: …]".
    #     Those are scaffolding for the model to read, never something to say —
    #     and a voice reply beginning "[Áudio enviado por Kaya Bot]" was read out
    #     loud, word for word, by Piper.
    cleaned = _RETRIEVAL_SCAFFOLD.sub("", cleaned).lstrip()

    # 1c. Drop a leading meta-narration leak ("A <user> está a tentar…", "o bot…").
    cleaned = _strip_meta_narration(cleaned, user_name)

    # 2. Cut at the first hallucinated user turn, keeping all prior lines.
    user_turn_labels = [f"{user_name}:", "User:", "Utilizador:"]
    kept_lines = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if any(stripped.startswith(label) for label in user_turn_labels):
            break
        kept_lines.append(line)

    return strip_clause_dashes("\n".join(kept_lines).strip())


def build_member_prompt_suffix(members_data: dict, shuffle: bool = False,
                               max_facts: int = 0, sample_facts: bool = False) -> str:
    """Build the group-members system-prompt suffix from a loaded
    group_members.json dict. Returns "" when there are no members.

    Shared by chat.py, web_app.py and the benchmark so every entry point injects
    the same member knowledge and can't drift apart. Each member contributes its
    aliases plus its curated ``key_facts`` (falling back to ``notes``) so the model
    actually has the member details at inference time — not just names. The phrasing
    is deliberately conversational (no "(também conhecido como: …)" template) so the
    model doesn't echo a typed-looking list back at the user.

    ``shuffle`` randomizes the member order each call. The live inference path sets
    this so no single member is always listed first (the early/first-mention slot
    gets disproportionate model attention, which fed the "favours one member" bias);
    deterministic callers (benchmark, training-data generation) leave it False.

    ``max_facts`` caps how many key_facts each member contributes, because
    shuffling only fixed the ORDER, not the amount. The profiles are wildly
    uneven — 6 facts for the most-discussed member down to 1 for the quietest —
    and "pick someone from the group" reliably resolves to whoever the prompt
    has the most material about. 0 means no cap (the previous behaviour).

    ``sample_facts`` picks that handful at random instead of taking the first N.
    It is for open-ended answers only — a roast, an opinion, an insult — where
    the same facts every turn produce the same joke every turn. A factual answer
    must never sample: "o que faz o Gil?" cannot depend on whether his job
    survived the draw.
    """
    members = list(members_data.get("members", []))
    if shuffle:
        random.shuffle(members)
    lines = []
    for member in members:
        name = member["name"]
        aliases = [a for a in member.get("aliases", []) if a.lower() != name.lower()]
        line = f"- {name}"
        if aliases:
            line += f" (também lhe chamam {', '.join(aliases)})"

        key_facts = member.get("key_facts") or []
        if max_facts > 0:
            # Truncating takes the SAME first N every time, which is how a roast
            # became a recital: Peter has five facts, the prompt showed all five
            # on every turn, and four separate roasts across three days all
            # reached for Rotterdam, editing other people's videos and Five Guys.
            # Sampling gives the model a different handful to work from — ten
            # possible triples out of five facts, in a different order each time.
            # Deterministic callers (benchmark, training data) keep the truncation.
            key_facts = (random.sample(key_facts, min(max_facts, len(key_facts)))
                         if sample_facts else key_facts[:max_facts])
        notes = member.get("notes", "")
        if key_facts:
            line += ": " + " ".join(
                fact.rstrip(".") + "." for fact in key_facts if fact.strip()
            )
        elif notes:
            sentences = [s.strip() for s in notes.split(".") if s.strip()]
            if sentences:
                line += ": " + ". ".join(sentences[:3]) + "."
        lines.append(line)

    if not lines:
        return ""

    intro = (
        "\n\nO que sabes sobre cada membro do grupo Kaya (usa isto para responder, "
        "incluindo palpites e avaliações sobre o grupo; fala deles de forma natural, "
        "não como uma lista formatada). Cada facto pertence à pessoa na cuja linha "
        "está e a mais ninguém: nunca atribuas a alguém um facto que está listado "
        "noutra pessoa. Se não souberes algo sobre quem te perguntam, di-lo, em vez "
        "de usares o que sabes de outro membro:\n"
    )
    # The roster has to state that it is complete. Asked "a que Kaya-Avenger devo
    # ligar?", the bot answered "liga à Mel" — Mel is not in the group and never
    # was. The list named the members but never said they were the only ones, so
    # any name that turned up in a retrieved chunk was fair game.
    names = ", ".join(member["name"] for member in members)
    closing = (
        f"\n\nO grupo Kaya tem {len(members)} membros e são exatamente estes: {names}. "
        "Mais ninguém é do grupo. Se aparecer outro nome nas conversas é alguém de "
        "fora, e nunca o deves tratar como membro. Não inventes membros que não "
        "estejam nesta lista."
        # These facts are accumulated over years of chat and carry no dates, so
        # the model presents all of them as current. Challenged on calling
        # somebody the group's "ladies man", it answered that the label was "um
        # título honorífico que ficou registado na memória coletiva" — which is
        # an accurate description of the data and a bad way to talk about people.
        "\n\nEstes perfis são o acumulado de anos de conversa, não uma fotografia "
        "de agora. Uma etiqueta antiga pode já não valer: se só a viste em "
        "conversas antigas, trata-a como história (\"durante uns tempos foi...\") "
        "em vez de a apresentares como o que a pessoa é hoje, e dá prioridade ao "
        "que aparece nas conversas recentes."
    )
    return intro + "\n".join(lines) + closing
