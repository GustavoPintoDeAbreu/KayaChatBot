"""Post-processing helpers for generated chat responses.

Kept dependency-free so it can be imported and unit-tested without loading the
model stack, and reused by every chat entry point (chat.py, web_app.py).
"""

import random
import re

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


def detect_language(text: str) -> str:
    """Best-effort language of an incoming message: ``"en"`` or ``"pt"`` (default).

    Lightweight + dependency-free: any Portuguese diacritic ⇒ "pt"; otherwise a
    distinctive-stopword count decides, defaulting to Portuguese on a tie/empty.
    Used to steer the reply language so an English message isn't answered in PT.
    """
    if not text:
        return "pt"
    lowered = text.lower()
    if any(ch in lowered for ch in "ãõçáéíóúâêà"):
        return "pt"
    tokens = re.findall(r"[a-zà-ÿ']+", lowered)
    pt = sum(tok in _PT_WORDS for tok in tokens)
    en = sum(tok in _EN_WORDS for tok in tokens)
    return "en" if en > pt and en > 0 else "pt"


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

    return "\n".join(kept_lines).strip()


def build_member_prompt_suffix(members_data: dict, shuffle: bool = False) -> str:
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
        "não como uma lista formatada):\n"
    )
    return intro + "\n".join(lines)
