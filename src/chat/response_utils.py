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
