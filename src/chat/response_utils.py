"""Post-processing helpers for generated chat responses.

Kept dependency-free so it can be imported and unit-tested without loading the
model stack, and reused by every chat entry point (chat.py, web_app.py).
"""


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

    # 2. Cut at the first hallucinated user turn, keeping all prior lines.
    user_turn_labels = [f"{user_name}:", "User:", "Utilizador:"]
    kept_lines = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if any(stripped.startswith(label) for label in user_turn_labels):
            break
        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def build_member_prompt_suffix(members_data: dict) -> str:
    """Build the group-members system-prompt suffix from a loaded
    group_members.json dict. Returns "" when there are no members.

    Shared by chat.py, web_app.py and the benchmark so every entry point injects
    the same member knowledge and can't drift apart. Each member contributes its
    aliases plus its curated ``key_facts`` (falling back to ``notes``) so the model
    actually has the member details at inference time — not just names. The phrasing
    is deliberately conversational (no "(também conhecido como: …)" template) so the
    model doesn't echo a typed-looking list back at the user.
    """
    lines = []
    for member in members_data.get("members", []):
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
