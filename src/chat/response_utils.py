"""Post-processing helpers for generated chat responses.

Kept dependency-free so it can be imported and unit-tested without loading the
model stack, and reused by every chat entry point (chat.py, web_app.py).
"""


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
    """Build the "Membros do grupo Kaya: ..." system-prompt suffix from a loaded
    group_members.json dict. Returns "" when there are no members.

    Shared by chat.py and web_app.py so the two entry points can't drift apart.
    """
    lines = []
    for member in members_data.get("members", []):
        line = member["name"]
        aliases = [a for a in member.get("aliases", []) if a.lower() != member["name"].lower()]
        if aliases:
            line += f" (também conhecido como: {', '.join(aliases)})"
        notes = member.get("notes", "")
        if notes:
            # Keep only the first 2 sentences to stay within the token budget.
            sentences = [s.strip() for s in notes.split(".") if s.strip()]
            line += f" — {'. '.join(sentences[:2])}."
        lines.append(line)
    if not lines:
        return ""
    return f"\n\nMembros do grupo Kaya: {'; '.join(lines)}."
