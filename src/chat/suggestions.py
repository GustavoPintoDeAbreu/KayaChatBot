"""
Follow-up question suggestions for the Kaya web UI.

After each answer, the already-loaded fine-tuned model is prompted a second time
to propose a few natural follow-up questions a group member might ask next,
grounded in the just-answered turn and the retrieved RAG context. This reuses the
in-memory model/tokenizer — no extra model load and no external API call.
"""
from typing import Any, Dict, List

import torch

# Instruction used to coax the local model into emitting short follow-up
# questions, one per line. Kept in European Portuguese to match the bot persona.
_SUGGESTION_SYSTEM_PROMPT = (
    "És um assistente que sugere perguntas de seguimento curtas e naturais sobre "
    "o grupo de amigos Kaya. Com base na conversa e no contexto, propõe perguntas "
    "que um membro do grupo poderia fazer a seguir. Responde APENAS com as "
    "perguntas, uma por linha, sem numeração nem texto extra. Cada pergunta deve "
    "ser curta (máx. 12 palavras) e terminar com '?'."
)

# Leading list markers to strip from each generated line.
_LIST_PREFIXES = ("- ", "* ", "• ", "Q:", "q:", "P:", "p:")


def parse_suggestions(raw: str, count: int = 3) -> List[str]:
    """Parse raw model output into a clean, deduplicated list of questions.

    Drops empty/malformed lines, strips numbering/bullets, keeps only lines that
    look like questions, and caps the result at ``count``. Pure function — no
    model dependency, so it is unit-testable in isolation.
    """
    if not raw:
        return []

    seen = set()
    questions: List[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue

        # Strip a leading "1.", "2)", "3 -" style enumeration.
        while text and (text[0].isdigit()):
            stripped = text.lstrip("0123456789").lstrip()
            if stripped.startswith((".", ")", "-", ":")):
                stripped = stripped[1:].lstrip()
            if stripped == text:
                break
            text = stripped

        for prefix in _LIST_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        text = text.strip("\"'“”").strip()
        if not text or "?" not in text:
            continue
        # Keep only up to the first question mark's sentence to avoid trailing junk.
        text = text[: text.index("?") + 1].strip()

        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        questions.append(text)
        if len(questions) >= count:
            break

    return questions


def generate_suggestions(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    user_message: str,
    assistant_response: str,
    context: str = "",
    count: int = 3,
) -> List[str]:
    """Generate up to ``count`` follow-up questions using the local model.

    Returns an empty list on any failure so the UI degrades gracefully (no chips).
    """
    sug_cfg = config.get("chat", {}).get("suggestions", {})
    if not sug_cfg.get("enabled", True):
        return []

    count = sug_cfg.get("count", count)
    max_new_tokens = sug_cfg.get("max_new_tokens", 64)
    temperature = sug_cfg.get("temperature", 0.7)

    context_block = f"=== Contexto ===\n{context}\n\n" if context else ""
    user_prompt = (
        f"{context_block}"
        f"Pergunta do utilizador: {user_message}\n"
        f"Resposta do bot: {assistant_response}\n\n"
        f"Sugere {count} perguntas de seguimento, uma por linha."
    )

    messages = [
        {"role": "system", "content": _SUGGESTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        device = getattr(model, "device", None) or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        inputs = tokenizer(text=[prompt], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                use_cache=True,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    except Exception as exc:  # noqa: BLE001 — suggestions are best-effort
        print(f"⚠️  Suggestion generation failed: {exc}")
        return []

    return parse_suggestions(raw, count)
