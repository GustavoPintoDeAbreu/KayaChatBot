"""
Direct Training Data Formatter

Converts extracted WhatsApp/Instagram messages into ShareGPT-formatted
training examples without any API calls.

Each example is a sliding-window conversation turn:
  - user:      the last N messages formatted as a RAG context block, followed
               by an explicit question about what was said
  - assistant: a third-person bot observation that cites the next message as
               evidence — never in the voice of a group member

This mirrors the inference-time format used in chat.py and matches the
synthetic_targeted examples so merge_datasets.py handles both identically.
The source is tagged "synthetic_kaya" for merge_datasets.py.
"""

import json
import random
import re
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if os.path.exists('/app'):
    DATA_DIR = Path("/app/data")
else:
    DATA_DIR = Path(__file__).parent.parent.parent / "data"

INPUT_FILE = DATA_DIR / "all_messages_cleaned.jsonl"
OUTPUT_FILE = DATA_DIR / "synthetic_kaya.jsonl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONTEXT_WINDOW = 6        # How many preceding messages to use as context
SESSION_GAP_MINUTES = 30  # Silence gap that signals a new conversation session
MIN_SESSION_MSGS = 4      # Skip sessions shorter than this
MIN_RESPONSE_CHARS = 10   # Skip next messages shorter than this (too trivial to cite)
RANDOM_SEED = 3407

# ---------------------------------------------------------------------------
# Identity-leak filter: skip any next message whose raw text contains
# first-person member claims — even when cited in third person the quoted
# content would still confuse training.
# ---------------------------------------------------------------------------
_IDENTITY_LEAK_RE = re.compile(
    r"\bmeu\s+(amigo|colega|parceiro)\b"
    r"|\bvivemos\s+juntos\b"
    r"|\bj[aá]\s+vivemos\b"
    r"|\bconhe[cç]o.{0,20}\bdesde\b"
    r"|\bsomos\s+amigos\s+desde\b"
    r"|\bnos\s+conhecemos\s+h[aá]\b"
    r"|\b(fui|fomos)\s+(ao|para|com).{0,20}\bele\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Question bank — randomised per example to create variety.
# Mix of generic (what happened?) and sender-specific (what did X say?).
# Both PT and EN since the group uses both.
# Placeholder {sender} is filled at runtime.
# ---------------------------------------------------------------------------
_GENERIC_QUESTIONS = [
    "O que está a acontecer nesta conversa?",
    "Sobre o que está o grupo a falar?",
    "O que é que se passou nas mensagens?",
    "Podes resumir o que foi dito?",
    "O que discutiram os membros aqui?",
    "Que tópico está em discussão?",
    "O que se está a debater no grupo?",
    "What is the group talking about?",
    "Can you summarize this conversation?",
    "What was discussed in these messages?",
    "What is happening in this chat?",
]

_SENDER_QUESTIONS = [
    "O que disse {sender} a seguir?",
    "Qual foi a resposta de {sender}?",
    "O que mencionou {sender} nesta conversa?",
    "O que é que {sender} disse?",
    "What did {sender} say next?",
    "What did {sender} mention here?",
]

# ---------------------------------------------------------------------------
# Answer templates — third-person bot observations that cite the next message.
# ---------------------------------------------------------------------------
_ANSWER_TEMPLATES = [
    'De acordo com as conversas do grupo, {sender} disse: "{text}".',
    'Com base nas mensagens, {sender} mencionou: "{text}".',
    'Nas conversas do grupo, {sender} escreveu: "{text}".',
    'According to the group messages, {sender} said: "{text}".',
    'Based on the conversations, {sender} mentioned: "{text}".',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_messages(file_path: Path) -> List[Dict]:
    messages: List[Dict] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return messages


def split_into_sessions(messages: List[Dict]) -> List[List[Dict]]:
    """Group messages into conversation sessions separated by long silences."""
    if not messages:
        return []

    sessions: List[List[Dict]] = []
    session: List[Dict] = [messages[0]]

    for msg in messages[1:]:
        try:
            prev_time = datetime.fromisoformat(session[-1]['timestamp'])
            curr_time = datetime.fromisoformat(msg['timestamp'])
            gap_minutes = (curr_time - prev_time).total_seconds() / 60
            if gap_minutes > SESSION_GAP_MINUTES:
                if len(session) >= MIN_SESSION_MSGS:
                    sessions.append(session)
                session = [msg]
            else:
                session.append(msg)
        except (ValueError, KeyError):
            session.append(msg)

    if len(session) >= MIN_SESSION_MSGS:
        sessions.append(session)

    return sessions


def format_context(messages: List[Dict]) -> str:
    """Format context messages byte-identically to the inference-time retriever.

    Must mirror ``ConversationRetriever.format_context`` (src/chat/retriever.py):
    a "=== Conversas relevantes do grupo ===" header, a "--- Conversa 1 ---"
    sub-header, then ``{sender}: {text}`` lines (no brackets). Previously this
    used ``[{sender}]: {text}`` with no sub-header, so the model trained on a
    different context shape than it sees at inference (train/serve skew).
    """
    body = "\n".join(f"{m['sender']}: {m['text']}" for m in messages)
    return (
        "=== Conversas relevantes do grupo ===\n\n"
        "--- Conversa 1 ---\n"
        f"{body}\n\n"
        "=== Fim das conversas ==="
    )


def _pick_question(next_msg: Dict, rng: random.Random) -> str:
    """Pick a random question — roughly half generic, half sender-specific."""
    sender_pool = [q.format(sender=next_msg['sender']) for q in _SENDER_QUESTIONS]
    pool = _GENERIC_QUESTIONS + sender_pool
    return rng.choice(pool)


def _format_observation(next_msg: Dict, rng: random.Random) -> str:
    """Format the next message as a third-person bot observation."""
    template = rng.choice(_ANSWER_TEMPLATES)
    return template.format(sender=next_msg['sender'], text=next_msg['text'])


def create_examples(sessions: List[List[Dict]], rng: random.Random) -> List[Dict]:
    """
    Build one ShareGPT example per sliding window position.

    Format understood by SyntheticDatasetMerger.format_conversation():
        {
          "conversations": [
              {"role": "user",      "content": "<context>\n\nCom base nestas conversas passadas, responde:\n<question>"},
              {"role": "assistant", "content": "<third-person observation>"}
          ],
          "source": "synthetic_kaya"
        }
    """
    examples: List[Dict] = []

    for session in sessions:
        for i in range(CONTEXT_WINDOW, len(session)):
            context_msgs = session[i - CONTEXT_WINDOW:i]
            next_msg = session[i]

            if len(next_msg['text']) < MIN_RESPONSE_CHARS:
                continue

            if _IDENTITY_LEAK_RE.search(next_msg['text']):
                continue

            context_text = format_context(context_msgs)
            question = _pick_question(next_msg, rng)
            user_content = (
                f"{context_text}\n\n"
                f"Com base nestas conversas passadas, responde:\n"
                f"{question}"
            )
            assistant_content = _format_observation(next_msg, rng)

            examples.append({
                "conversations": [
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "source": "synthetic_kaya",
            })

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DIRECT TRAINING DATA FORMATTER")
    print("=" * 60)
    print(f"\nInput:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        print(f"\nInput file not found: {INPUT_FILE}")
        print("   Please run extract_all_messages.py first.")
        return

    print(f"\nLoading messages...")
    messages = load_messages(INPUT_FILE)
    print(f"Loaded {len(messages):,} messages")

    print(f"\nSplitting into sessions (gap > {SESSION_GAP_MINUTES} min, "
          f"min length {MIN_SESSION_MSGS})...")
    sessions = split_into_sessions(messages)
    total_session_msgs = sum(len(s) for s in sessions)
    print(f"{len(sessions):,} sessions  ({total_session_msgs:,} messages kept)")

    rng = random.Random(RANDOM_SEED)

    print(f"\nCreating examples (context window = {CONTEXT_WINDOW} messages)...")
    examples = create_examples(sessions, rng)
    print(f"{len(examples):,} training examples created")

    if not examples:
        print("\nNo examples created — check SESSION_GAP_MINUTES and CONTEXT_WINDOW settings.")
        return

    rng.shuffle(examples)

    OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {OUTPUT_FILE.name}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"Saved {len(examples):,} examples")

    avg_ctx = sum(len(ex['conversations'][0]['content']) for ex in examples) / len(examples)
    avg_resp = sum(len(ex['conversations'][1]['content']) for ex in examples) / len(examples)
    print(f"\nAvg context length : {avg_ctx:.0f} chars")
    print(f"Avg response length: {avg_resp:.0f} chars")

    print(f"\nDone!  Next: python src/data/merge_datasets.py")


if __name__ == "__main__":
    main()
