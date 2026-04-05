"""
Direct Training Data Formatter

Converts extracted WhatsApp/Instagram messages into ShareGPT-formatted
training examples without any API calls.

Each example is a sliding-window conversation turn:
  - user:      the last N messages in the session (as "[Sender]: text" lines)
  - assistant: the next message text (any sender)

The source is tagged "synthetic_kaya" so merge_datasets.py injects the
Kaya system prompt and handles it correctly.
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
MIN_RESPONSE_CHARS = 10   # Skip responses shorter than this
RANDOM_SEED = 3407

# ---------------------------------------------------------------------------
# Identity-leak filter: skip any response that looks like first-person member
# claims (these would teach the bot to impersonate group members).
# ---------------------------------------------------------------------------
_IDENTITY_LEAK_RE = re.compile(
    r"\bmeu\s+(amigo|colega|parceiro)\b"         # "meu amigo"
    r"|\bvivemos\s+juntos\b"                       # "vivemos juntos"
    r"|\bj[aá]\s+vivemos\b"                        # "já vivemos"
    r"|\bconhe[cç]o.{0,20}\bdesde\b"              # "conheço-o desde"
    r"|\bsomos\s+amigos\s+desde\b"                 # "somos amigos desde"
    r"|\bnos\s+conhecemos\s+h[aá]\b"               # "nos conhecemos há"
    r"|\b(fui|fomos)\s+(ao|para|com).{0,20}\bele\b",  # "fui com ele"
    re.IGNORECASE,
)


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
    """Format context messages using the same RAG markers used at inference."""
    lines = [f"[{m['sender']}]: {m['text']}" for m in messages]
    body = "\n".join(lines)
    return f"=== Conversas relevantes do grupo ===\n{body}\n=== Fim das conversas ==="


def create_examples(sessions: List[List[Dict]]) -> List[Dict]:
    """
    Build one ShareGPT example per sliding window position.

    Format understood by SyntheticDatasetMerger.format_conversation():
        {
          "conversations": [
              {"role": "user",      "content": "<context>"},
              {"role": "assistant", "content": "<response>"}
          ],
          "source": "synthetic_kaya"
        }
    """
    examples: List[Dict] = []

    for session in sessions:
        for i in range(CONTEXT_WINDOW, len(session)):
            context_msgs = session[i - CONTEXT_WINDOW:i]
            response_msg = session[i]

            # Skip very short responses — they add little training signal
            if len(response_msg['text']) < MIN_RESPONSE_CHARS:
                continue

            context_text = format_context(context_msgs)
            response_text = response_msg['text']

            # Skip responses that contain first-person member claims — these
            # would teach the bot to speak as a member rather than as a bot.
            if _IDENTITY_LEAK_RE.search(response_text):
                continue

            example = {
                "conversations": [
                    {
                        "role": "user",
                        "content": context_text
                    },
                    {
                        "role": "assistant",
                        "content": response_text
                    }
                ],
                "source": "synthetic_kaya"
            }
            examples.append(example)

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("📊 DIRECT TRAINING DATA FORMATTER")
    print("=" * 60)
    print(f"\nInput:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        print(f"\n❌ Input file not found: {INPUT_FILE}")
        print("   Please run extract_all_messages.py first.")
        return

    # Load
    print(f"\n📂 Loading messages...")
    messages = load_messages(INPUT_FILE)
    print(f"✅ Loaded {len(messages):,} messages")

    # Split into sessions
    print(f"\n🔄 Splitting into sessions (gap > {SESSION_GAP_MINUTES} min, "
          f"min length {MIN_SESSION_MSGS})...")
    sessions = split_into_sessions(messages)
    total_session_msgs = sum(len(s) for s in sessions)
    print(f"✅ {len(sessions):,} sessions  ({total_session_msgs:,} messages kept)")

    # Build examples
    print(f"\n🔧 Creating examples (context window = {CONTEXT_WINDOW} messages)...")
    examples = create_examples(sessions)
    print(f"✅ {len(examples):,} training examples created")

    if not examples:
        print("\n❌ No examples created — check SESSION_GAP_MINUTES and CONTEXT_WINDOW settings.")
        return

    # Shuffle for good measure
    random.seed(RANDOM_SEED)
    random.shuffle(examples)

    # Save
    OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)
    print(f"\n💾 Saving to {OUTPUT_FILE.name}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"✅ Saved {len(examples):,} examples")

    # Quick stats
    avg_ctx = sum(
        len(ex['conversations'][0]['content']) for ex in examples
    ) / len(examples)
    avg_resp = sum(
        len(ex['conversations'][1]['content']) for ex in examples
    ) / len(examples)
    print(f"\n📊 Avg context length : {avg_ctx:.0f} chars")
    print(f"   Avg response length: {avg_resp:.0f} chars")

    print(f"\n✅ Done!  Next: python src/data/merge_datasets.py")


if __name__ == "__main__":
    main()
