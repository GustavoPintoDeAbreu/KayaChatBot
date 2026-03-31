"""
generate_knowledge_base.py

Uses Azure GPT-4.1-mini to extract biographical facts about Kaya group members
from the cleaned message history (all_messages_cleaned.jsonl).

The script:
  1. Chunks all_messages_cleaned.jsonl into ~2000-token segments.
  2. For each chunk, sends it to the LLM with the current known bios and asks
     for updated/new facts about each mentioned member.
  3. Merges returned facts into group_members.json (notes field) and
     group_knowledge.json (text field for each member entry).
  4. Checkpoints every N chunks so it can be resumed with --resume-from.

CLI flags:
  --test           Process only the first 3 chunks (quick smoke test).
  --resume-from N  Skip the first N chunks (resume after crash/partial run).

Usage:
  python src/data/generate_knowledge_base.py
  python src/data/generate_knowledge_base.py --test
  python src/data/generate_knowledge_base.py --resume-from 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MESSAGES_FILE = DATA_DIR / "all_messages_cleaned.jsonl"
GROUP_MEMBERS_FILE = DATA_DIR / "group_members.json"
GROUP_KNOWLEDGE_FILE = DATA_DIR / "group_knowledge.json"

# ---------------------------------------------------------------------------
# Azure config (matches the project's known endpoint / model)
# ---------------------------------------------------------------------------
AZURE_ENDPOINT = "https://kaya-openai.openai.azure.com/"
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_MODEL = "gpt-4.1-mini"
AZURE_MAX_TOKENS = 4096
AZURE_TEMPERATURE = 0.3  # Low temp — factual extraction, not creative writing

# ---------------------------------------------------------------------------
# Chunking config
# ---------------------------------------------------------------------------
CHUNK_SIZE_TOKENS = 2000   # Approximate; we use word-count as proxy
WORDS_PER_TOKEN = 0.75     # ~1.33 tokens/word → 1500 words ≈ 2000 tokens
CHUNK_SIZE_WORDS = int(CHUNK_SIZE_TOKENS * WORDS_PER_TOKEN)

# ---------------------------------------------------------------------------
# Extraction prompt template
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """You are a meticulous archivist. Your task is to extract factual biographical
information about members of a Portuguese friend group called "Kaya" from chat message logs.

For each group member mentioned in the provided conversation chunk, extract:
- Personal facts (job, education, hobbies, interests)
- Events they were involved in (trips, parties, milestones)
- Relationships / dynamics with other members
- Opinions, preferences, or recurring themes

Return ONLY a valid JSON object (no markdown, no explanation) with this structure:
{
  "members": {
    "MemberName": "Concise factual biography in 1-3 sentences. Only include what is clearly supported by the chat.",
    "AnotherMember": "..."
  },
  "recent_summaries": {
    "MemberName": "Short paragraph (1-2 sentences) about their most recent discussion topics, opinions, or notable events in this chunk.",
    "AnotherMember": "..."
  }
}

Rules:
- Only include members that appear in this specific chunk.
- Do NOT invent facts. Only use what is clearly stated or implied in the messages.
- Use English for the biographies and recent summaries.
- Keep each biography concise: 1-3 sentences max.
- Keep each recent_summary concise: 1-2 sentences capturing the most recent activity or topics.
- If nothing new or useful is found for a member, omit them from the output entirely.
"""


def load_azure_client() -> AzureOpenAI:
    """Initialise the Azure OpenAI client using project-specific env var."""
    load_dotenv()
    api_key = os.getenv("AZURE_OPENAI_API_KEY_gpt_41_mini")
    if not api_key:
        print(
            "ERROR: AZURE_OPENAI_API_KEY_gpt_41_mini not found in environment. "
            "Ensure it is set in your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)
    return AzureOpenAI(
        api_key=api_key,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )


def load_messages() -> List[Dict]:
    """Load all cleaned messages from JSONL file."""
    messages = []
    with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def chunk_messages(messages: List[Dict], chunk_size_words: int) -> List[List[Dict]]:
    """Split messages into chunks of approximately chunk_size_words words."""
    chunks: List[List[Dict]] = []
    current_chunk: List[Dict] = []
    current_words = 0

    for msg in messages:
        word_count = len(msg.get("text", "").split())
        if current_words + word_count > chunk_size_words and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_words = 0
        current_chunk.append(msg)
        current_words += word_count

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def format_chunk_for_prompt(messages: List[Dict]) -> str:
    """Convert a list of message dicts into a readable transcript."""
    lines = []
    for msg in messages:
        ts = msg.get("timestamp", "")[:16]  # "YYYY-MM-DDTHH:MM"
        sender = msg.get("sender", "Unknown")
        text = msg.get("text", "").strip()
        if text:
            lines.append(f"[{ts}] {sender}: {text}")
    return "\n".join(lines)


def get_mentioned_members(messages: List[Dict], member_aliases: Dict[str, List[str]]) -> List[str]:
    """Return member names that appear (as sender or mentioned) in a chunk."""
    mentioned = set()
    all_text = " ".join(
        (msg.get("sender", "") + " " + msg.get("text", "")).lower()
        for msg in messages
    )
    for name, aliases in member_aliases.items():
        for alias in aliases:
            if alias.lower() in all_text:
                mentioned.add(name)
                break
    return sorted(mentioned)


def build_extraction_prompt(
    chunk_text: str,
    current_bios: Dict[str, str],
    mentioned_members: List[str],
) -> str:
    """Build the user message for the LLM extraction call."""
    bio_block = "\n".join(
        f"- {name}: {bio}" if bio else f"- {name}: (no bio yet)"
        for name, bio in current_bios.items()
        if name in mentioned_members
    )
    prompt = (
        f"=== Members mentioned in this chunk ===\n{', '.join(mentioned_members)}\n\n"
        f"=== Current known bios (for context, do not repeat unchanged info) ===\n{bio_block}\n\n"
        f"=== Chat messages ===\n{chunk_text}\n\n"
        "Extract and return updated/new factual notes for the members above."
    )
    return prompt


def call_azure_llm(
    client: AzureOpenAI,
    user_prompt: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> Optional[Dict]:
    """Call Azure GPT-4.1-mini and return parsed dict with 'members' and 'recent_summaries' or None on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=AZURE_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=AZURE_TEMPERATURE,
                max_tokens=AZURE_MAX_TOKENS,
            )
            content = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    line for line in lines if not line.startswith("```")
                ).strip()

            result = json.loads(content)
            members_dict = result.get("members", {})
            recent_summaries_dict = result.get("recent_summaries", {})
            if isinstance(members_dict, dict):
                return {
                    "members": members_dict,
                    "recent_summaries": recent_summaries_dict if isinstance(recent_summaries_dict, dict) else {},
                }

        except json.JSONDecodeError as e:
            print(f"  [attempt {attempt}] JSON parse error: {e}. Retrying...", flush=True)
        except Exception as e:
            print(f"  [attempt {attempt}] API error: {e}. Retrying in {retry_delay}s...", flush=True)
            if attempt < max_retries:
                time.sleep(retry_delay)

    print("  ⚠️  All retries exhausted for this chunk. Skipping.", flush=True)
    return None


def merge_bios(existing: str, new_info: str) -> str:
    """Merge new bio info with the existing bio, avoiding pure duplicates."""
    if not existing:
        return new_info.strip()
    if not new_info or new_info.strip() in existing:
        return existing
    # Append new sentences not already present
    existing_sentences = {s.strip() for s in existing.replace(". ", ".|").split("|")}
    new_sentences = [s.strip() for s in new_info.replace(". ", ".|").split("|") if s.strip()]
    additions = [s for s in new_sentences if s not in existing_sentences]
    if additions:
        return existing.rstrip(". ") + ". " + " ".join(additions)
    return existing


def save_group_members(members_data: Dict, bios: Dict[str, str], recent_summaries: Dict[str, str]) -> None:
    """Write updated notes and recent_summary to group_members.json."""
    for member in members_data["members"]:
        name = member["name"]
        if name in bios and bios[name]:
            member["notes"] = bios[name]
        if name in recent_summaries and recent_summaries[name]:
            member["recent_summary"] = recent_summaries[name]
    with open(GROUP_MEMBERS_FILE, "w", encoding="utf-8") as f:
        json.dump(members_data, f, ensure_ascii=False, indent=2)


def save_group_knowledge(knowledge_data: Dict, bios: Dict[str, str]) -> None:
    """Write updated text to group_knowledge.json for member entries."""
    for fact in knowledge_data["facts"]:
        name = fact.get("subject", "")
        if fact.get("category") == "member" and name in bios and bios[name]:
            fact["text"] = bios[name]
    with open(GROUP_KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_data, f, ensure_ascii=False, indent=2)


def main(test_mode: bool = False, resume_from: int = 0) -> None:
    """Main knowledge extraction pipeline."""
    print("=" * 60, flush=True)
    print("📚 KAYA KNOWLEDGE BASE GENERATOR", flush=True)
    if test_mode:
        print("   [TEST MODE — first 3 chunks only]", flush=True)
    if resume_from:
        print(f"   [RESUMING from chunk {resume_from}]", flush=True)
    print("=" * 60, flush=True)

    # Load data
    print("\n📂 Loading data...", flush=True)
    messages = load_messages()
    print(f"  Loaded {len(messages)} messages.", flush=True)

    with open(GROUP_MEMBERS_FILE, "r", encoding="utf-8") as f:
        members_data = json.load(f)
    with open(GROUP_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        knowledge_data = json.load(f)

    # Build lookup structures
    member_aliases: Dict[str, List[str]] = {
        m["name"]: m["aliases"] for m in members_data["members"]
    }
    # Also add sender name variants to aliases for detection
    sender_to_member: Dict[str, str] = {}
    for m in members_data["members"]:
        for alias in m["aliases"]:
            sender_to_member[alias.lower()] = m["name"]

    # Current bios (start from existing notes if resuming)
    current_bios: Dict[str, str] = {
        m["name"]: m.get("notes", "") for m in members_data["members"]
    }

    # Current recent summaries (always replaced with latest, not merged).
    # Messages are processed in chronological order, so later chunks naturally
    # produce more recent summaries that overwrite earlier ones.
    current_recent_summaries: Dict[str, str] = {
        m["name"]: m.get("recent_summary", "") for m in members_data["members"]
    }

    # Chunk messages
    chunks = chunk_messages(messages, CHUNK_SIZE_WORDS)
    total_chunks = len(chunks)
    print(f"  Split into {total_chunks} chunks (~{CHUNK_SIZE_WORDS} words each).", flush=True)

    if test_mode:
        chunks = chunks[:3]
        print(f"  Test mode: processing first {len(chunks)} chunks.", flush=True)

    if resume_from > 0:
        chunks = chunks[resume_from:]
        print(f"  Resuming: skipping first {resume_from} chunks.", flush=True)

    # Load Azure client
    print("\n🔗 Connecting to Azure OpenAI...", flush=True)
    client = load_azure_client()
    print("  Connected.", flush=True)

    # Config for checkpointing
    checkpoint_every = 5
    rate_limit_delay = 2.0  # seconds between API calls

    # Main loop
    print(f"\n🔄 Processing {len(chunks)} chunks...\n", flush=True)
    for i, chunk in enumerate(chunks):
        chunk_idx = i + resume_from
        print(f"[{chunk_idx + 1}/{total_chunks}] Chunk {chunk_idx}...", end=" ", flush=True)

        mentioned = get_mentioned_members(chunk, member_aliases)
        if not mentioned:
            print("no members mentioned, skipping.", flush=True)
            continue

        print(f"members: {', '.join(mentioned)}", flush=True)

        chunk_text = format_chunk_for_prompt(chunk)
        user_prompt = build_extraction_prompt(chunk_text, current_bios, mentioned)

        new_bios = call_azure_llm(client, user_prompt)
        if new_bios:
            for member_name, bio in new_bios["members"].items():
                # Normalise name: try exact match first, then partial
                matched_name = None
                if member_name in current_bios:
                    matched_name = member_name
                else:
                    for name in current_bios:
                        if name.lower() == member_name.lower():
                            matched_name = name
                            break

                if matched_name:
                    current_bios[matched_name] = merge_bios(current_bios[matched_name], bio)
                    print(f"    ✅ Updated bio for {matched_name}", flush=True)
                else:
                    print(f"    ⚠️  Unknown member '{member_name}' returned by LLM, skipping.", flush=True)

            # recent_summaries always replace (they represent the latest snapshot)
            for member_name, summary in new_bios["recent_summaries"].items():
                matched_name = None
                if member_name in current_recent_summaries:
                    matched_name = member_name
                else:
                    for name in current_recent_summaries:
                        if name.lower() == member_name.lower():
                            matched_name = name
                            break

                if matched_name and summary:
                    current_recent_summaries[matched_name] = summary.strip()
                    print(f"    ✅ Updated recent_summary for {matched_name}", flush=True)

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            print(f"  💾 Checkpoint after chunk {chunk_idx}...", flush=True)
            save_group_members(members_data, current_bios, current_recent_summaries)
            save_group_knowledge(knowledge_data, current_bios)

        # Rate limit
        if i < len(chunks) - 1:
            time.sleep(rate_limit_delay)

    # Final save
    print("\n💾 Saving final results...", flush=True)
    save_group_members(members_data, current_bios, current_recent_summaries)
    save_group_knowledge(knowledge_data, current_bios)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("✅ Knowledge base generation complete!", flush=True)
    print("\nBios generated:", flush=True)
    populated = 0
    for name, bio in current_bios.items():
        status = "✅" if bio else "⬜"
        print(f"  {status} {name}: {bio[:80] + '...' if len(bio) > 80 else bio or '(empty)'}", flush=True)
        if bio:
            populated += 1
    print(f"\n{populated}/{len(current_bios)} members have bios.", flush=True)
    print(f"\nFiles updated:", flush=True)
    print(f"  📄 {GROUP_MEMBERS_FILE}", flush=True)
    print(f"  📄 {GROUP_KNOWLEDGE_FILE}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kaya knowledge base from chat history.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only the first 3 chunks (quick smoke test).",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        metavar="N",
        help="Skip the first N chunks and resume from chunk N.",
    )
    args = parser.parse_args()
    main(test_mode=args.test, resume_from=args.resume_from)
