"""
generate_knowledge_base.py

Uses an LLM provider (Azure OpenAI or xAI Grok, configured via
``generation.provider`` in config.yaml) to extract structured per-member
biographical profiles from the cleaned message history.

The script:
  1. Chunks all_messages_cleaned.jsonl into ~2000-token segments.
  2. For each chunk, sends it to the LLM and asks for structured profile
     fields (age, interests, occupation, etc.) for each mentioned member.
  3. Merges returned profiles into group_members.json and writes a legacy
     ``notes`` string (biography_summary) for backward compatibility.
  4. Adds ``topic_mapping`` fact entries to group_knowledge.json for each
     member's frequently discussed topics.
  5. Checkpoints every N chunks so it can be resumed with --resume-from.

Config controls:
  knowledge_base.profile_fields   — list of fields to extract (subset for testing)
  knowledge_generation.chunk_size_tokens
  knowledge_generation.checkpoint_every
  knowledge_generation.rate_limit_delay

CLI flags:
  --test           Process only the first N chunks (test_mode.generation.chunks_limit).
  --resume-from N  Skip the first N chunks (resume after crash/partial run).

Usage:
  python src/data/generate_knowledge_base.py
  python src/data/generate_knowledge_base.py --test
  python src/data/generate_knowledge_base.py --resume-from 10
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MESSAGES_FILE = DATA_DIR / "all_messages_cleaned.jsonl"
GROUP_MEMBERS_FILE = DATA_DIR / "group_members.json"
GROUP_KNOWLEDGE_FILE = DATA_DIR / "group_knowledge.json"
CONFIG_FILE = BASE_DIR / "config.yaml"

# ---------------------------------------------------------------------------
# Default field list (used when config section is absent)
# ---------------------------------------------------------------------------
DEFAULT_PROFILE_FIELDS = [
    "name",
    "age",
    "interests",
    "occupation",
    "living_place",
    "marital_status",
    "political_preference",
    "state_of_mind",
    "biography_summary",
    "frequently_discussed_topics",
]

# Fields that must NOT be embedded into ChromaDB (sensitive / private)
SENSITIVE_FIELDS = {"political_preference"}

# ---------------------------------------------------------------------------
# Extraction prompt template
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """You are a meticulous archivist. Your task is to extract structured biographical
profiles about members of a Portuguese friend group called "Kaya" from chat message logs.

For each group member mentioned in the provided conversation chunk, extract only what is
clearly supported by the messages. Do NOT invent facts.

Return ONLY a valid JSON object (no markdown, no explanation) with this structure:
{
  "members": {
    "MemberName": {
      "age": "approximate age or null",
      "interests": ["hobby1", "hobby2"],
      "occupation": "job or studies or null",
      "living_place": "city/country or null",
      "marital_status": "single/relationship/engaged/married or null",
      "political_preference": "political leaning in Portuguese spectrum or null",
      "state_of_mind": "brief mood/attitude from recent messages or null",
      "biography_summary": "2-3 sentence factual bio",
      "frequently_discussed_topics": ["topic1", "topic2", "topic3"]
    }
  },
  "recent_summaries": {
    "MemberName": "Short paragraph (1-2 sentences) about their most recent discussion topics, opinions, or notable events in this chunk.",
    "AnotherMember": "..."
  }
}

Rules:
- Only include members that appear in this specific chunk.
- Only include fields you have evidence for — use null for unknown fields.
- Do NOT invent facts. Only use what is clearly stated or implied in the messages.
- Use English for all text values.
- biography_summary: 2-3 sentences max, factual only.
- frequently_discussed_topics: up to 10 topics, single words or short phrases.
- recent_summary: 1-2 sentences capturing the most recent activity or topics for this chunk.
- If nothing new is found for a member, omit them entirely.
"""


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_profile_fields(config: Dict[str, Any]) -> List[str]:
    """Return the list of profile fields to extract from config."""
    return config.get("knowledge_base", {}).get("profile_fields", DEFAULT_PROFILE_FIELDS)


def load_provider(config: Dict[str, Any]):
    """Load the configured LLM provider (Azure or xAI)."""
    sys.path.insert(0, str(BASE_DIR / "src"))
    from llm_providers import get_provider  # noqa: PLC0415
    return get_provider(config)


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


def get_mentioned_members(
    messages: List[Dict], member_aliases: Dict[str, List[str]]
) -> List[str]:
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
    current_profiles: Dict[str, Dict],
    mentioned_members: List[str],
    profile_fields: List[str],
) -> str:
    """Build the user message for the LLM extraction call."""
    # Summarise existing profiles for context
    profile_lines = []
    for name in mentioned_members:
        existing = current_profiles.get(name, {})
        bio = existing.get("biography_summary") or existing.get("notes", "")
        if bio:
            profile_lines.append(f"- {name}: {bio[:200]}")
        else:
            profile_lines.append(f"- {name}: (no profile yet)")
    profile_block = "\n".join(profile_lines)

    # Build field restriction note if only a subset is requested
    fields_note = ""
    all_fields = set(DEFAULT_PROFILE_FIELDS) - {"name"}
    requested = set(profile_fields) - {"name"}
    if requested != all_fields:
        fields_note = (
            f"\nOnly extract the following fields: {', '.join(sorted(requested))}.\n"
        )

    return (
        f"=== Members mentioned in this chunk ===\n{', '.join(mentioned_members)}\n\n"
        f"=== Current known profiles (for context, do not repeat unchanged info) ===\n"
        f"{profile_block}\n"
        f"{fields_note}\n"
        f"=== Chat messages ===\n{chunk_text}\n\n"
        "Extract and return updated/new profile data for the members above."
    )


def strip_markdown_fences(content: str) -> str:
    """Remove markdown code fences from LLM output."""
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()
    return content


def call_llm_for_profiles(
    provider,
    user_prompt: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> Optional[Dict]:
    """Call the LLM provider and return parsed dict with 'members' (profile dicts) and 'recent_summaries' or None on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            content = provider.generate_text(EXTRACTION_SYSTEM_PROMPT, user_prompt)
            content = strip_markdown_fences(content)

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


def merge_list_field(existing: Optional[List[str]], new: Optional[List[str]]) -> Optional[List[str]]:
    """Merge two lists, deduplicating while preserving order."""
    if not new:
        return existing
    if not existing:
        return list(new)
    seen = set(existing)
    merged = list(existing)
    for item in new:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def merge_profiles(
    existing: Dict[str, Any], new_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge new profile data into the existing profile dict.

    - String fields: prefer new non-null values.
    - List fields (interests, frequently_discussed_topics): deduplicated union.
    - biography_summary: append new sentences not already present.
    - notes: kept as legacy alias for biography_summary.
    """
    merged = dict(existing)

    list_fields = {"interests", "frequently_discussed_topics"}

    for field, new_val in new_data.items():
        if new_val is None:
            continue  # Skip null values from LLM

        if field in list_fields:
            if isinstance(new_val, list):
                merged[field] = merge_list_field(merged.get(field), new_val)
        elif field == "biography_summary":
            old_bio = merged.get("biography_summary", "") or ""
            if not new_val or new_val.strip() in old_bio:
                pass  # Nothing new
            else:
                # Split on period followed by optional whitespace; also handles trailing period
                existing_sentences = {
                    s.strip() for s in re.split(r"\.\s*", old_bio) if s.strip()
                }
                new_sentences = [
                    s.strip() for s in re.split(r"\.\s*", new_val) if s.strip()
                ]
                additions = [s for s in new_sentences if s not in existing_sentences]
                if additions:
                    additions_text = ". ".join(additions) + "."
                    merged["biography_summary"] = (
                        old_bio.rstrip(".").rstrip() + ". " + additions_text if old_bio
                        else additions_text
                    )
        else:
            # String scalar fields — prefer newer non-null value
            if new_val:
                merged[field] = new_val

    # Keep legacy notes in sync with biography_summary
    if merged.get("biography_summary"):
        merged["notes"] = merged["biography_summary"]

    return merged


def save_group_members(
    members_data: Dict, profiles: Dict[str, Dict], profile_fields: List[str],
    recent_summaries: Optional[Dict[str, str]] = None
) -> None:
    """Write updated structured profiles and recent_summaries to group_members.json."""
    public_fields = [f for f in profile_fields if f != "name"]

    for member in members_data["members"]:
        name = member["name"]
        if name not in profiles:
            continue
        profile = profiles[name]

        # Write all extracted fields (including sensitive ones) to local file
        for field in public_fields:
            val = profile.get(field)
            if val is not None:
                member[field] = val
            elif field in SENSITIVE_FIELDS and field not in member:
                # Ensure sensitive field key exists (even if empty) so schema is consistent
                member[field] = None

        # Maintain legacy notes field
        if profile.get("biography_summary"):
            member["notes"] = profile["biography_summary"]

        # Write recent_summary if provided
        if recent_summaries and name in recent_summaries and recent_summaries[name]:
            member["recent_summary"] = recent_summaries[name]

    with open(GROUP_MEMBERS_FILE, "w", encoding="utf-8") as f:
        json.dump(members_data, f, ensure_ascii=False, indent=2)


def save_group_knowledge(
    knowledge_data: Dict, profiles: Dict[str, Dict]
) -> None:
    """Update group_knowledge.json with member bios and topic_mapping entries.

    - Updates existing ``category: member`` facts with the latest biography_summary.
    - Adds / updates ``category: topic_mapping`` facts for each member's
      frequently_discussed_topics.
    - Sensitive fields (political_preference) are never written here.
    """
    # Build lookup of existing fact IDs
    fact_index: Dict[str, int] = {
        fact["id"]: idx for idx, fact in enumerate(knowledge_data["facts"])
    }

    for member_name, profile in profiles.items():
        # --- Update member bio fact ---
        bio = profile.get("biography_summary") or profile.get("notes", "")
        member_fact_id = f"member_{member_name.lower()}"
        if bio and member_fact_id in fact_index:
            knowledge_data["facts"][fact_index[member_fact_id]]["text"] = bio

        # --- Add / update topic_mapping fact ---
        topics = profile.get("frequently_discussed_topics")
        if not topics:
            continue

        topic_fact_id = f"topics_{member_name.lower()}"
        topic_text = (
            f"{member_name} frequently discusses: {', '.join(topics[:10])}."
        )
        topic_fact = {
            "id": topic_fact_id,
            "category": "topic_mapping",
            "subject": member_name,
            "text": topic_text,
        }

        if topic_fact_id in fact_index:
            knowledge_data["facts"][fact_index[topic_fact_id]].update(topic_fact)
        else:
            knowledge_data["facts"].append(topic_fact)
            fact_index[topic_fact_id] = len(knowledge_data["facts"]) - 1

    with open(GROUP_KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_data, f, ensure_ascii=False, indent=2)


def main(test_mode: bool = False, resume_from: int = 0) -> None:
    """Main knowledge extraction pipeline."""
    print("=" * 60, flush=True)
    print("📚 KAYA KNOWLEDGE BASE GENERATOR (Structured Profiles)", flush=True)
    if test_mode:
        print("   [TEST MODE]", flush=True)
    if resume_from:
        print(f"   [RESUMING from chunk {resume_from}]", flush=True)
    print("=" * 60, flush=True)

    # Load config
    config = load_config()
    profile_fields = get_profile_fields(config)
    kg_config = config.get("knowledge_generation", {})
    words_per_token = 0.75
    chunk_size_tokens = kg_config.get("chunk_size_tokens", 2000)
    chunk_size_words = int(chunk_size_tokens * words_per_token)
    checkpoint_every = kg_config.get("checkpoint_every", 5)
    rate_limit_delay = float(kg_config.get("rate_limit_delay", 2.0))

    print(f"\n⚙️  Config: provider={config['generation']['provider']}", flush=True)
    print(f"   Profile fields: {', '.join(profile_fields)}", flush=True)

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

    # Current profiles — start from existing structured data if resuming
    current_profiles: Dict[str, Dict] = {}
    for m in members_data["members"]:
        profile: Dict[str, Any] = {}
        for field in DEFAULT_PROFILE_FIELDS:
            val = m.get(field)
            if val is not None:
                profile[field] = val
        # Legacy fallback
        if not profile.get("biography_summary") and m.get("notes"):
            profile["biography_summary"] = m["notes"]
            profile["notes"] = m["notes"]
        current_profiles[m["name"]] = profile

    # Current recent summaries (always replaced with latest, not merged).
    # Messages are processed in chronological order, so later chunks naturally
    # produce more recent summaries that overwrite earlier ones.
    current_recent_summaries: Dict[str, str] = {
        m["name"]: m.get("recent_summary", "") for m in members_data["members"]
    }

    # Chunk messages
    chunks = chunk_messages(messages, chunk_size_words)
    total_chunks = len(chunks)
    print(f"  Split into {total_chunks} chunks (~{chunk_size_words} words each).", flush=True)

    if test_mode:
        limit = config.get("test_mode", {}).get("generation", {}).get("chunks_limit", 3)
        chunks = chunks[:limit]
        print(f"  Test mode: processing first {len(chunks)} chunks.", flush=True)

    if resume_from > 0:
        chunks = chunks[resume_from:]
        print(f"  Resuming: skipping first {resume_from} chunks.", flush=True)

    # Load LLM provider
    print(f"\n🔗 Loading LLM provider ({config['generation']['provider']})...", flush=True)
    try:
        provider = load_provider(config)
        print("  Provider loaded.", flush=True)
    except Exception as e:
        print(f"  ⚠️  Could not load provider: {e}", flush=True)
        print("  Ensure API keys are set in .env", flush=True)
        sys.exit(1)

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
        user_prompt = build_extraction_prompt(
            chunk_text, current_profiles, mentioned, profile_fields
        )

        new_profiles = call_llm_for_profiles(provider, user_prompt)
        if new_profiles:
            for member_name, profile_data in new_profiles["members"].items():
                if not isinstance(profile_data, dict):
                    continue
                # Normalise member name
                matched_name = None
                if member_name in current_profiles:
                    matched_name = member_name
                else:
                    for name in current_profiles:
                        if name.lower() == member_name.lower():
                            matched_name = name
                            break

                if matched_name:
                    current_profiles[matched_name] = merge_profiles(
                        current_profiles[matched_name], profile_data
                    )
                    print(f"    ✅ Updated profile for {matched_name}", flush=True)
                else:
                    print(
                        f"    ⚠️  Unknown member '{member_name}' returned by LLM, skipping.",
                        flush=True,
                    )

            # recent_summaries always replace (they represent the latest snapshot)
            for member_name, summary in new_profiles["recent_summaries"].items():
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
            save_group_members(members_data, current_profiles, profile_fields, current_recent_summaries)
            save_group_knowledge(knowledge_data, current_profiles)

        # Rate limit
        if i < len(chunks) - 1:
            time.sleep(rate_limit_delay)

    # Final save
    print("\n💾 Saving final results...", flush=True)
    save_group_members(members_data, current_profiles, profile_fields, current_recent_summaries)
    save_group_knowledge(knowledge_data, current_profiles)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("✅ Knowledge base generation complete!", flush=True)
    print("\nProfiles generated:", flush=True)
    populated = 0
    for name, profile in current_profiles.items():
        bio = profile.get("biography_summary") or profile.get("notes", "")
        status = "✅" if bio else "⬜"
        snippet = (bio[:80] + "...") if len(bio) > 80 else (bio or "(empty)")
        print(f"  {status} {name}: {snippet}", flush=True)
        if bio:
            populated += 1
    print(f"\n{populated}/{len(current_profiles)} members have profiles.", flush=True)
    print(f"\nFiles updated:", flush=True)
    print(f"  📄 {GROUP_MEMBERS_FILE}", flush=True)
    print(f"  📄 {GROUP_KNOWLEDGE_FILE}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate structured Kaya member profiles from chat history."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only the first N chunks (test_mode.generation.chunks_limit in config).",
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

