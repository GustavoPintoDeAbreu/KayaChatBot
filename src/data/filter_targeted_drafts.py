"""Filter hallucinated entries from a targeted QA draft JSONL.

Usage:
    python src/data/filter_targeted_drafts.py <input.jsonl> <output.jsonl>

Reads the canonical member roster from data/group_members.json and removes any
conversation that:
  - Contains a female indicator word (ela, dela, sua used as a female possessive,
    "she ", "her ") for a group member context.
  - References a name that looks like a proper noun but is NOT in the canonical
    member set (potential hallucinated member).

Prints a per-category removal report to stdout and writes the clean conversations
to the output file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEMALE_PATTERNS = re.compile(
    r"\b(ela\b|dela\b|lhe\b.*?\bela\b|she\b|her\b)",
    re.IGNORECASE,
)

# Words that look like Portuguese capitalized names but are obviously not members.
# We use a heuristic: any Title-Case word that's not in the canonical set and not
# a common Portuguese word (article, preposition, etc.).
COMMON_PT_WORDS = {
    # Portuguese articles / prepositions / conjunctions
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "sobre", "entre",
    "e", "ou", "mas", "que", "se", "não", "sim", "já",
    "este", "esta", "esse", "essa", "aquele", "aquela",
    "eu", "tu", "ele", "ela", "nós", "vocês", "eles", "elas",
    "me", "te", "lhe", "nos", "vos", "lhes",
    "meu", "minha", "teu", "tua", "seu", "sua",
    "muito", "mais", "também", "mesmo", "quando", "como",
    "aqui", "ali", "lá", "hoje", "amanhã", "ontem",
    # Common PT phrases that start with capital (appear in RAG template headers)
    "conversa", "conversas", "fim", "início",
    # Greetings / sentence starters that happen to be Title-Case
    "olá", "boa", "bom", "tudo", "tiveste",
    # English equivalents
    "the", "and", "but", "not", "yes", "no", "is", "are", "was", "were",
    "he", "she", "it", "we", "they", "you", "i",
    "my", "your", "his", "her", "our", "their",
    "ok", "okay", "yeah", "hey", "hi", "bye",
    "what", "who", "when", "where", "how",
    # Bot / group name
    "kaya",
    # Known non-member entities referenced in member knowledge
    # --- Gil's pets & partner ---
    "cuca", "luana",
    # --- Rafa's partner & son ---
    "mel", "martim",
    # --- Brands / services mentioned in member facts ---
    "dazn", "fuel",                         # Peter's work
    "dolby", "atmos",                       # Gil's audio preference
    "five", "guys",                         # Peter's food pref
    "skype",                                # Carnall's classes
    # --- Places in Portugal ---
    "lisboa", "porto", "caxias", "sintra", "marginalíssimo", "marginal",
    # --- Football clubs / events ---
    "benfica", "sporting", "porto", "benfiquista", "sportinguista",
    # --- Other proper nouns commonly referenced ---
    "natal", "instagram", "whatsapp",
    # --- Part of canonical multi-word aliases ---
    "pereira",                              # from "benny pereira"
    # --- Common conversational capitalised words ---
    "inception", "marvel",                  # film references
}


def load_canonical_names(members_path: Path) -> set[str]:
    """Return lowercase set of all canonical names and aliases."""
    with open(members_path, encoding="utf-8") as f:
        data = json.load(f)
    names: set[str] = set()
    for member in data["members"]:
        names.add(member["name"].lower())
        for alias in member.get("aliases", []):
            names.add(alias.lower())
    return names


def extract_all_text(turns: list[dict]) -> str:
    """Concatenate all turn content fields into one string."""
    return " ".join(t.get("content", "") for t in turns if isinstance(t, dict))


def has_female_indicator(text: str) -> bool:
    """Return True if the text contains a female-gender indicator word."""
    return bool(FEMALE_PATTERNS.search(text))


def has_hallucinated_name(text: str, canonical: set[str]) -> tuple[bool, str]:
    """Return (True, name) if a capitalized word looks like a hallucinated member name."""
    # Look for Title-Case words (likely names) that are NOT canonical and not common words
    # Exclude words that immediately follow a sentence-start punctuation (those are just
    # normal sentence beginnings).
    # Strategy: find every Title-Case token; skip if it's in canonical or COMMON_PT_WORDS.
    tokens = re.findall(r"\b([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]{2,})\b", text)
    for token in tokens:
        lower = token.lower()
        if lower in canonical or lower in COMMON_PT_WORDS:
            continue
        # Heuristic: only flag if the word appears ≥2 times (more likely a name reference)
        # or if the text explicitly seems to attribute it as a person's name
        count = len(re.findall(rf"\b{re.escape(token)}\b", text))
        if count >= 2:
            return True, token
    return False, ""


def is_contaminated(turns: list[dict], canonical: set[str]) -> tuple[bool, str]:
    """Check a conversation for contamination.

    Returns (is_bad, reason) where reason describes the first failure found.
    """
    text = extract_all_text(turns)
    if has_female_indicator(text):
        # Extra check: avoid false positives from Portuguese "sua" used as
        # a possessive for a male (which happens). Only flag "ela"/"she"/"her".
        if re.search(r"\b(ela\b|she\b)", text, re.IGNORECASE):
            return True, "female_pronoun"
    flagged, name = has_hallucinated_name(text, canonical)
    if flagged:
        return True, f"unknown_name:{name}"
    return False, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Filter hallucinated entries from targeted QA draft.")
    parser.add_argument("input", help="Input JSONL file (e.g. data/targeted_qa_draft.jsonl)")
    parser.add_argument("output", help="Output JSONL file (filtered)")
    parser.add_argument(
        "--members",
        default=None,
        help="Path to group_members.json (default: data/group_members.json relative to project root)",
    )
    args = parser.parse_args()

    # Resolve project root (two levels up from this script: src/data/ -> root)
    project_root = Path(__file__).resolve().parent.parent.parent
    members_path = Path(args.members) if args.members else project_root / "data" / "group_members.json"

    if not members_path.exists():
        print(f"❌ Members file not found: {members_path}", file=sys.stderr)
        sys.exit(1)

    canonical = load_canonical_names(members_path)
    print(f"✅ Loaded {len(canonical)} canonical name tokens from {members_path.name}")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    total = 0
    kept = 0
    removed_by_category: dict[str, int] = {}
    removed_by_reason: dict[str, int] = {}

    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            turns = record.get("conversations", [])
            category = record.get("category", "unknown")

            bad, reason = is_contaminated(turns, canonical)
            if bad:
                removed_by_category[category] = removed_by_category.get(category, 0) + 1
                removed_by_reason[reason] = removed_by_reason.get(reason, 0) + 1
            else:
                fout.write(line + "\n")
                kept += 1

    removed = total - kept
    print(f"\n{'='*55}")
    print(f"FILTER REPORT")
    print(f"{'='*55}")
    print(f"  Total records  : {total}")
    print(f"  Kept           : {kept}")
    print(f"  Removed        : {removed}")
    if removed_by_category:
        print(f"\n  Removed by category:")
        for cat, count in sorted(removed_by_category.items()):
            print(f"    {cat:40s}  {count}")
    if removed_by_reason:
        print(f"\n  Removed by reason:")
        for reason, count in sorted(removed_by_reason.items()):
            print(f"    {reason:40s}  {count}")
    print(f"{'='*55}")
    print(f"  Output written to: {output_path}")


if __name__ == "__main__":
    main()
