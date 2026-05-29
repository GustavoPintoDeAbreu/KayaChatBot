"""
Incremental pipeline for processing new WhatsApp and Instagram chat exports.

Only processes messages newer than the latest timestamp already in the dataset,
and deduplicates by SHA-256 hash of (timestamp + sender + text) to guard
against exact duplicates at the boundary.

CLI usage:
    # Single WhatsApp file
    python src/data/incremental_update.py --input data/wpp/new_export.txt

    # Single Instagram file
    python src/data/incremental_update.py --input data/insta/message_1.json

    # Whole WhatsApp directory — every *.txt file is processed; also picks up
    # message_*.json files from a sibling 'insta/' directory automatically.
    python src/data/incremental_update.py --input data/wpp/

    # Skip vector-DB rebuild (e.g. when called from run_full_pipeline.py)
    python src/data/incremental_update.py --input data/wpp/ --no-rebuild-db
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

# ---------------------------------------------------------------------------
# Repo root — resolved relative to this file so the script works from any CWD
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.config_loader import load_config
from src.data.extract_all_messages import MAX_TOKENS_PER_CHUNK, MessageChunker, MessageExtractor

# ---------------------------------------------------------------------------
# Configuration & paths
# ---------------------------------------------------------------------------
# Load via the single entry point (load_config — never read config.yaml directly).
CONFIG_PATH = _REPO_ROOT / "config.yaml"
_config = load_config(str(CONFIG_PATH))

# Docker vs. local
if os.path.exists("/app"):
    DATA_DIR = Path("/app/data")
    PYTHON = "python"
else:
    DATA_DIR = _REPO_ROOT / "data"
    _venv_python = _REPO_ROOT / "kaya_chatbot_env" / "bin" / "python"
    PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

OUTPUT_CLEANED = DATA_DIR / "all_messages_cleaned.jsonl"
OUTPUT_FINETUNE_CHUNKS = DATA_DIR / "finetune_chunks.jsonl"
METADATA_FILE = DATA_DIR / "pipeline_metadata.json"


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def compute_message_hash(msg: Dict) -> str:
    """Return SHA-256 hex digest of ``timestamp + sender + text``."""
    key = f"{msg['timestamp']}{msg['sender']}{msg['text']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# State loading / saving
# ---------------------------------------------------------------------------

def load_existing_messages() -> tuple:
    """
    Load cleaned messages from *OUTPUT_CLEANED*.

    Returns
    -------
    (messages, hashes, last_timestamp)
        *messages*        – list of existing message dicts
        *hashes*          – set of SHA-256 hashes for deduplication
        *last_timestamp*  – ISO timestamp of the most recent message, or None
    """
    if not OUTPUT_CLEANED.exists():
        return [], set(), None

    messages: List[Dict] = []
    hashes: Set[str] = set()

    with open(OUTPUT_CLEANED, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                messages.append(msg)
                hashes.add(compute_message_hash(msg))
            except json.JSONDecodeError:
                continue

    last_timestamp: Optional[str] = messages[-1]["timestamp"] if messages else None
    return messages, hashes, last_timestamp


def load_metadata() -> Dict:
    """Load ``pipeline_metadata.json``, returning defaults if the file is absent."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "last_processed_date": None,
        "total_messages": 0,
        "processing_history": [],
    }


def save_metadata(metadata: Dict) -> None:
    """Persist *metadata* to ``pipeline_metadata.json``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_new_file(
    input_path: Path,
    existing_hashes: Set[str],
    last_timestamp: Optional[str],
) -> List[Dict]:
    """
    Parse *input_path* (WhatsApp ``.txt`` or Instagram ``.json``) and return
    only genuinely new messages.

    The file type is determined by the file extension:
    - ``.txt``  → WhatsApp export (``MessageExtractor.extract_whatsapp``)
    - ``.json`` → Instagram export (``MessageExtractor.extract_instagram``)

    A message is accepted when:
    - Its timestamp is >= *last_timestamp*.
    - Its SHA-256 hash is NOT already in *existing_hashes*.

    Side-effect: accepted hashes are added to *existing_hashes* in place.
    """
    extractor = MessageExtractor()
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        raw_messages = extractor.extract_instagram(input_path)
    else:
        raw_messages = extractor.extract_whatsapp(input_path)

    new_messages: List[Dict] = []
    duplicates_skipped = 0

    for msg in raw_messages:
        # Date filter — skip messages that are strictly older than the last processed date
        if last_timestamp and msg["timestamp"] < last_timestamp:
            continue

        # Hash-based deduplication
        h = compute_message_hash(msg)
        if h in existing_hashes:
            duplicates_skipped += 1
            continue

        existing_hashes.add(h)
        new_messages.append(msg)

    print(
        f"   ✅ {len(new_messages)} new messages found, "
        f"{duplicates_skipped} duplicates skipped"
    )
    return new_messages


def rebuild_finetune_chunks(all_messages: List[Dict]) -> List[Dict]:
    """Re-chunk *all_messages* and overwrite ``finetune_chunks.jsonl``."""
    chunker = MessageChunker(max_tokens=MAX_TOKENS_PER_CHUNK)
    chunks = chunker.create_finetune_chunks(all_messages)

    with open(OUTPUT_FINETUNE_CHUNKS, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"   ✅ Created {len(chunks)} finetune chunks")
    return chunks


def rebuild_vector_db() -> None:
    """Invoke ``build_vector_db.py`` as a subprocess to rebuild the RAG DB."""
    print("   🔨 Rebuilding vector database...")
    script = _REPO_ROOT / "src" / "data" / "build_vector_db.py"
    result = subprocess.run(
        [PYTHON, str(script)],
        cwd=str(_REPO_ROOT),
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"   ⚠️  Vector DB rebuild exited with code {result.returncode}")
    else:
        print("   ✅ Vector database rebuilt")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_incremental_update(
    input_paths: List[Path],
    rebuild_db: bool = True,
) -> bool:
    """
    Incremental update pipeline.

    Parameters
    ----------
    input_paths:
        One or more WhatsApp ``.txt`` or Instagram ``.json`` export files.
    rebuild_db:
        When *True* (default) the ChromaDB vector database is rebuilt after
        the dataset is updated.  Pass *False* when called from
        ``run_full_pipeline.py`` so that the pipeline can handle the DB step
        separately (or skip it).

    Returns
    -------
    bool
        *True* on success.
    """
    print("=" * 60)
    print("🔄 INCREMENTAL DATA UPDATE PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Load existing state
    # ------------------------------------------------------------------
    print("\n📖 Loading existing messages...")
    existing_messages, existing_hashes, last_timestamp = load_existing_messages()
    print(f"   Existing messages  : {len(existing_messages)}")
    print(f"   Last processed date: {last_timestamp or 'none (first run)'}")

    metadata = load_metadata()

    # ------------------------------------------------------------------
    # Step 2 — Process each input file
    # ------------------------------------------------------------------
    all_new_messages: List[Dict] = []

    for input_path in input_paths:
        print(f"\n📥 Processing: {input_path.name}")
        new_msgs = process_new_file(input_path, existing_hashes, last_timestamp)
        all_new_messages.extend(new_msgs)

    if not all_new_messages:
        print("\n✅ No new messages found — dataset is already up to date.")
        return True

    total_new = len(all_new_messages)
    print(f"\n📊 Total new messages across all files: {total_new}")

    # ------------------------------------------------------------------
    # Step 3 — Sort new messages and append to cleaned file
    # ------------------------------------------------------------------
    all_new_messages.sort(key=lambda x: x["timestamp"])

    print(f"\n💾 Appending {total_new} new messages to {OUTPUT_CLEANED.name}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CLEANED, "a", encoding="utf-8") as fh:
        for msg in all_new_messages:
            fh.write(json.dumps(msg, ensure_ascii=False) + "\n")

    # Build the full sorted message list for re-chunking
    all_messages = existing_messages + all_new_messages
    all_messages.sort(key=lambda x: x["timestamp"])
    print(f"   ✅ Dataset now has {len(all_messages)} messages")

    # ------------------------------------------------------------------
    # Step 4 — Re-chunk the full dataset
    # ------------------------------------------------------------------
    print(f"\n🔨 Re-chunking full dataset ({len(all_messages)} messages)...")
    chunks = rebuild_finetune_chunks(all_messages)

    # ------------------------------------------------------------------
    # Step 5 — (Optional) Rebuild vector DB
    # ------------------------------------------------------------------
    if rebuild_db:
        print("\n🔨 Rebuilding vector database...")
        rebuild_vector_db()

    # ------------------------------------------------------------------
    # Step 6 — Update pipeline metadata
    # ------------------------------------------------------------------
    new_last_date = all_messages[-1]["timestamp"]
    history_entry = {
        "date": datetime.now(timezone.utc).isoformat(),
        "files": [p.name for p in input_paths],
        "messages_added": total_new,
        "chunks_created": len(chunks),
    }

    metadata["last_processed_date"] = new_last_date
    metadata["total_messages"] = len(all_messages)
    metadata["processing_history"].append(history_entry)
    save_metadata(metadata)
    print(f"\n💾 Metadata saved to {METADATA_FILE.name}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 INCREMENTAL UPDATE COMPLETE")
    print("=" * 60)
    print(f"  New messages added  : {total_new}")
    print(f"  Total messages now  : {len(all_messages)}")
    print(f"  Finetune chunks     : {len(chunks)}")
    print(f"  Last processed date : {new_last_date}")

    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally process new WhatsApp / Instagram chat exports, "
            "appending only previously-unseen messages to the cleaned dataset."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to a WhatsApp .txt or Instagram .json export file, "
            "or a directory.  When a directory is provided, all *.txt files "
            "in it are processed as WhatsApp exports and all message_*.json "
            "files in a sibling 'insta/' directory are processed as Instagram "
            "exports."
        ),
    )
    parser.add_argument(
        "--no-rebuild-db",
        action="store_true",
        default=False,
        help=(
            "Skip vector DB rebuild after the update "
            "(useful when called from the full pipeline)."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        # Collect WhatsApp TXT files from the given directory
        input_files = sorted(input_path.glob("*.txt"))
        # Also collect Instagram JSON files from a sibling 'insta/' directory
        insta_dir = input_path.parent / "insta"
        if not insta_dir.exists():
            # Try treating the directory itself as insta if no sibling found
            insta_dir = input_path / "insta"
        if insta_dir.exists():
            insta_files = sorted(insta_dir.glob("message_*.json"))
            input_files = list(input_files) + insta_files
            if insta_files:
                print(f"📁 Also found {len(insta_files)} Instagram file(s) in {insta_dir}")
        if not input_files:
            print(f"❌ No .txt or message_*.json files found under: {input_path}")
            sys.exit(1)
        print(f"📁 Directory mode: found {len(input_files)} file(s)")
    elif input_path.is_file():
        input_files = [input_path]
    else:
        print(f"❌ Input path does not exist: {input_path}")
        sys.exit(1)

    success = run_incremental_update(
        input_paths=input_files,
        rebuild_db=not args.no_rebuild_db,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
