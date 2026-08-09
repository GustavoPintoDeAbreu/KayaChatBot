#!/usr/bin/env python3
"""Fold a fresh WhatsApp export's NEW messages into the vector store.

The existing `incremental_update.py` ends in a full `build_vector_db` rebuild,
which drops the collection — taking the image descriptions and any live-ingested
conversation with it. This does the same job through the incremental upsert path
instead, so nothing already in the store is destroyed.

Only messages newer than what the store already covers are ingested. The cutoff is
detected from the newest `timestamp_end` in the collection, so re-running is a
no-op rather than a duplicate. Chunk ids derive from the message content they
contain, so even an overlapping run upserts rather than duplicating.

Everything from the group export is written as `shared` scope: it is the group's
collective memory, readable from every chat including DMs.

    # see what would happen
    kaya_chatbot_env/bin/python scripts/ingest_export_text.py \
        --export "$HOME/Downloads/chat.txt" --dry-run

    # ingest
    kaya_chatbot_env/bin/python scripts/ingest_export_text.py \
        --export "$HOME/Downloads/chat.txt"
"""

import argparse
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.chat.scope import SHARED

LINE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2})\s*-\s*"
    r"(?P<sender>[^:]{1,60}):\s*(?P<text>.*)$"
)
ATTACHMENT = re.compile(r"(IMG|STK|VID|PTT|AUD)-\d{8}-WA\d+\.\w+\s*\(.*?\)")
SYSTEM_NOISE = ("<Media omitted>", "This message was deleted", "Esta mensagem foi apagada",
                "Missed voice call", "Missed video call", "null")


def parse_iso(date_s: str, time_s: str) -> Optional[str]:
    for fmt in ("%m/%d/%y %H:%M", "%d/%m/%y %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(f"{date_s} {time_s}", fmt).isoformat()
        except ValueError:
            continue
    return None


def parse_export(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            m = LINE.match(raw.rstrip("\n"))
            if not m:
                if out:
                    out[-1]["text"] += " " + raw.strip()
                continue
            ts = parse_iso(m.group("date"), m.group("time"))
            if not ts:
                continue
            out.append({
                "ts": ts,
                "sender": m.group("sender").strip(),
                "text": m.group("text").strip(),
            })
    return out


def is_content(text: str) -> bool:
    """Drop attachment stubs and WhatsApp's own system lines."""
    if not text or not text.strip():
        return False
    if ATTACHMENT.search(text):
        return False   # media is handled by scripts/ingest_media.py
    return not any(n.lower() in text.lower() for n in SYSTEM_NOISE)


def chunk(messages: List[Dict[str, Any]], max_messages=16, max_chars=1800) -> List[Dict[str, Any]]:
    """Group consecutive messages, mirroring the shape of the existing chunks."""
    chunks, current, chars = [], [], 0

    def flush():
        nonlocal current, chars
        if not current:
            return
        lines, participants = [], []
        for m in current:
            lines.append(f"{m['sender']}: {m['text']}")
            if m["sender"] not in participants:
                participants.append(m["sender"])
        body = "\n".join(lines)
        cid = "exp_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]
        chunks.append({
            "id": cid, "text": body,
            "metadata": {
                "participants": ",".join(participants), "mentioned": "",
                "message_count": len(current), "token_count": len(body) // 4,
                "timestamp_start": current[0]["ts"], "timestamp_end": current[-1]["ts"],
                "scope": SHARED, "source": "export",
            },
        })
        current, chars = [], 0

    for m in messages:
        current.append(m)
        chars += len(m["text"])
        if len(current) >= max_messages or chars >= max_chars:
            flush()
    flush()
    return chunks


def newest_in_store(collection) -> Optional[str]:
    """The newest timestamp already covered, so we only add what is new."""
    newest = None
    offset, batch = 0, 1000
    total = collection.count()
    while offset < total:
        got = collection.get(limit=batch, offset=offset, include=["metadatas"])
        for meta in got.get("metadatas") or []:
            ts = (meta or {}).get("timestamp_end")
            if ts and (newest is None or ts > newest):
                newest = ts
        offset += batch
    return newest


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest new messages from a WhatsApp export.")
    ap.add_argument("--export", required=True)
    ap.add_argument("--since", default=None,
                    help="ISO cutoff; default = newest timestamp already in the store")
    ap.add_argument("--scope", default=SHARED)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_dir = Path(__file__).parent.parent
    config = load_config(str(base_dir / "config.yaml"))

    from src.data.ingest import Ingester
    ing = Ingester(config)

    cutoff = args.since or newest_in_store(ing.collection)
    print(f"store holds {ing.collection.count()} chunks; newest covered: {cutoff}")

    messages = parse_export(Path(args.export).expanduser())
    content = [m for m in messages if is_content(m["text"])]
    new = [m for m in content if not cutoff or m["ts"] > cutoff]
    print(f"export: {len(messages)} lines, {len(content)} with content, {len(new)} newer than cutoff")

    if not new:
        print("nothing new to ingest")
        return

    print(f"   range: {new[0]['ts'][:16]} → {new[-1]['ts'][:16]}")
    chunks = chunk(new)
    print(f"   -> {len(chunks)} chunks")

    if args.dry_run:
        for c in chunks[:2]:
            print(f"\n  {c['id']}  [{c['metadata']['timestamp_start'][:16]}]")
            print("   " + c["text"][:200].replace("\n", " | "))
        return

    docs = [c["text"] for c in chunks]
    print("embedding …")
    embeddings = ing.encoder.encode(docs, show_progress_bar=False,
                                    normalize_embeddings=True).tolist()
    for i in range(0, len(chunks), 200):
        sl = slice(i, i + 200)
        ing.collection.upsert(
            ids=[c["id"] for c in chunks[sl]], documents=docs[sl],
            metadatas=[c["metadata"] for c in chunks[sl]], embeddings=embeddings[sl],
        )
    print(f"✓ collection now holds {ing.collection.count()} chunks")


if __name__ == "__main__":
    main()
