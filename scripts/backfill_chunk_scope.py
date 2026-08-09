#!/usr/bin/env python3
"""One-off migration: stamp every existing conversation chunk as `shared`.

Retrieval now filters by a `scope` metadata field so a DM can never surface in
the group (src/chat/scope.py). Everything already in the store predates that
field — it is the historical Kaya group export — so it is all shared memory, but
Chroma's `$in` filter will not match a chunk that has no `scope` key at all.
Without this backfill, scope-filtered retrieval silently returns nothing.

Idempotent: re-running only touches chunks that are still missing the field.

    kaya_chatbot_env/bin/python scripts/backfill_chunk_scope.py --dry-run
    kaya_chatbot_env/bin/python scripts/backfill_chunk_scope.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.chat.scope import SHARED

BATCH = 500


def main() -> None:
    ap = argparse.ArgumentParser(description="Stamp existing chunks as shared scope.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; change nothing.")
    ap.add_argument("--scope", default=SHARED, help=f"Scope to apply (default {SHARED}).")
    ap.add_argument("--collection", default="kaya_conversations")
    args = ap.parse_args()

    base_dir = Path(__file__).parent.parent
    config = load_config(str(base_dir / "config.yaml"))
    db_path = config.get("rag", {}).get("db_path", "./data/rag_db")
    path = Path(db_path)
    if not path.is_absolute():
        path = base_dir / db_path

    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_collection(args.collection)
    total = collection.count()
    print(f"collection '{args.collection}': {total} chunks at {path}")

    scanned = missing = 0
    offset = 0
    while offset < total:
        got = collection.get(limit=BATCH, offset=offset, include=["metadatas"])
        ids = got.get("ids") or []
        metadatas = got.get("metadatas") or []
        if not ids:
            break

        to_fix_ids, to_fix_meta = [], []
        for cid, meta in zip(ids, metadatas):
            scanned += 1
            meta = dict(meta or {})
            if meta.get("scope"):
                continue
            missing += 1
            meta["scope"] = args.scope
            to_fix_ids.append(cid)
            to_fix_meta.append(meta)

        if to_fix_ids and not args.dry_run:
            collection.update(ids=to_fix_ids, metadatas=to_fix_meta)

        offset += len(ids)
        print(f"  scanned {scanned}/{total}  missing-scope {missing}", end="\r", flush=True)

    print(f"\nscanned {scanned}, {'would stamp' if args.dry_run else 'stamped'} {missing} as '{args.scope}'")

    if not args.dry_run:
        # Verify by querying the way retrieval will.
        probe = collection.get(where={"scope": args.scope}, limit=1, include=["metadatas"])
        n_scoped = len(probe.get("ids") or [])
        print(f"verification: scope filter returns results = {n_scoped > 0}")


if __name__ == "__main__":
    main()
