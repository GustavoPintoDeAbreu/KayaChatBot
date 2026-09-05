#!/usr/bin/env python3
"""Index documents into the group's searchable memory, with page numbers.

Documents shared from now on are captured by the webhook (`src/chat/documents.py`,
wired in `whatsapp_adapter.handle_event`). This script is for the ones already
sent, and for measuring what a corpus costs before committing to one.

**Backfill is the reason this exists.** WAHA cannot replay them: the NOWEB store
is disabled on this deployment, so `GET /api/{session}/chats/{id}/messages`
answers "Enable NOWEB store ... and full_sync". The four papers shared in the
group on 2026-09-05 are not on disk anywhere and cannot be fetched. The route in
is a WhatsApp export taken with media, the same way `scripts/ingest_media.py`
backfills photos and voice notes.

    # measure a corpus without writing anything
    kaya_chatbot_env/bin/python scripts/ingest_documents.py --dir ~/papers --stats

    # index loose files
    kaya_chatbot_env/bin/python scripts/ingest_documents.py --file ~/papers/labour.pdf \\
        --sender Bernardo

    # backfill from a WhatsApp export taken WITH media
    kaya_chatbot_env/bin/python scripts/ingest_documents.py \\
        --export ~/Downloads/chat.txt --media ~/Downloads/chat

Idempotent: chunk ids come from the file's content hash, so re-running upserts
rather than duplicating, and a document shared twice is stored once.

The synopsis needs a llama server. Point `KAYA_LLAMA_URL` at one, or pass
`--no-synopsis` to index without it — the pages are still searchable, the
document just has no one-line description.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import documents
from src.chat.scope import SHARED
from src.config_loader import load_config

# "05/09/2026, 14:59 - Bernardo: labour_markets.pdf (file attached)".
# Unlike photos, a document keeps its ORIGINAL name in an export, so this cannot
# reuse ingest_media's IMG-/PTT- pattern.
LINE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2})\s*-\s*"
    r"(?P<sender>[^:]{1,60}):\s*(?P<text>.*)$"
)
ATTACHMENT = re.compile(
    r"(?P<file>[^\s:][^:]*?\.(?:pdf|txt|md|markdown))\s*\((?:file attached|ficheiro anexado)\)",
    re.IGNORECASE,
)


def parse_export(path: Path) -> List[Dict[str, Any]]:
    """Every line of the export that attaches a document."""
    found: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            match = LINE.match(raw.rstrip("\n"))
            if not match:
                continue
            attachment = ATTACHMENT.search(match.group("text").strip())
            if not attachment:
                continue
            found.append({
                "file": attachment.group("file").strip(),
                "sender": match.group("sender").strip(),
                "timestamp": _epoch(match.group("date"), match.group("time")),
            })
    return found


def _epoch(date_s: str, time_s: str) -> Optional[float]:
    for fmt in ("%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%y %H:%M", "%m/%d/%y %H:%M"):
        try:
            return datetime.strptime(f"{date_s} {time_s}", fmt).timestamp()
        except ValueError:
            continue
    return None


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _store_bytes(config: Dict[str, Any]) -> int:
    path = Path(documents._db_path(config))
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_argument_group("what to index")
    source.add_argument("--file", action="append", default=[],
                        help="a document; repeatable")
    source.add_argument("--dir", default="", help="every document in a directory")
    source.add_argument("--export", default="", help="a WhatsApp export .txt")
    source.add_argument("--media", default="",
                        help="the export's media folder (with --export)")
    parser.add_argument("--sender", default="", help="who shared it")
    parser.add_argument("--scope", default=SHARED,
                        help="'shared' (the group) or 'dm:<hash>'. A document in a "
                             "DM is only ever retrievable from that DM.")
    parser.add_argument("--stats", action="store_true",
                        help="report size and timing; indexes nothing")
    parser.add_argument("--no-synopsis", action="store_true",
                        help="skip the local-model description")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_synopsis:
        config.setdefault("documents", {})["synopsis_max_new_tokens"] = 0

    jobs: List[Dict[str, Any]] = []
    for path in args.file:
        jobs.append({"path": Path(path).expanduser(), "sender": args.sender,
                     "timestamp": None})
    if args.dir:
        for path in sorted(Path(args.dir).expanduser().rglob("*")):
            if path.is_file() and documents.is_document("", path.name, config):
                jobs.append({"path": path, "sender": args.sender, "timestamp": None})
    if args.export:
        media = Path(args.media).expanduser() if args.media else Path(args.export).parent
        for entry in parse_export(Path(args.export).expanduser()):
            path = media / entry["file"]
            if not path.exists():
                print(f"⚠️  referenced but missing: {entry['file']}")
                continue
            jobs.append({"path": path, "sender": entry["sender"],
                         "timestamp": entry["timestamp"]})
    if args.limit:
        jobs = jobs[:args.limit]

    if not jobs:
        parser.error("nothing to do — pass --file, --dir or --export")

    before = _store_bytes(config)
    collection = None if args.stats else documents.get_collection(config)
    encoder = None if args.stats else documents._encoder(config)

    total_pages = total_chunks = total_bytes = 0
    started = time.time()
    rows = []
    for job in jobs:
        path: Path = job["path"]
        try:
            payload = path.read_bytes()
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  could not read {path.name}: {exc}")
            continue
        mimetype = documents.guess_mimetype("", path.name)
        cap = int(float((config.get("documents", {}) or {}).get("max_file_mb", 50)))
        if len(payload) > cap * 1024 * 1024:
            print(f"⚠️  {path.name} is {_human(len(payload))}, over the {cap}MB cap")
            continue

        t0 = time.time()
        if args.stats:
            pages = documents.extract_pages(
                payload, mimetype,
                max_pages=int((config.get("documents", {}) or {}).get("max_pages", 0) or 0))
            chunks = documents.chunk_pages(
                pages,
                int((config.get("documents", {}) or {}).get("chunk_size_tokens", 400)),
                int((config.get("documents", {}) or {}).get("chunk_overlap_tokens", 60)))
            report = {"ok": bool(chunks), "pages": len(pages), "chunks": len(chunks),
                      "filename": path.name, "synopsis": ""}
        else:
            report = documents.index_document(
                payload, filename=path.name, scope=args.scope,
                sender=job["sender"], config=config, mimetype=mimetype,
                timestamp=job["timestamp"], collection=collection, encoder=encoder)
        elapsed = time.time() - t0

        if not report.get("ok"):
            print(f"✗ {path.name}: {report.get('error', 'nothing extractable')}")
            continue
        total_pages += report["pages"]
        total_chunks += report["chunks"]
        total_bytes += len(payload)
        rows.append((path.name, len(payload), report["pages"], report["chunks"], elapsed))
        print(f"✓ {path.name[:44]:46s} {_human(len(payload)):>8s} "
              f"{report['pages']:4d}p {report['chunks']:5d}ch {elapsed:6.1f}s")
        if report.get("synopsis"):
            print(f"    {report['synopsis'][:150]}")

    after = _store_bytes(config)
    print("\n" + "=" * 78)
    print(f"{len(rows)} document(s), {total_pages} pages, {total_chunks} chunks, "
          f"{_human(total_bytes)} of source, {time.time() - started:.1f}s")
    if args.stats:
        # Measured, not guessed: 105 chunks from 99 pages grew a fresh store by
        # 3.3MB (~32KB/chunk, including Chroma's fixed overhead), and the live
        # 3,669-chunk store sits at 95MB (~26KB/chunk amortised).
        print("--stats: nothing was written. Projection uses the measured "
              "~26-32KB per chunk (1024-dim bge-m3 vector + HNSW graph + text).")
        print(f"projected vector-store growth: ~{_human(total_chunks * 26 * 1024)} "
              f"to ~{_human(total_chunks * 32 * 1024)}")
        print(f"plus the archived files themselves: {_human(total_bytes)}")
    else:
        print(f"vector store: {_human(before)} -> {_human(after)} "
              f"(+{_human(max(0, after - before))})")
        if total_chunks:
            print(f"~{_human(max(0, after - before) / total_chunks)} per chunk")


if __name__ == "__main__":
    main()
