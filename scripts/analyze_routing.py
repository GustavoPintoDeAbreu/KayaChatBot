#!/usr/bin/env python3
"""What the router actually did, read back out of the interaction log.

The log recorded latency, delivery medium and whether a web search fired, but
never which route a turn took. So when the group said the bot was "obcecado com
o Gil", there was no way to answer beyond re-reading the thread — and no way to
tell a bad classification apart from a bad generation.

`Reply.telemetry` now carries the route and the members each turn named, and this
reads those fields back:

    route_mode      how the message was classified
    route_fallback  the classification failed and fell back
    retrieved_chars how much group context was injected
    query_members   members named in the QUESTION
    reply_members   members named in the ANSWER  <- the dominance signal

``reply_members`` is the one to watch. A healthy group bot spreads its attention
roughly evenly; a share far above 1/N means the answer is being driven by whose
profile is fattest rather than by what was asked.

Turns logged before the telemetry existed have no ``route_mode`` and are counted
separately as untagged rather than silently skewing the histogram.

Usage:
    scripts/analyze_routing.py [log.jsonl ...] [--since YYYY-MM-DD] [--source whatsapp]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOGS = [
    BASE_DIR / "data" / "feedback" / "live_interactions.jsonl",
    Path.home() / "kaya-prod" / "data" / "feedback" / "live_interactions.jsonl",
]
DEFAULT_MEMBERS = BASE_DIR / "data" / "group_members.json"


def load_roster(path: Path) -> List[tuple]:
    """``[(canonical_name, [alias regexes])]`` for deriving members from raw text.

    Mirrors ``ConversationRetriever.named_members``. Duplicated deliberately: this
    script must run against an exported log on a box with no model, no vector
    store and no CUDA, and importing the retriever drags all three in.
    """
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    roster = []
    for member in data.get("members", []):
        name = member.get("name")
        if not name:
            continue
        aliases = {name.lower(), *(a.lower() for a in member.get("aliases", []))}
        roster.append((name, [re.compile(rf"\b{re.escape(a)}\b") for a in aliases]))
    return roster


def derive_members(text: str, roster: List[tuple]) -> List[str]:
    lowered = (text or "").lower()
    return [name for name, patterns in roster if any(p.search(lowered) for p in patterns)]


def read_rows(paths: List[Path]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        if not path.exists():
            print(f"⚠️  no such log: {path}", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _bar(share: float, width: int = 28) -> str:
    return "█" * int(round(share * width))


def _percent(count: int, total: int) -> str:
    return f"{(100.0 * count / total):5.1f}%" if total else "  n/a"


def report(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No interactions matched.")
        return

    tagged = [r for r in rows if r.get("route_mode")]
    untagged = len(rows) - len(tagged)

    print(f"\n{len(rows)} interaction(s)"
          + (f"  ({untagged} logged before routing telemetry existed)" if untagged else ""))
    span = sorted(r.get("timestamp", "") for r in rows if r.get("timestamp"))
    if span:
        print(f"span: {span[0][:19]} → {span[-1][:19]}")

    if not tagged:
        print("\nNo turn carries route_mode yet — deploy the telemetry change and "
              "collect traffic before comparing.")
        _members(rows)
        return

    print("\n── routes ──")
    modes = Counter(r["route_mode"] for r in tagged)
    for mode, count in modes.most_common():
        latencies = [r["latency_ms"] for r in tagged
                     if r.get("route_mode") == mode and r.get("latency_ms")]
        median = f"{statistics.median(latencies)/1000:5.1f}s" if latencies else "    -"
        retrieved = [r.get("retrieved_chars", 0) for r in tagged
                     if r.get("route_mode") == mode]
        rag = f"{int(statistics.mean(retrieved)):6d}" if retrieved else "     -"
        print(f"  {mode:9} {count:4}  {_percent(count, len(tagged))}  "
              f"median {median}  mean rag {rag} chars  {_bar(count/len(tagged))}")

    fallbacks = sum(1 for r in tagged if r.get("route_fallback"))
    flag = "  ⚠️ above 5%" if fallbacks / len(tagged) > 0.05 else ""
    print(f"\n  fallback to default route: {fallbacks} of {len(tagged)}"
          f" ({_percent(fallbacks, len(tagged))}){flag}")
    for raw, count in Counter(
        (r.get("route_raw") or "").strip()[:60] for r in tagged if r.get("route_fallback")
    ).most_common(5):
        print(f"    {count:3}x  {raw!r}")

    _members(tagged)


def _members(rows: List[Dict[str, Any]]) -> None:
    """Who the bot talks about, against who was asked about."""
    asked = Counter(name for r in rows for name in (r.get("query_members") or []))
    named = Counter(name for r in rows for name in (r.get("reply_members") or []))
    if not named and not asked:
        print("\n── members ──\n  no member telemetry on these rows.")
        return

    total = sum(named.values())
    roster = len(set(named) | set(asked)) or 1
    even = 1.0 / roster
    print(f"\n── who the answers are about ── ({total} mention(s), "
          f"even split would be {100*even:.0f}%)")
    for name, count in named.most_common():
        share = count / total if total else 0.0
        flag = "  ⚠️ dominant" if share > max(0.25, 2 * even) else ""
        print(f"  {name:12} {count:4}  {_percent(count, total)}  "
              f"(asked about {asked.get(name, 0):3})  {_bar(share)}{flag}")

    unprompted = sum(
        1 for r in rows
        if (r.get("reply_members") or []) and not (r.get("query_members") or [])
    )
    print(f"\n  turns naming a member nobody asked about: {unprompted} of {len(rows)}"
          f" ({_percent(unprompted, len(rows))})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="*", type=Path,
                        help="interaction logs (default: dev + prod live_interactions.jsonl)")
    parser.add_argument("--since", default="", metavar="YYYY-MM-DD",
                        help="only interactions on or after this date")
    parser.add_argument("--source", default="", help="filter by surface (whatsapp / web)")
    parser.add_argument("--group-only", action="store_true",
                        help="only group messages, not DMs")
    parser.add_argument("--members", type=Path, default=DEFAULT_MEMBERS,
                        help="group_members.json, used to derive members on older rows")
    args = parser.parse_args()

    rows = list(read_rows(args.logs or [p for p in DEFAULT_LOGS if p.exists()]))

    # Rows predating the telemetry carry no member fields. Deriving them from the
    # stored text is what makes a BASELINE possible at all — otherwise the first
    # measurement of the dominance problem could only be taken after the fix.
    roster = load_roster(args.members)
    if roster:
        for row in rows:
            if "reply_members" not in row:
                row["reply_members"] = derive_members(row.get("assistant_response", ""), roster)
            if "query_members" not in row:
                row["query_members"] = derive_members(row.get("user_message", ""), roster)

    if args.since:
        rows = [r for r in rows if (r.get("timestamp") or "") >= args.since]
    if args.source:
        rows = [r for r in rows if r.get("source") == args.source]
    if args.group_only:
        rows = [r for r in rows if r.get("is_group")]

    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
