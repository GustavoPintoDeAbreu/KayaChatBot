#!/usr/bin/env python3
"""Find the group's birthdays in its own history, and confirm them.

Every year the group writes a burst of "parabéns" to whoever's birthday it is.
This finds those bursts in the July export and the live log, works out who they
were for, and proposes a date. Nothing is used until a person accepts it: a
burst can be a birth, a new job or a wedding just as easily.

    scripts/mine_birthdays.py                     # list candidates with evidence
    scripts/mine_birthdays.py --accept Pedro=07-15
    scripts/mine_birthdays.py --set Manuel=05-02  # a date you simply know
    scripts/mine_birthdays.py --show              # what the bot will use, and why

It writes ``birthdays.json`` in the prod data directory when that exists, since
that is the copy the running bot reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chat import birthdays
from src.config_loader import load_config

BASE_DIR = Path(__file__).resolve().parent.parent
PROD_DATA = Path.home() / "kaya-prod" / "data"


def data_dir(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    return PROD_DATA if PROD_DATA.exists() else BASE_DIR / "data"


def export_rows(path: Path) -> Iterator[Tuple[datetime, str, str]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                yield datetime.fromisoformat(row["timestamp"]), row.get("sender", ""), row.get("text", "")
            except (ValueError, KeyError):
                continue


def live_rows(directory: Path) -> Iterator[Tuple[datetime, str, str]]:
    """The shared (group) log only: a birthday learned in a DM is not the group's."""
    path = directory / "shared.jsonl"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("timestamp"):
                yield datetime.fromtimestamp(int(row["timestamp"])), row.get("sender", ""), row.get("text", "")


def aliases_for(members: Dict) -> Dict[str, List[str]]:
    return {member["name"]: list(member.get("aliases") or [])
            for member in members.get("members", []) if member.get("name")}


def parse_assignment(value: str) -> Tuple[str, str]:
    member, _, raw = value.partition("=")
    month_day = birthdays.parse_date(raw.replace("-", "/", 1)) if "-" in raw else birthdays.parse_date(raw)
    if not member or not month_day:
        raise SystemExit(f"expected Member=MM-DD, got {value!r}")
    if "-" in raw:
        month, day = raw.split("-")
        month_day = f"{int(month):02d}-{int(day):02d}"
    return member.strip(), month_day


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--accept", action="append", default=[], metavar="MEMBER=MM-DD")
    parser.add_argument("--set", action="append", default=[], metavar="MEMBER=MM-DD")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--min-senders", type=int, default=3)
    args = parser.parse_args()

    config = load_config(str(BASE_DIR / "config.yaml"))
    directory = data_dir(args.data_dir)
    members = json.loads((directory / "group_members.json").read_text(encoding="utf-8"))
    store = birthdays.BirthdayStore(directory / "birthdays.json")
    known = {member["name"] for member in members.get("members", [])}

    for value, source in [*((v, "mined") for v in args.accept), *((v, "profile") for v in args.set)]:
        member, month_day = parse_assignment(value)
        if member not in known:
            raise SystemExit(f"{member!r} is not a member ({', '.join(sorted(known))})")
        store.set(member, month_day, source)
        print(f"✓ {member}: {birthdays.spoken_date(month_day)} ({source})")
    if args.accept or args.set:
        return 0

    resolved = birthdays.resolved_dates(store.load(), members)
    if args.show:
        for member in sorted(known):
            if member in resolved:
                month_day, source = resolved[member]
                print(f"{member:<12} {birthdays.spoken_date(month_day):<18} {source}")
            else:
                print(f"{member:<12} —")
        return 0

    from src.data.identity_resolver import SenderResolver

    resolver = SenderResolver(directory / "group_members.json",
                              (config.get("data", {}) or {}).get("sender_aliases"))
    rows = [(when, resolver.resolve(sender), text) for when, sender, text in
            [*export_rows(BASE_DIR / "data" / "all_messages_cleaned.jsonl"),
             *live_rows(directory / "live_messages")]]
    candidates = birthdays.mine(rows, aliases_for(members), min_senders=args.min_senders)
    if not candidates:
        print("No bursts found.")
        return 0
    for candidate in candidates:
        current = resolved.get(candidate.member)
        note = f"  (already {current[0]} from {current[1]})" if current else ""
        print(f"\n{candidate.member} {candidate.month_day} "
              f"[{candidate.confidence}] years={sorted(set(candidate.years))} "
              f"senders={candidate.senders}{note}")
        for line in candidate.evidence[:6]:
            print(f"    {line}")
    print("\nAccept with --accept Member=MM-DD; a date you know with --set Member=MM-DD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
