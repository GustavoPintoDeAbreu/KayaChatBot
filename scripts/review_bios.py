#!/usr/bin/env python3
"""Review what the biography refresh wants to change, and accept or reject it.

The refresh proposes; nothing reaches the bot until somebody says yes. That is
deliberate — these are statements about real people, written by a model, and the
group reads them back in the bot's answers. This is the step where a human sees
them, so it has to be fast enough to actually happen.

    scripts/review_bios.py                       # show everything pending
    scripts/review_bios.py --member Rafa         # just one person
    scripts/review_bios.py --accept Rafa         # apply, keeping a snapshot
    scripts/review_bios.py --accept all
    scripts/review_bios.py --reject Gil          # drop the proposal, keep the profile
    scripts/review_bios.py --pin "Gil works in sales."

Every addition is printed with the messages that produced it. A proposed fact
with no evidence you recognise is the one to reject.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent
PROD_DATA = Path.home() / "kaya-prod" / "data"

GREEN, RED, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = DIM = BOLD = OFF = ""


def data_dir(explicit: Path | None) -> Path:
    """Where the profiles and proposals live.

    Defaults to the prod data directory when it exists, because that is the copy
    the running bot reads — editing the dev one and wondering why nothing changed
    is the obvious mistake to design out.
    """
    if explicit:
        return explicit
    return PROD_DATA if PROD_DATA.exists() else BASE_DIR / "data"


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def evidence_index(directory: Path) -> Dict[str, str]:
    """message id -> "Sender: text", for showing WHY a fact was proposed."""
    index: Dict[str, str] = {}
    log = directory / "live_messages" / "shared.jsonl"
    if not log.exists():
        return index
    try:
        with open(log, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("id"):
                    index[row["id"]] = f"{row.get('sender', '?')}: {row.get('text', '')}"
    except OSError:
        pass
    return index


_STOPWORDS = frozenset(
    "the a an and or of to in is are was were for on at with que de da do e o as os um uma "
    "no na nos nas para com por se lhe ele ela isso este esta muito mais já não sim".split()
)


def best_quotes(fact: str, member: str, ids: List[str], index: Dict[str, str],
                limit: int) -> List[str]:
    """The messages most likely to be the reason for this fact.

    A proposal carries every message id in the chunk it came from, which is a
    whole conversation. Showing the first few of those is showing nothing — the
    reviewer needs the lines that actually mention the thing. Ranked by word
    overlap with the fact, with the member's own messages preferred, because "if
    you do not recognise the evidence, reject it" only works when the evidence
    is the evidence.
    """
    wanted = {w for w in _tokens(fact) if w not in _STOPWORDS and len(w) > 3}
    wanted.discard(member.lower())

    scored, about, own = [], [], []
    for mid in ids:
        line = index.get(mid)
        if not line:
            continue
        speaker, _, body = line.partition(": ")
        mine = speaker.strip() == member
        overlap = len(wanted & set(_tokens(body)))
        # Two content words, not one: a single match is usually an ordinary word
        # the stopword list happens not to carry ("before", "many"), and quoting
        # an unrelated line as the reason for a fact is worse than quoting none.
        if overlap >= 2 or (overlap and mine):
            scored.append((overlap + (1 if mine else 0), line))
        elif member.lower() in body.lower():
            about.append(line)
        elif mine:
            own.append(line)

    scored.sort(key=lambda row: -row[0])
    # The facts are written in English and the group writes Portuguese, so word
    # overlap catches proper nouns and numbers and little else. Falling back to
    # lines that NAME the member, then to lines they wrote themselves, is a far
    # better guess than the first lines of a conversation about something else.
    lines: List[str] = []
    for pool in ([line for _, line in scored], about, own):
        for line in pool:
            if len(lines) >= limit:
                return lines
            if line not in lines:
                lines.append(line)
    return lines


def _tokens(text: str) -> set:
    import re

    return set(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def show(name: str, proposal: Dict[str, Any], index: Dict[str, str], quotes: int) -> None:
    current = proposal.get("current") or []
    proposed = proposal.get("proposed") or []
    added = set(proposal.get("added") or [])
    removed = proposal.get("removed") or []

    print(f"\n{BOLD}── {name} ──{OFF}")
    for fact in proposed:
        mark = f"{GREEN}+{OFF}" if fact in added else " "
        print(f" {mark} {fact}")
    for fact in removed:
        print(f" {RED}-{OFF} {DIM}{fact}{OFF}")
    if not added and not removed:
        print(f"   {DIM}(no change){OFF}")

    seen = set()
    for entry in proposal.get("new_facts") or []:
        if entry.get("fact") not in added or entry.get("fact") in seen:
            continue
        seen.add(entry["fact"])
        lines = best_quotes(entry["fact"], name, entry.get("evidence") or [],
                            index, quotes)
        print(f"   {DIM}why: {entry['fact']}{OFF}")
        if lines:
            for line in lines:
                print(f"     {DIM}| {line[:110]}{OFF}")
        else:
            # Said out loud rather than left blank. A fact with nothing behind it
            # is the model elaborating on the profile it was shown, and it is the
            # one to reject — but only if the reviewer can see that it is one.
            print(f"     {RED}| nothing in these messages supports this — "
                  f"check it{OFF}")


def snapshot(members_path: Path) -> Path:
    """Copy the profiles before writing, so any accepted batch can be undone.

    The name is made unique rather than just timestamped: accepting two members
    in the same second would otherwise overwrite the first snapshot with one
    taken AFTER the first change, losing the only copy of the original.
    """
    target = members_path.parent / "profile_snapshots"
    target.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = target / f"group_members.{stamp}.json"
    serial = 1
    while path.exists():
        path = target / f"group_members.{stamp}-{serial}.json"
        serial += 1
    shutil.copy2(members_path, path)
    return path


def write_members(members_path: Path, payload: Dict[str, Any]) -> None:
    """Atomic, surgical write.

    Deliberately not ``generate_knowledge_base.save_group_members``: that writes
    to a module-level path and rewrites every profile field from a profiles dict,
    which is the wrong shape for accepting a key_facts proposal against whichever
    copy of the file the bot is actually reading.
    """
    tmp = members_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(members_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="defaults to ~/kaya-prod/data when present")
    parser.add_argument("--member", default="", help="show only this member")
    parser.add_argument("--accept", default="", metavar="NAME|all")
    parser.add_argument("--reject", default="", metavar="NAME|all")
    parser.add_argument("--pin", default="", metavar="FACT",
                        help="with --member: keep this fact through every future refresh")
    parser.add_argument("--quotes", type=int, default=2,
                        help="messages shown per proposed fact (default 2)")
    args = parser.parse_args()

    directory = data_dir(args.data_dir)
    members_path = directory / "group_members.json"
    proposals_path = directory / "bio_proposals.json"

    if not members_path.exists():
        print(f"no profiles at {members_path}")
        return 1

    members_doc = load_json(members_path, {"members": []})
    by_name = {m["name"]: m for m in members_doc.get("members", [])}
    document = load_json(proposals_path, {})
    proposals: Dict[str, Any] = document.get("proposals", {}) if document else {}

    # ── pin ──────────────────────────────────────────────────────────────────
    if args.pin:
        if args.member not in by_name:
            print("--pin needs --member <name> of an existing member")
            return 1
        member = by_name[args.member]
        pinned = list(member.get("pinned_facts") or [])
        if args.pin not in pinned:
            pinned.append(args.pin)
        member["pinned_facts"] = pinned
        if args.pin not in (member.get("key_facts") or []):
            member["key_facts"] = [args.pin] + list(member.get("key_facts") or [])
        print(f"snapshot: {snapshot(members_path)}")
        write_members(members_path, members_doc)
        print(f"pinned for {args.member}: {args.pin}")
        return 0

    # ── reject ───────────────────────────────────────────────────────────────
    if args.reject:
        names = list(proposals) if args.reject == "all" else [args.reject]
        for name in names:
            proposals.pop(name, None)
        document["proposals"] = proposals
        proposals_path.write_text(json.dumps(document, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"rejected: {', '.join(names) or 'nothing'} (profiles untouched)")
        return 0

    # ── accept ───────────────────────────────────────────────────────────────
    if args.accept:
        names = list(proposals) if args.accept == "all" else [args.accept]
        applied: List[str] = []
        for name in names:
            proposal = proposals.get(name)
            if not proposal or name not in by_name:
                print(f"  no pending proposal for {name}")
                continue
            by_name[name]["key_facts"] = list(proposal.get("proposed") or [])
            applied.append(name)
        if not applied:
            return 1
        print(f"snapshot: {snapshot(members_path)}")
        write_members(members_path, members_doc)
        for name in applied:
            proposals.pop(name, None)
        document["proposals"] = proposals
        proposals_path.write_text(json.dumps(document, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"accepted: {', '.join(applied)}")
        print("  the bot picks these up on its next restart "
              "(the system prompt is built once, at import).")
        return 0

    # ── show ─────────────────────────────────────────────────────────────────
    if not proposals:
        print(f"nothing pending in {proposals_path}")
    else:
        index = evidence_index(directory)
        for name, proposal in proposals.items():
            if args.member and name != args.member:
                continue
            show(name, proposal, index, args.quotes)
        print(f"\n{len(proposals)} member(s) pending. "
              f"--accept <name>|all, --reject <name>|all")

    unresolved = (document or {}).get("unresolved_senders") or {}
    if unresolved:
        print(f"\n{BOLD}senders nobody could be matched to{OFF} — their messages are "
              "skipped entirely:")
        for sender, count in sorted(unresolved.items(), key=lambda kv: -kv[1]):
            print(f"  {sender!r}: {count} message(s)")
        print(f"  {DIM}add an alias in group_members.json, or a mapping in "
              f"data/whatsapp_sender_aliases.json{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
