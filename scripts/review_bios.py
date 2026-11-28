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
    scripts/review_bios.py --queue                         # the gate: waiting, applied, dropped
    scripts/review_bios.py --undo NAME                     # undo the last applied fact about NAME
    scripts/review_bios.py --gate-backlog [--dry-run]      # sort the old proposals through the gate
    scripts/review_bios.py --member NAME --add "FACT"      # add a fact he gave, dated now
    scripts/review_bios.py --member NAME --drop "FACT"     # take a fact off (close wording is enough)
    scripts/review_bios.py --member NAME --alias "ALIAS"   # another name for the same person
    scripts/review_bios.py --member NAME --clear-summary   # empty a stale recent_summary
    scripts/review_bios.py --date-backfill                 # date the facts the gate applied

Since 2026-11-19 the refresh no longer just proposes: src/data/bio_gate.py applies
grounded, non-sensitive facts on its own and sends the rest to Gustavo weekly.

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


def show(name: str, proposal: Dict[str, Any]) -> None:
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
        print(f"   {DIM}why: {entry['fact']}{OFF}")
        # The quote the extractor copied out of the messages, already verified
        # to be there. This used to be guessed by ranking the chunk's messages
        # on word overlap, which meant an invented fact was handed an unrelated
        # line as its "evidence" and looked supported.
        quote = (entry.get("quote") or "").strip()
        if quote:
            print(f"     {DIM}| {quote[:150]}{OFF}")
        else:
            print(f"     {RED}| no quote recorded — proposed before quotes were "
                  f"required, treat with suspicion{OFF}")


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
    parser.add_argument("--queue", action="store_true", help="show the gate's queue")
    parser.add_argument("--undo", default="", metavar="NAME",
                        help="restore the snapshot from before the last applied fact about NAME")
    parser.add_argument("--gate-backlog", action="store_true",
                        help="sort every pending proposal through the gate, then archive the file")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --gate-backlog: work on copies and print what would happen")
    parser.add_argument("--add", default="", metavar="FACT", help="with --member: add a fact, dated now")
    parser.add_argument("--drop", default="", metavar="FACT", help="with --member: take a fact off")
    parser.add_argument("--alias", default="", metavar="ALIAS", help="with --member: add an alias")
    parser.add_argument("--clear-summary", action="store_true",
                        help="with --member: empty recent_summary")
    parser.add_argument("--date-backfill", action="store_true",
                        help="date the facts bio_queue.json says the gate applied")
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

    if args.queue or args.undo or args.gate_backlog:
        return gate_commands(args, directory, members_path, proposals_path)
    if args.add or args.drop or args.alias or args.clear_summary or args.date_backfill:
        return edit_commands(args, directory, members_path, members_doc, by_name)

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
        for name, proposal in proposals.items():
            if args.member and name != args.member:
                continue
            show(name, proposal)
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


def edit_commands(args, directory: Path, members_path: Path, members_doc: Dict[str, Any],
                  by_name: Dict[str, Dict[str, Any]]) -> int:
    """Hand edits Gustavo asked for, each with a snapshot that --undo NAME restores."""
    from src.data import bio_gate, fact_meta

    queue = bio_gate.BioQueue(directory / "bio_queue.json")
    if args.date_backfill:
        dated = fact_meta.backfill(members_doc.get("members", []), queue.data["applied"])
        if dated:
            print(f"snapshot: {bio_gate.snapshot(members_path)}")
            bio_gate.write_members(members_path, members_doc)
        print(f"dated {dated} fact(s) from the gate's log; the rest stay undated")
        return 0
    member = by_name.get(args.member)
    if member is None:
        print("--add/--drop/--alias/--clear-summary need --member <name> of an existing member")
        return 1
    done = []
    facts = list(member.get("key_facts") or [])
    if args.drop:
        hit = next((f for f in facts if f == args.drop or bio_gate.same_fact(f, args.drop)), None)
        if hit is None:
            print(f"no fact like {args.drop!r} on {args.member}")
            return 1
        facts.remove(hit)
        fact_meta.forget(member, hit)
        done.append(f"dropped: {hit}")
    if args.add:
        if not any(bio_gate.same_fact(f, args.add) for f in facts):
            facts.append(args.add.strip())
        fact_meta.stamp(member, args.add.strip(), "manual")
        done.append(f"added: {args.add.strip()}")
    member["key_facts"] = facts
    if args.alias:
        aliases = list(member.get("aliases") or [])
        if args.alias.lower() not in (a.lower() for a in aliases):
            aliases.append(args.alias.lower())
        member["aliases"] = aliases
        done.append(f"alias: {args.alias.lower()}")
    if args.clear_summary:
        member["recent_summary"] = ""
        done.append("recent_summary cleared")
    snap = bio_gate.snapshot(members_path)
    bio_gate.write_members(members_path, members_doc)
    queue.data.setdefault("manual", []).append(
        {"member": args.member, "changes": done, "snapshot": str(snap), "moved_at": time.time()})
    queue.save()
    print(f"snapshot: {snap}")
    for line in done:
        print(f"  {args.member} {line}")
    print("  the bot reads it on the next turn (the prompt is rebuilt when the file changes); "
          f"--undo {args.member} puts it back")
    return 0


def _grounding_generator(config: Dict[str, Any]):
    """The live Kaya model through the broker, as on the 2026-09-28 replays."""
    import os

    from src.chat.engine import _load_model
    from src.chat.inference_backend import build_backend

    os.environ.setdefault("KAYA_INFERENCE_BACKEND", "ollama")
    os.environ.setdefault("KAYA_OLLAMA_URL", "http://127.0.0.1:8200/upstream/kaya")
    _model, tokenizer = _load_model(config)
    backend = build_backend(config, None, tokenizer)

    def generate(system: str, user: str) -> str:
        return backend.generate([{"role": "system", "content": system},
                                 {"role": "user", "content": user}], max_new_tokens=6,
                                sampling={"temperature": 0.0, "top_p": 1.0, "top_k": 0,
                                          "repetition_penalty": 1.0})
    return generate


def _message_lookup(directory: Path, members_path: Path, config: Dict[str, Any]):
    from src.data.bio_refresh import resolve_sender
    from src.data.identity_resolver import SenderResolver
    from src.data.message_log import MessageLog

    resolver = SenderResolver(members_path, (config.get("data", {}) or {}).get("sender_aliases") or {})
    lookup = {}
    for row in MessageLog(base_dir=str(directory / "live_messages")).read("shared"):
        if row.get("id"):
            lookup[row["id"]] = {"sender": resolve_sender(row, resolver, config),
                                 "text": row.get("text") or ""}
    return lookup


def gate_commands(args, directory: Path, members_path: Path, proposals_path: Path) -> int:
    from src.config_loader import load_config
    from src.data import bio_gate

    queue_path = directory / "bio_queue.json"
    config = load_config(str(BASE_DIR / "config.yaml"))
    limit = int((config.get("bio_refresh") or {}).get("max_key_facts", 8))

    if args.undo:
        restored = bio_gate.undo(bio_gate.BioQueue(queue_path), members_path, args.undo)
        print(f"restored {restored}" if restored else f"nothing applied to undo for {args.undo}")
        return 0 if restored else 1

    if args.queue:
        queue = bio_gate.BioQueue(queue_path)
        print(json.dumps(bio_gate.status(queue), indent=2))
        for item in queue.data["ask"]:
            print(f"  ? {item['fact']}  ({item.get('reason')})")
        for item in queue.data["applied"][-10:]:
            print(f"  {GREEN}+{OFF} {item['fact']}  ({item.get('applied')})")
        return 0

    document = load_json(proposals_path, {})
    new_facts = {name: list(p.get("new_facts") or [])
                 for name, p in (document.get("proposals") or {}).items()}
    total = sum(len(v) for v in new_facts.values())
    print(f"{total} pending fact(s) across {len(new_facts)} member(s)")
    work_members, work_queue = members_path, queue_path
    if args.dry_run:
        import tempfile
        scratch = Path(tempfile.mkdtemp(prefix="bio-gate-"))
        work_members = scratch / "group_members.json"
        shutil.copy2(members_path, work_members)
        work_queue = scratch / "bio_queue.json"
        if queue_path.exists():
            shutil.copy2(queue_path, work_queue)
    members = json.loads(work_members.read_text(encoding="utf-8"))["members"]
    lookup = _message_lookup(directory, members_path, config)
    queue = bio_gate.BioQueue(work_queue)
    counts = bio_gate.gate(new_facts, members, lookup, queue, _grounding_generator(config),
                           work_members, limit)
    queue.save()
    print(json.dumps(counts))
    for key in ("applied", "ask", "candidates"):
        print(f"\n{BOLD}{key}{OFF} ({len(queue.data[key])})")
        for item in queue.data[key][:25]:
            print(f"  {item['member']}: {item['fact']}  [{item.get('reason') or ''}] "
                  f"{DIM}«{(item['quotes'][0]['quote'] if item['quotes'] else '')[:80]}»{OFF}")
    if args.dry_run:
        print(f"\n(dry run: nothing written outside {work_queue.parent})")
        return 0
    archived = proposals_path.with_name(f"bio_proposals.archived-{time.strftime('%Y%m%d')}.json")
    proposals_path.rename(archived)
    print(f"\nold proposals archived to {archived.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
