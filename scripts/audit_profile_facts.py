#!/usr/bin/env python3
"""Audit curated key_facts against message evidence (provenance + attribution).

Flags curated member facts that the chat logs do not support, that are more
strongly associated with a *different* member (attribution error), or that look
stale. Prints receipts (source message ids + which members the fact's key term
is actually tied to). Read-only; no GPU, no teacher.

    kaya_chatbot_env/bin/python scripts/audit_profile_facts.py
    kaya_chatbot_env/bin/python scripts/audit_profile_facts.py --member Gil
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.data.profile_store import MemberEvidenceIndex, audit_member_key_facts


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit key_facts against log evidence.")
    ap.add_argument("--member", default=None, help="Only audit this member.")
    ap.add_argument("--messages", default="data/all_messages_cleaned.jsonl")
    ap.add_argument("--only-flagged", action="store_true", help="Hide 'ok' facts.")
    args = ap.parse_args()

    config = load_config("config.yaml")
    members_file = Path(config["data"]["group_members_file"])
    members = json.loads(members_file.read_text(encoding="utf-8")).get("members", [])
    sender_aliases = config.get("data", {}).get("sender_aliases", {})

    messages = [json.loads(l) for l in open(args.messages, encoding="utf-8")]
    print(f"Indexing {len(messages)} messages …")
    index = MemberEvidenceIndex(members_file, sender_aliases).index(messages)

    flagged = 0
    for member in members:
        if args.member and member["name"].lower() != args.member.lower():
            continue
        rows = audit_member_key_facts(member, index)
        printed_header = False
        for r in rows:
            if args.only_flagged and r["verdict"] == "ok":
                continue
            if not printed_header:
                print(f"\n=== {member['name']} ===")
                printed_header = True
            mark = {"ok": "  ", "UNSUPPORTED": "??", "CROSS-ATTRIBUTED": "!!"}.get(r["verdict"], "??")
            if r["verdict"] not in ("ok", "no-salient-terms"):
                flagged += 1
            print(f" {mark} [{r['verdict']}] {r['fact']}")
            if r.get("terms"):
                print(f"      terms={r['terms']}  support={r.get('support_count')}  last_seen={r.get('last_seen')}")
                assoc = r.get("associated_members") or {}
                if assoc:
                    top = sorted(assoc.items(), key=lambda kv: kv[1], reverse=True)[:4]
                    print(f"      term tied to: {top}")
                if r.get("sample_msg_ids"):
                    print(f"      receipts: {r['sample_msg_ids']}")

    print(f"\n{flagged} fact(s) flagged for review.")


if __name__ == "__main__":
    main()
