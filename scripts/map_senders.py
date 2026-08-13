#!/usr/bin/env python3
"""Who in the group the bot cannot put a name to.

A WhatsApp display name is chosen by the person and two people can share one, so
the bot identifies senders by number and falls back to the name only when it has
no mapping. That fallback is where the damage happens: one member is known to the
group by a nickname while his display name is another member's actual name, so
everything he said was logged, retrieved and attributed to the wrong man for
weeks. Two members answering to the same first name had the same problem.

This lists the group's participants that map to nobody, with whatever the log
knows about them, so the gap is visible instead of being guessed at.

    scripts/map_senders.py                          # who is unmapped
    scripts/map_senders.py --set 351911002262=Ricky # write a mapping
    scripts/map_senders.py --audit                  # members with several ids

`--audit` is the check that would have caught the mismapping: two ids for one
member is normal (a phone and a @lid), but two *people* behind one member is not,
and it looks exactly the same from here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent
PROD_DATA = Path.home() / "kaya-prod" / "data"


def data_dir(explicit: Path | None) -> Path:
    """Prod by default — that is the copy the running bot reads."""
    if explicit:
        return explicit
    return PROD_DATA if PROD_DATA.exists() else BASE_DIR / "data"


def load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def waha_participants(base_url: str, session: str, api_key: str,
                      chat_id: str) -> List[str]:
    """Participant ids from WAHA. Empty when it cannot be reached."""
    try:
        import requests

        response = requests.get(
            f"{base_url.rstrip('/')}/api/{session}/groups/{chat_id}",
            headers={"X-Api-Key": api_key} if api_key else {}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        people = (payload.get("participants")
                  or payload.get("groupMetadata", {}).get("participants") or [])
        return [str(p.get("id") or p.get("jid") or "") for p in people if p]
    except Exception as exc:  # noqa: BLE001 — the log still works without it
        print(f"⚠️  could not ask WAHA for participants ({exc}); "
              f"falling back to the message log alone")
        return []


def known(contacts: Dict[str, str], identity: str) -> str:
    """The member an id maps to, trying the shapes resolve_speaker tries."""
    key = (identity or "").strip().lower()
    if not key:
        return ""
    local = key.split("@", 1)[0]
    for candidate in (key, f"{local}@c.us", local):
        if candidate in contacts:
            return contacts[candidate]
    return ""


def seen_in_log(directory: Path) -> Dict[str, Dict]:
    """id -> {name it was logged under, message count, a few examples}.

    Only rows written since the id started being recorded carry one; older rows
    are why this cannot repair the past, only the future.
    """
    found: Dict[str, Dict] = {}
    log = directory / "live_messages" / "shared.jsonl"
    if not log.exists():
        return found
    try:
        with open(log, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                identity = row.get("sender_id") or row.get("sender_phone")
                if not identity:
                    continue
                entry = found.setdefault(str(identity), {"names": Counter(), "texts": []})
                entry["names"][row.get("sender", "?")] += 1
                if len(entry["texts"]) < 3 and (row.get("text") or "").strip():
                    entry["texts"].append(row["text"].strip())
    except OSError:
        pass
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--chat", default="", help="group chat id (default: the shared ones)")
    parser.add_argument("--set", default="", metavar="ID=NAME", action="append",
                        dest="assignments", help="map an id to a member; repeatable")
    parser.add_argument("--audit", action="store_true",
                        help="show members with more than one id")
    parser.add_argument("--bot", default="", metavar="ID",
                        help="the bot's own id, so it is not listed as a stranger "
                             "(also read from KAYA_BOT_JID / KAYA_BOT_LID)")
    args = parser.parse_args()

    directory = data_dir(args.data_dir)
    contacts_path = directory / "whatsapp_contacts.json"
    contacts = load(contacts_path, {})

    if args.assignments:
        for pair in args.assignments:
            if "=" not in pair:
                print(f"  ignored {pair!r}: expected ID=NAME")
                continue
            identity, name = (part.strip() for part in pair.split("=", 1))
            previous = contacts.get(identity)
            contacts[identity] = name
            print(f"  {identity} -> {name}" + (f"  (was {previous!r})" if previous else ""))
        tmp = contacts_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(contacts, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, contacts_path)
        print(f"wrote {contacts_path} — restart the bot to pick it up")
        return 0

    if args.audit:
        by_member: Dict[str, List[str]] = {}
        for identity, name in contacts.items():
            by_member.setdefault(name, []).append(identity)
        print("members with more than one id (a phone and a @lid is normal; two "
              "PEOPLE behind one name is the bug):\n")
        for name, ids in sorted(by_member.items()):
            if len(ids) > 1:
                print(f"  {name:10} {ids}")
        return 0

    shared = load(directory / "whatsapp_shared_chats.json", {}).get("shared_chats", [])
    chats = [args.chat] if args.chat else shared
    participants: List[str] = []
    for chat in chats:
        participants += waha_participants(
            os.environ.get("KAYA_WAHA_URL", "http://localhost:3000"),
            os.environ.get("KAYA_WAHA_SESSION", "default"),
            os.environ.get("KAYA_WAHA_API_KEY", ""), chat)

    logged = seen_in_log(directory)
    # Anyone WAHA lists, plus anyone the log has an id for — the second catches
    # people who left the group but whose messages are still in the corpus.
    everyone = list(dict.fromkeys(participants + list(logged)))

    # The bot is a participant too, and listing it as an unknown person every
    # time is noise that trains you to skim the list.
    bot_ids = {b.strip().lower() for b in (
        os.environ.get("KAYA_BOT_JID", ""), os.environ.get("KAYA_BOT_LID", ""),
        args.bot) if b.strip()}
    for identity, entry in logged.items():
        if "Kaya Bot" in entry["names"]:
            bot_ids.add(identity.strip().lower())

    unmapped = [i for i in everyone
                if not known(contacts, i) and i.strip().lower() not in bot_ids]
    print(f"{len(everyone)} identities seen, {len(everyone) - len(unmapped)} mapped, "
          f"{len(unmapped)} not:\n")
    for identity in unmapped:
        entry = logged.get(identity)
        if entry:
            names = ", ".join(f"{n} x{c}" for n, c in entry["names"].most_common(3))
            print(f"  {identity}\n      logged as: {names}")
            for text in entry["texts"]:
                print(f"      | {text[:90]}")
        else:
            print(f"  {identity}\n      no messages logged with an id yet")
        print()

    if unmapped:
        print("map one with:  scripts/map_senders.py --set <id>=<MemberName>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
