#!/usr/bin/env python3
"""Build `data_sim/` — the simulator's own data directory.

The simulator invents conversations. If it wrote them into `data/`, those
invented messages would be logged as group memory and ingested into the real
vector store, and the bot would afterwards "remember" things that never happened.
`kaya-dev` mounts the same `data/` as prod, so that is exactly what would occur
if the simulator were pointed at it.

So the sim instance gets a directory of its own. The vector store is **copied**,
not symlinked: retrieval is then tested against the group's real memory (which
never leaves the box) while anything the run writes lands in the copy and is
thrown away.

    kaya_chatbot_env/bin/python scripts/seed_sim_data.py          # create/refresh
    kaya_chatbot_env/bin/python scripts/seed_sim_data.py --check  # verify isolation

`--check` compares `data/` against a manifest taken at seed time and reports any
drift, which is how a run proves it did not touch the real data.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).parent.parent
REAL = BASE_DIR / "data"
SIM = BASE_DIR / "data_sim"

# Copied so the bot has real memory to answer from.
COPY_DIRS = ["rag_db"]
# Real member profiles are copied so RAG has something true to answer from —
# they stay on the box. Phone numbers, contacts and chat ids are NOT copied:
# they are PII the simulator has no use for, and the sim's own synthetic
# identities have to be there instead or every synthetic DM is dropped by the
# anti-spam whitelist.
COPY_FILES = [
    "group_members.json",
    "group_knowledge.json",
    "media_descriptions.json",
]
# Created empty: everything the run itself produces.
EMPTY_DIRS = ["live_messages", "whatsapp_prefs", "feedback", "sessions"]

# Directories under data/ whose contents the simulator must never alter.
WATCHED = ["rag_db", "live_messages", "whatsapp_prefs", "feedback"]
MANIFEST = BASE_DIR / "reports" / "sim" / "real_data_manifest.json"


def fingerprint(root: Path) -> Dict[str, str]:
    """Size+mtime per file under the watched paths. Cheap, and enough to catch a
    write — hashing 95MB of ChromaDB on every check would not be."""
    out: Dict[str, str] = {}
    for name in WATCHED:
        target = root / name
        if not target.exists():
            continue
        for path in sorted(target.rglob("*")):
            if path.is_file():
                stat = path.stat()
                out[str(path.relative_to(root))] = f"{stat.st_size}:{int(stat.st_mtime)}"
    return out


def seed(force: bool = False) -> None:
    if SIM.exists() and not force:
        print(f"{SIM} already exists — pass --force to rebuild it")
    else:
        if SIM.exists():
            shutil.rmtree(SIM)
        SIM.mkdir(parents=True)

        for name in COPY_DIRS:
            source = REAL / name
            if source.exists():
                print(f"  copying {name} …")
                shutil.copytree(source, SIM / name, symlinks=False)
            else:
                print(f"  ! {name} missing in {REAL}")

        for name in COPY_FILES:
            source = REAL / name
            if source.exists():
                shutil.copy2(source, SIM / name)
                print(f"  copied {name}")

        for name in EMPTY_DIRS:
            (SIM / name).mkdir(parents=True, exist_ok=True)

        # A fresh watermark file, so the sim ingester starts from nothing rather
        # than inheriting the real one's position.
        (SIM / "ingest_state.json").write_text("{}", encoding="utf-8")

        # Synthetic identities — see src/testing/sim_world.py for why none of
        # this is real.
        sys.path.insert(0, str(BASE_DIR))
        from src.testing import sim_world

        (SIM / "whatsapp_whitelist.json").write_text(
            json.dumps({"allowed": sim_world.whitelist()}, indent=2), encoding="utf-8")
        (SIM / "whatsapp_contacts.json").write_text(
            json.dumps(sim_world.contacts_map(), indent=2, ensure_ascii=False),
            encoding="utf-8")
        (SIM / "whatsapp_shared_chats.json").write_text(
            json.dumps({"shared_chats": sim_world.shared_chats()}, indent=2),
            encoding="utf-8")
        print(f"  wrote synthetic identities for {len(sim_world.PERSONAS)} personas")
        print(f"✓ seeded {SIM}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(fingerprint(REAL), indent=0), encoding="utf-8")
    print(f"✓ manifest of the real data written to {MANIFEST}")


def check_identities() -> int:
    """The direct question: did anything the simulator invented reach the real data?

    This is the check that matters, and unlike the manifest it is immune to prod
    running at the same time. Prod legitimately writes to live_messages/ and
    rag_db/ whenever it answers a message or runs its periodic ingest, so a
    timestamp diff cries wolf; a sim phone number or the sim group id appearing in
    the real store would be an actual leak.
    """
    sys.path.insert(0, str(BASE_DIR))
    from src.testing import sim_world

    needles = ([p.phone for p in sim_world.PERSONAS]
               + [p.name for p in sim_world.PERSONAS]
               + [sim_world.SIM_GROUP.split("@")[0]])

    hits = []
    for path in (REAL / "live_messages").rglob("*.jsonl"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for needle in needles:
            if needle in text:
                hits.append(f"{path.name}: {needle}")

    state = REAL / "ingest_state.json"
    if state.exists():
        text = state.read_text(encoding="utf-8", errors="ignore")
        for needle in needles:
            if needle in text:
                hits.append(f"ingest_state.json: {needle}")

    if hits:
        print("!! SIM DATA FOUND IN THE REAL STORE — isolation is broken")
        for hit in hits[:15]:
            print(f"   {hit}")
        return 1
    print(f"✓ no sim identity appears in {REAL} (checked {len(needles)} markers)")
    return 0


def check() -> int:
    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST} — run without --check first")
        return 1
    before = json.loads(MANIFEST.read_text(encoding="utf-8"))
    after = fingerprint(REAL)

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])

    if not (added or removed or changed):
        print(f"✓ real data untouched ({len(after)} files across {', '.join(WATCHED)})")
        return 0

    print("note: real data changed since the manifest was taken. If kaya-prod is")
    print("running this is expected — it logs messages and ingests on a timer.")
    print("What matters is the identity check above; this is only a hint.")
    for label, items in (("added", added), ("removed", removed), ("changed", changed)):
        for item in items[:10]:
            print(f"   {label}: {item}")
        if len(items) > 10:
            print(f"   … and {len(items) - 10} more {label}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the simulator's data directory.")
    parser.add_argument("--force", action="store_true", help="rebuild data_sim from scratch")
    parser.add_argument("--check", action="store_true",
                        help="verify the real data/ has not been modified")
    args = parser.parse_args()

    if args.check:
        # Identity first: it is the one that can actually fail meaningfully.
        code = check_identities()
        check()
        sys.exit(code)
    seed(force=args.force)


if __name__ == "__main__":
    main()
