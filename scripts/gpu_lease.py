#!/usr/bin/env python3
"""Borrow the image-generation card without taking the bot down.

Heavy GPU work used to mean stopping prod. It does not have to: the two cards do
different jobs.

    GPU1   llama-server + Whisper — every reply, every voice note, every photo
    GPU0   image generation only, idle between the occasional ~90-180s render

So pausing image generation frees a whole 24GB while chat, voice and vision keep
running untouched — enough for a 27B teacher in 4-bit, a bake-off, or any other
job that used to require an outage.

    scripts/gpu_lease.py acquire --reason "regenerate member profiles" --ttl 45m
    kaya_chatbot_env/bin/python src/data/generate_knowledge_base.py --backend local
    scripts/gpu_lease.py release

``acquire`` returns only once the card is actually free: it writes the lease so
no NEW render starts, then waits for any render already running to finish.
Requests made meanwhile are queued, not dropped, and the asker is told the wait.

The lease carries an expiry, so a crashed job cannot leave image generation dead
— the worst case is that it comes back by itself at the TTL. ``--ttl`` is
therefore a promise about the longest the group will go without pictures, not an
estimate; overrun by re-running ``acquire`` to extend it.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chat import imagegen  # noqa: E402
from src.config_loader import load_config  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
# Prod serves from its own checkout, and the app only ever reads the lease from
# the data directory it was started with.
PROD_DATA = Path.home() / "kaya-prod" / "data" / "gpu0_lease.json"


def parse_duration(text: str) -> float:
    """"45m", "2h", "90s" or bare seconds."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", text or "", re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(f"unreadable duration: {text!r}")
    value = float(match.group(1))
    return value * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2).lower()]


def card_memory_mb(index: str) -> int:
    """Memory in use on one physical card, or -1 if nvidia-smi cannot say.

    Asked of the card directly rather than of the bot, because "is GPU0 free" is
    the actual question — a render in another container, or a stray process,
    counts just as much as one of ours.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 — no nvidia-smi is not a reason to fail
        return -1


def wait_for_card(index: str, free_below_mb: int, timeout: float) -> bool:
    """Block until the card is idle. True if it got there before ``timeout``."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        used = card_memory_mb(index)
        if used < 0:
            print("⚠️  cannot read nvidia-smi; not waiting for the card")
            return True
        if used <= free_below_mb:
            return True
        remaining = int(deadline - time.time())
        print(f"   render in flight ({used} MiB on GPU{index}); waiting… "
              f"{remaining}s left", flush=True)
        time.sleep(5)
    return False


def describe(lease) -> str:
    if not lease:
        return "free"
    left = int(float(lease["expires_at"]) - time.time())
    since = datetime.fromtimestamp(float(lease["acquired_at"])).strftime("%H:%M:%S")
    return (f"held by {lease.get('holder')} since {since} "
            f"({lease.get('reason') or 'no reason given'}), {left}s left")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["acquire", "release", "status"])
    parser.add_argument("--reason", default="", help="what the card is being used for")
    parser.add_argument("--ttl", type=parse_duration, default="45m",
                        help="how long before it releases itself (default 45m)")
    parser.add_argument("--lease-file", type=Path, default=None,
                        help=f"defaults to the prod lease at {PROD_DATA} when it "
                             "exists, else this checkout's data/")
    parser.add_argument("--wait", type=parse_duration, default="10m",
                        help="how long to wait for an in-flight render (default 10m)")
    parser.add_argument("--free-below", type=int, default=1500,
                        help="MiB at or under which the card counts as idle")
    args = parser.parse_args()

    config = load_config(str(BASE_DIR / "config.yaml"))
    lease_file = args.lease_file or (PROD_DATA if PROD_DATA.parent.exists()
                                     else imagegen.lease_path(config))
    config.setdefault("chat", {}).setdefault("imagegen", {})["lease_file"] = str(lease_file)
    card = str((config.get("chat", {}).get("imagegen", {}) or {}).get("device", "0"))

    if args.action == "status":
        print(f"lease file : {lease_file}")
        print(f"lease      : {describe(imagegen.read_lease(config))}")
        used = card_memory_mb(card)
        print(f"GPU{card}       : {used} MiB in use" if used >= 0 else "GPU: unreadable")
        return 0

    if args.action == "release":
        imagegen.release_lease(config)
        print(f"✓ released {lease_file}")
        print("  queued images drain on their own from here.")
        return 0

    if not args.reason:
        parser.error("--reason is required for acquire, so `status` can say what "
                     "the card is doing")

    lease = imagegen.acquire_lease(config, args.reason, args.ttl)
    print(f"✓ lease taken: {args.reason} (expires in {int(args.ttl)}s)")
    print(f"  {lease_file}")
    print("  no new render will start; waiting for any already running …")

    if not wait_for_card(card, args.free_below, args.wait):
        print(f"✗ GPU{card} still busy after {int(args.wait)}s. The lease is held, so "
              "nothing new will start — retry, or release and try later.")
        return 1

    print(f"✓ GPU{card} is free. Run your job, then: scripts/gpu_lease.py release")
    print(f"  it releases itself in {int(args.ttl)}s if you do not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
