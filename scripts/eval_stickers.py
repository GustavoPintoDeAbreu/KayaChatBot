#!/usr/bin/env python3
"""Does the model understand the group's stickers? The gate before it is allowed to try.

Samples stickers from the WhatsApp export, each with the messages sent just
before it, and asks the serving model three ways:

    A  the photo prompt stickers get today (the baseline)
    B  a sticker prompt: visible text, what it shows, the reaction it conveys,
       with three frames of an animated sticker instead of one
    C  B plus the preceding messages: what the sender meant by it HERE

Then a person rates every answer on a local page, blind to which variant wrote
it. Understanding goes live only if the best variant is rated right at least 80%
of the time and never invents a member's name.

    scripts/eval_stickers.py --run                  # sample 60, describe, write items
    scripts/eval_stickers.py --serve [stamp]        # rate on http://127.0.0.1:8765
    scripts/eval_stickers.py --score [stamp]        # the verdict

The export's stickers are real group media, so everything stays in
reports/stickers/ (gitignored) and is served on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chat import stickers, vision
from src.config_loader import load_config
from src.testing import review_page

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS = BASE_DIR / "reports" / "stickers"
DEFAULT_MEDIA = Path.home() / "Downloads" / "WhatsApp kaya media"
DEFAULT_EXPORT = BASE_DIR / "data" / "wpp" / "WhatsApp Chat with Kaya 👀 1.txt"
PASS_RATE = 0.80

_LINE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}) - ([^:]+): (.*)$")
_STICKER = re.compile(r"(STK-\d{8}-WA\d+\.webp) \(file attached\)")
_OMITTED = re.compile(r"<Media omitted>|\(file attached\)")


def read_export(path: Path) -> List[Dict[str, str]]:
    """The export as ``[{when, sender, text}]``, continuation lines folded in."""
    messages: List[Dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = _LINE.match(raw)
        if match:
            messages.append({"when": f"{match.group(1)} {match.group(2)}",
                             "sender": match.group(3).strip(), "text": match.group(4)})
        elif messages:
            messages[-1]["text"] += "\n" + raw
    return messages


def sticker_items(messages: List[Dict[str, str]], media: Path,
                  before: int = 5) -> List[Dict[str, Any]]:
    items = []
    for index, message in enumerate(messages):
        match = _STICKER.search(message["text"])
        if not match or not (media / match.group(1)).exists():
            continue
        context = []
        for previous in reversed(messages[max(0, index - 3 * before):index]):
            if _OMITTED.search(previous["text"]):
                continue
            context.insert(0, f"{previous['sender']}: {previous['text'][:200]}")
            if len(context) == before:
                break
        items.append({"file": match.group(1), "sender": message["sender"],
                      "when": message["when"], "context": context})
    return items


def is_animated(path: Path) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as image:
            return bool(getattr(image, "is_animated", False))
    except Exception:  # noqa: BLE001
        return False


def sample(items: List[Dict[str, Any]], media: Path, count: int, seed: int) -> List[Dict[str, Any]]:
    """Stratified by animated/static and by year, one sticker file at most once."""
    rng = random.Random(seed)
    seen, unique = set(), []
    for item in items:
        if item["file"] not in seen and item["context"]:
            seen.add(item["file"])
            unique.append(item)
    strata: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for item in unique:
        year = item["file"][4:8]
        strata[(is_animated(media / item["file"]), year)].append(item)
    for bucket in strata.values():
        rng.shuffle(bucket)
    # In proportion to how many each stratum has, so the sample looks like the
    # group's actual use (mostly 2025-26), with every stratum present at least once.
    total = sum(len(bucket) for bucket in strata.values())
    chosen: List[Dict[str, Any]] = []
    for key in sorted(strata, key=str):
        share = max(1, round(count * len(strata[key]) / total))
        chosen.extend(strata[key][:share])
        del strata[key][:share]
    leftovers = [item for bucket in strata.values() for item in bucket]
    rng.shuffle(leftovers)
    chosen.extend(leftovers[:max(0, count - len(chosen))])
    rng.shuffle(chosen)
    return chosen[:count]


def run(config: Dict[str, Any], media: Path, export: Path, count: int, seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = REPORTS / stamp
    (run_dir / "img").mkdir(parents=True)
    chosen = sample(sticker_items(read_export(export), media), media, count, seed)
    items = []
    started = time.time()
    for number, item in enumerate(chosen, start=1):
        image = (media / item["file"]).read_bytes()
        shutil.copy(media / item["file"], run_dir / "img" / item["file"])
        candidates = [
            {"key": "A", "text": vision.describe_bytes(image, config, "image/webp") or ""},
            {"key": "B", "text": stickers.describe(image, config) or ""},
            {"key": "C", "text": stickers.describe(image, config, item["context"]) or ""},
        ]
        items.append({
            "id": item["file"], "title": f"{item['sender']} · {item['when']}",
            "image": f"img/{item['file']}", "context": item["context"],
            "animated": is_animated(media / item["file"]), "candidates": candidates})
        print(f"[{number}/{len(chosen)}] {item['file']} "
              f"({time.time() - started:.0f}s) B: {candidates[1]['text'][:90]}")
        (run_dir / "items.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(items)} stickers described in {time.time() - started:.0f}s → {run_dir}")
    return run_dir


def latest(stamp: str | None) -> Path:
    if stamp:
        return REPORTS / stamp
    runs = sorted(path for path in REPORTS.iterdir() if (path / "items.json").exists())
    if not runs:
        raise SystemExit("no sticker eval yet: run with --run first")
    return runs[-1]


def verdict(run_dir: Path) -> int:
    scores = review_page.score(run_dir)
    items = json.loads((run_dir / "items.json").read_text(encoding="utf-8"))
    names = {"A": "photo prompt (today)", "B": "sticker prompt", "C": "sticker + context"}
    best = None
    for key in ("A", "B", "C"):
        result = scores.get(key)
        if not result:
            print(f"{key} {names[key]:<22} not rated")
            continue
        print(f"{key} {names[key]:<22} right {result['right_rate']:.0%} "
              f"({result['right']}/{result['n']}), partly {result['partly']}, "
              f"wrong {result['wrong']}, invented {result['invented']}")
        passes = result["n"] >= len(items) and result["right_rate"] >= PASS_RATE \
            and result["invented"] == 0
        if passes and (best is None or result["right_rate"] > scores[best]["right_rate"]):
            best = key
    if best:
        print(f"\nPASS: variant {best} ({names[best]}) clears ≥{PASS_RATE:.0%} with nothing invented.")
        return 0
    print(f"\nNOT YET: no variant fully rated at ≥{PASS_RATE:.0%} with nothing invented. "
          "Stickers stay off.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--serve", nargs="?", const="", metavar="STAMP")
    mode.add_argument("--score", nargs="?", const="", metavar="STAMP")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--media", type=Path, default=DEFAULT_MEDIA)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.run:
        run(load_config(str(BASE_DIR / "config.yaml")), args.media, args.export,
            args.count, args.seed)
        return 0
    if args.serve is not None:
        review_page.serve(
            latest(args.serve or None), "Stickers do Kaya",
            "Para cada sticker: a descrição diz o que ele mostra e o que a pessoa quis "
            "dizer ao mandá-lo ali? Marca 'inventa algo' se nomear alguém do grupo ou "
            "afirmar algo que não está lá.", port=args.port)
        return 0
    return verdict(latest(args.score or None))


if __name__ == "__main__":
    raise SystemExit(main())
