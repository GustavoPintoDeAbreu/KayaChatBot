#!/usr/bin/env python3
"""Does the bot read the group's links right? The gate before `chat.links` goes on.

Takes every link the group has shared (the prod shared log), reads each one the
way the bot would, and reports what came back per site. Then a person rates the
synopses on a local page against the real page. Links go live only if at least
90% are rated right and none invents anything. A paywalled page that yields only
its title is a pass: saying less is not saying something wrong.

    scripts/eval_links.py --run [--limit 40]     # read the links, write items
    scripts/eval_links.py --serve [stamp]        # rate on http://127.0.0.1:8766
    scripts/eval_links.py --score [stamp]        # the verdict

Nothing leaves the box but the URLs themselves; the synopses are local.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chat import links
from src.config_loader import load_config
from src.testing import review_page

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS = BASE_DIR / "reports" / "links"
PROD_LOG = Path.home() / "kaya-prod" / "data" / "live_messages" / "shared.jsonl"
PASS_RATE = 0.90


def shared_urls(path: Path) -> List[Dict[str, str]]:
    found, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        text = re.sub(r"\n\[Link:.*", "", row.get("text") or "")
        for url in links.extract_urls(text, 5):
            if url not in seen:
                seen.add(url)
                found.append({"url": url, "sender": row.get("sender", ""),
                              "text": text[:200]})
    return found


def run(config: Dict[str, Any], log: Path, limit: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = REPORTS / stamp
    run_dir.mkdir(parents=True)
    config = {**config, "chat": {**config.get("chat", {}),
                                 "links": {**links.settings(config),
                                           "cache_dir": str(run_dir / "cache")}}}
    urls = shared_urls(log)[:limit] if limit else shared_urls(log)
    items, outcomes = [], defaultdict(Counter)
    started = time.time()
    for number, entry in enumerate(urls, start=1):
        before = time.time()
        content = links.read(entry["url"], config)
        elapsed = time.time() - before
        words = len(content.text.split())
        outcome = ("unreadable" if not content.ok else
                   "title only" if words < 15 else "text")
        outcomes[content.site or "?"][outcome] += 1
        line = links.render(content) if content.ok else "(nada lido)"
        print(f"[{number}/{len(urls)}] {elapsed:4.1f}s {content.kind or '-':<8} "
              f"{words:>5}w {line[:110]}")
        items.append({
            "id": str(number), "title": entry["url"], "link": entry["url"],
            "context": [f"{entry['sender']}: {entry['text']}",
                        f"extraído ({content.kind}, {words} palavras): "
                        + " ".join(content.text.split()[:80])],
            "candidates": [{"key": "link", "text": line}],
            "kind": content.kind, "words": words, "seconds": round(elapsed, 1)})
    (run_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"\n{len(items)} links in {time.time() - started:.0f}s → {run_dir}\n")
    for site, counts in sorted(outcomes.items(), key=lambda pair: -sum(pair[1].values())):
        print(f"  {site:<24} " + ", ".join(f"{name} {count}" for name, count in counts.items()))
    return run_dir


def latest(stamp: str | None) -> Path:
    if stamp:
        return REPORTS / stamp
    runs = sorted(path for path in REPORTS.iterdir() if (path / "items.json").exists())
    if not runs:
        raise SystemExit("no link eval yet: run with --run first")
    return runs[-1]


def verdict(run_dir: Path) -> int:
    items = json.loads((run_dir / "items.json").read_text(encoding="utf-8"))
    result = review_page.score(run_dir).get("link")
    if not result:
        print("Nothing rated yet.")
        return 1
    seconds = sorted(item["seconds"] for item in items)
    print(f"right {result['right_rate']:.0%} ({result['right']}/{result['n']}), "
          f"partly {result['partly']}, wrong {result['wrong']}, invented {result['invented']}; "
          f"median read {seconds[len(seconds) // 2]}s, slowest {seconds[-1]}s")
    if result["n"] >= len(items) and result["right_rate"] >= PASS_RATE and not result["invented"]:
        print(f"\nPASS: ≥{PASS_RATE:.0%} right with nothing invented. chat.links.enabled may go on.")
        return 0
    print(f"\nNOT YET: needs every link rated, ≥{PASS_RATE:.0%} right and nothing invented.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--serve", nargs="?", const="", metavar="STAMP")
    mode.add_argument("--score", nargs="?", const="", metavar="STAMP")
    parser.add_argument("--log", type=Path, default=PROD_LOG)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    if args.run:
        run(load_config(str(BASE_DIR / "config.yaml")), args.log, args.limit)
        return 0
    if args.serve is not None:
        review_page.serve(
            latest(args.serve or None), "Links do Kaya",
            "Abre cada link: a linha diz o que a página diz? Só o título certo (um "
            "paywall) conta como 'right'; 'partly' se falhar o essencial; marca "
            "'inventa algo' se afirmar alguma coisa que a página não diz.", port=args.port)
        return 0
    return verdict(latest(args.score or None))


if __name__ == "__main__":
    raise SystemExit(main())
