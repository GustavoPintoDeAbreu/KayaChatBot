#!/usr/bin/env python3
"""What a recap would read for a topic, without any LLM call.

Tunes the topic filter (``src/chat/recap.py``) on the real log: prints the stats,
the transcript block and the instruction the model would get. The output is group
data and is printed to the local terminal only.

    scripts/recap_preview.py --topic "almoço de natal"
    scripts/recap_preview.py --topic "viagem" --scope dm:351... --semantic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chat import recap, tally
from src.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", required=True, help="The subject, as the router's Q: line names it.")
    parser.add_argument("--scope", default="shared", help="The chat's scope (default shared).")
    parser.add_argument("--semantic", action="store_true",
                        help="Also run the semantic half (loads the retriever and bge-m3).")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    retriever = None
    if args.semantic:
        from src.chat.retriever import get_retriever

        retriever = get_retriever(config)
    block, instruction, stats = recap.gather(config, args.topic, args.scope,
                                             retriever=retriever,
                                             resolver=tally.load_resolver(config))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print()
    print(block or "(no transcript)")
    print()
    print(instruction)


if __name__ == "__main__":
    main()
