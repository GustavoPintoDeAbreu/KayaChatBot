#!/usr/bin/env python3
"""Re-route a logged session with the history it actually had, and diff.

The conversation probe scores made-up cases. This scores the real ones: it takes
the messages a chat actually sent (``data/live_messages/<scope>.jsonl``), rebuilds
the interleaved history each of them arrived with — every message the room sent,
plus the bot's own replies from the interaction log, in timestamp order — and asks
the router to classify them again. The result is diffed against the ``route_mode``
that was recorded live.

That reconstruction is the point. The live log was produced by a bot whose history
held only the turns it answered, and whose router saw two lines of it. Both changed
on 2026-08-17, and the only honest way to show what that buys is to replay the same
messages against the same model.

    KAYA_INFERENCE_BACKEND=gguf KAYA_LLAMA_URL=http://127.0.0.1:8081 \
      kaya_chatbot_env/bin/python scripts/replay_routing.py --date 2026-08-17

Generation only ever produces a routing label, so this is cheap — one short call
per message. It still occupies whichever llama-server it is pointed at: use the
bench server, not the one answering the group.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import router
from src.config_loader import load_config

BOT = "Kaya Bot"


def _iso_to_epoch(value: str) -> float:
    """Seconds since the epoch for an ISO timestamp, tz-aware or not."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def load_timeline(messages_path: Path, interactions_path: Path, date: str):
    """Every line of the chat that day, oldest first, as (epoch, who, text, row).

    ``row`` is the interaction record for a message the bot answered, and None for
    everything else — the chatter it saw and the replies it wrote.
    """
    timeline = []
    for line in messages_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        stamp = item.get("timestamp")
        if not isinstance(stamp, (int, float)):
            continue
        when = datetime.fromtimestamp(stamp, tz=timezone.utc)
        if date and when.strftime("%Y-%m-%d") != date:
            continue
        timeline.append([float(stamp), item.get("sender") or "?",
                         (item.get("text") or "").strip(), None])

    answered = []
    for line in interactions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("timestamp"):
            continue
        if date and not str(row["timestamp"]).startswith(date):
            continue
        answered.append(row)

    # The interaction log records when the REPLY was finished, so matching on time
    # alone drifts by however long the reply took. Matching on text alone does not
    # work either: the message log keeps "@Kaya Bot diz mal do Gil" while the
    # interaction log holds the mention-stripped "diz mal do Gil", sometimes with a
    # quoted parent glued above it. So: containment either way, nearest message at
    # or before the reply, each used once.
    def _norm(value: str) -> str:
        return " ".join((value or "").lower().split())

    taken = set()
    for row in answered:
        asked = _norm(row.get("user_message"))
        if not asked:
            continue
        finished = _iso_to_epoch(row["timestamp"])
        best = None
        for index, entry in enumerate(timeline):
            if index in taken or entry[0] > finished:
                continue
            text = _norm(entry[2])
            if not text or (asked not in text and text not in asked):
                continue
            if best is None or entry[0] > timeline[best][0]:
                best = index
        if best is not None:
            taken.add(best)
            timeline[best][3] = row

    # The bot's own replies, placed at the moment they were logged.
    for row in answered:
        reply = (row.get("assistant_response") or "").strip()
        if reply:
            timeline.append([_iso_to_epoch(row["timestamp"]), BOT, reply, None])

    timeline.sort(key=lambda entry: entry[0])
    return timeline


def build_server_backend(config):
    """A generation backend without loading the model into this process.

    The llama.cpp backend needs the tokenizer to apply the chat template, and
    nothing else — the weights live in the server. Same trick ``engine._load_model``
    uses when the backend is ``gguf``.
    """
    from transformers import AutoTokenizer

    from src.chat.inference_backend import build_backend

    tokenizer = AutoTokenizer.from_pretrained(config["training"]["output_dir"])
    return build_backend(config, None, tokenizer)


def main() -> None:
    base = Path(__file__).parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="", help="YYYY-MM-DD; empty replays everything.")
    ap.add_argument("--messages", default=str(base / "data/live_messages/shared.jsonl"))
    ap.add_argument("--log", default=str(base / "data/feedback/live_interactions.jsonl"))
    ap.add_argument("--history-lines", type=int, default=60,
                    help="Verbatim window, matching whatsapp.history_turns.")
    ap.add_argument("--out", default="", help="Write the per-message rows as JSON.")
    args = ap.parse_args()

    config = load_config(str(base / "config.yaml"))
    timeline = load_timeline(Path(args.messages), Path(args.log), args.date)
    replayed = [entry for entry in timeline if entry[3] is not None]
    if not replayed:
        print("nothing to replay — check --date and the two log paths")
        return

    try:
        from src.chat.retriever import get_retriever

        retriever = get_retriever(config)
    except Exception as exc:  # noqa: BLE001 — the reconcile step is optional here
        print(f"⚠️  no retriever ({exc}); reporting the raw router decision only")
        retriever = None

    backend = build_server_backend(config)
    window = int(config.get("chat", {}).get("router", {}).get("context_lines", 6))

    rows, changed = [], 0
    print(f"{'was':<9}{'now':<9}{'q?':<4}  message")
    print("-" * 100)
    for index, (_, who, text, record) in enumerate(timeline):
        if record is None:
            continue
        history = [f"{line[1]}: {line[2]}" for line in timeline[:index]
                   if line[2]][-args.history_lines:]
        route = router.classify(backend, config, text, history)
        if retriever is not None:
            named = retriever.named_members(f"{text} {route.query}")
            if named:
                recent = "\n".join(history[-window:])
                route = router.reconcile(route, named, retriever.named_members(recent))

        was = record.get("route_mode") or ""
        now = route.command or route.mode
        naming = bool(retriever and route.mode == router.GENERAL
                      and retriever.named_members(f"{text} {route.query}"))
        rows.append({"speaker": who, "message": text, "was": was, "now": now,
                     "query": route.query, "reconciled_from": route.reconciled_from,
                     "general_naming_a_member": naming})
        if was != now:
            changed += 1
            print(f"{was:<9}{now:<9}{'Q' if route.query else ' ':<4}  {text[:70]}")

    was_naming = sum(1 for row in rows
                     if row["was"] == router.GENERAL and row["general_naming_a_member"])
    print("-" * 100)
    print(f"replayed {len(rows)} messages, {changed} changed mode")
    print(f"  before: {Counter(row['was'] for row in rows)}")
    print(f"  after:  {Counter(row['now'] for row in rows)}")
    print(f"  rewrites produced: {sum(1 for row in rows if row['query'])}/{len(rows)}")
    print(f"  general turns still naming a member: "
          f"{sum(1 for row in rows if row['general_naming_a_member'])} "
          f"(was {was_naming} by this measure)")

    if args.out:
        Path(args.out).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
