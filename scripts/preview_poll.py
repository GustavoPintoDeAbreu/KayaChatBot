"""Preview the poll the bot would build for a chat, without sending anything.

The gate for turning ``chat.poll.enabled`` on: it runs the exact same path as
production (the same log read, the same prompt, the same parse and grounding)
against a real chat, and prints the draft a person can judge. Nothing is sent,
nothing is written.

Usage:
  kaya_chatbot_env/bin/python scripts/preview_poll.py --list-chats
  kaya_chatbot_env/bin/python scripts/preview_poll.py --chat 12036...@g.us \
      --topic "restaurante para o almoço de Natal"
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from src.config_loader import load_config
from src.chat import polls


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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-chats", action="store_true",
                    help="List the chat ids found in the log and exit.")
    ap.add_argument("--chat", default="", help="The chat JID to build the poll for.")
    ap.add_argument("--scope", default="shared", help="Scope file to read.")
    ap.add_argument("--topic", default="", help="What the poll is about.")
    ap.add_argument("--request", default="",
                    help="The asker's words; defaults to the topic.")
    ap.add_argument("--days", type=int, default=0,
                    help="Lookback in days; 0 uses the configured lookback_days.")
    ap.add_argument("--log-dir", default="",
                    help="Message log dir; defaults to the configured one.")
    args = ap.parse_args()

    config = load_config(str(BASE / "config.yaml"))
    pcfg = polls.settings(config)
    log_dir = args.log_dir or polls.log_dir(config)

    from src.data.message_log import MessageLog

    log = MessageLog(log_dir)
    chats: dict = {}
    for rec in log.read(args.scope, after_ts=0):
        chat_id = rec.get("chat_id") or "?"
        ts = int(rec.get("timestamp") or 0)
        entry = chats.get(chat_id)
        if entry is None:
            entry = {"count": 0, "first": ts, "last": ts}
            chats[chat_id] = entry
        entry["count"] += 1
        if ts:
            entry["first"] = min(entry["first"], ts)
            entry["last"] = max(entry["last"], ts)

    def _fmt(ts: int) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "?"

    if args.list_chats:
        for chat_id in sorted(chats):
            entry = chats[chat_id]
            print(f"{entry['count']:>6}  {_fmt(entry['first'])} .. {_fmt(entry['last'])}  {chat_id}")
        return

    if not args.chat:
        ap.error("--chat is required (or use --list-chats)")
    if not args.topic:
        ap.error("--topic is required")

    lookback_days = args.days if args.days > 0 else pcfg["lookback_days"]
    since_ts = (int(datetime.now().timestamp())
                - lookback_days * 86400) if lookback_days > 0 else 0
    records = polls.load_chat_messages(
        log_dir, args.scope, args.chat, since_ts, pcfg["max_messages"])
    print(f"read {len(records)} message(s) from {args.chat} "
          f"(scope {args.scope}, {lookback_days} day(s) back)")
    if not records:
        print("nothing to build a poll from")
        return

    backend = build_server_backend(config)
    # The engine runs the poll at {**config["inference"], "temperature": …} —
    # the same sampling base the rest of the reply uses, with the poll's own
    # temperature on top. Mirror it here, or the gate is not the production path.
    inf = config.get("inference", {}) or {}
    draft = polls.draft_poll(
        lambda messages, max_new_tokens: backend.generate(
            messages, max_new_tokens=max_new_tokens,
            sampling={**inf, "temperature": pcfg["temperature"]}),
        topic=args.topic,
        request=args.request or args.topic,
        records=records,
        settings=pcfg,
    )

    out = {
        "status": draft.status,
        "question": draft.question,
        "options": draft.options,
        "dropped": [{"option": name, "reason": reason}
                    for name, reason in draft.dropped],
        "invented": draft.invented,
        "messages_read": draft.messages_read,
        "chunks": draft.chunks,
        "left_out": polls.left_out_line(
            draft.dropped, "Deixei de fora: {items}.", pcfg["max_dropped_listed"]),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
