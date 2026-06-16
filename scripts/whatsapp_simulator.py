#!/usr/bin/env python
"""Interactive WhatsApp simulator — exercise the bridge with no real number.

Feeds synthetic WAHA ``message`` events into ``WhatsAppAdapter`` and prints what
the bot would send back (via ``MockWahaClient``). Lets you verify DM vs group
routing, @-mention / reply gating, speaker mapping and per-chat history before a
real eSIM / WAHA session exists.

    # Fast: fake responder, no model/GPU needed — tests routing & history
    kaya_chatbot_env/bin/python scripts/whatsapp_simulator.py

    # Real model (loads the fine-tuned model; needs the GPU free)
    kaya_chatbot_env/bin/python scripts/whatsapp_simulator.py --real

Commands inside the REPL:
    /dm <name>     talk as a DM from <name>
    /group <name>  talk in the group as <name>
    /mention       (group) next message @-mentions the bot
    /reply         (group) next message replies to the bot's last message
    /who           show current context
    /quit          exit
A group message is also treated as addressed to the bot if it contains '@kaya'.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.chat.waha_client import MockWahaClient
from src.chat.whatsapp_adapter import WhatsAppAdapter

BOT_JID = "351900000000@c.us"
GROUP_ID = "120363000000000000@g.us"


def _phone_jid(name: str) -> str:
    """Stable fake JID per sender name so history/contacts behave consistently."""
    digits = str(abs(hash(name)) % 10_000_000_000).zfill(10)
    return f"{digits}@c.us"


def build_event(text, sender_name, is_group, mention=False, reply=False):
    sender_jid = _phone_jid(sender_name)
    chat_id = GROUP_ID if is_group else sender_jid
    mentioned = [BOT_JID] if (mention or "@kaya" in text.lower()) else []
    payload = {
        "id": f"sim_{abs(hash(text)) % 10**8}",
        "from": chat_id,
        "participant": sender_jid if is_group else None,
        "notifyName": sender_name,
        "body": text,
        "fromMe": False,
        "mentionedIds": mentioned,
    }
    if reply:
        payload["replyTo"] = {"participant": BOT_JID}
    return {"event": "message", "session": "default", "payload": payload}


def make_responder(use_real, config, config_path):
    if not use_real:
        def fake(message, speaker, recent_lines):
            return f"(mock reply to {speaker}) ouvi: “{message}” · {len(recent_lines)} linhas de histórico"
        return fake

    from src.chat.engine import get_engine, build_system_prompt

    engine = get_engine(config)
    system_prompt = build_system_prompt(
        config, config_path, include_uncensored=config.get("chat", {}).get("uncensored_mode", False)
    )

    def real(message, speaker, recent_lines):
        return engine.generate_reply(message, speaker, recent_lines, system_prompt)

    return real


def main():
    parser = argparse.ArgumentParser(description="Simulate WhatsApp messages into the Kaya bridge.")
    parser.add_argument("--real", action="store_true", help="Use the real fine-tuned model (needs GPU).")
    args = parser.parse_args()

    config_path = str(Path(__file__).parent.parent / "config.yaml")
    config = load_config(config_path)
    # Inject a known bot identity + a sample contact map so mention/speaker work.
    config.setdefault("whatsapp", {})
    config["whatsapp"].update(
        {
            "bot_jid": BOT_JID,
            "group": {"respond_on_mention": True, "respond_on_reply": True},
            "send_seen": False,
            "history_turns": 10,
        }
    )

    responder = make_responder(args.real, config, config_path)
    client = MockWahaClient(echo=False)
    adapter = WhatsAppAdapter(responder, client, config)

    is_group = False
    sender = "Gustavo"
    pending_mention = False
    pending_reply = False

    print("Kaya WhatsApp simulator. Type a message, or /help. /quit to exit.")
    print(f"Context: DM as {sender}\n")

    while True:
        try:
            line = input(f"[{'GROUP' if is_group else 'DM'}:{sender}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            print(__doc__)
            continue
        if line == "/who":
            print(f"  context={'GROUP' if is_group else 'DM'} sender={sender}")
            continue
        if line.startswith("/dm"):
            is_group = False
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                sender = parts[1]
            print(f"  → DM as {sender}")
            continue
        if line.startswith("/group"):
            is_group = True
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                sender = parts[1]
            print(f"  → GROUP as {sender}")
            continue
        if line == "/mention":
            pending_mention = True
            print("  → next group message @-mentions the bot")
            continue
        if line == "/reply":
            pending_reply = True
            print("  → next group message replies to the bot")
            continue

        event = build_event(line, sender, is_group, mention=pending_mention, reply=pending_reply)
        pending_mention = pending_reply = False
        result = adapter.handle_event(event, system_prompt="")
        if result is None:
            print("  · (bot stayed silent — not addressed)")
        else:
            print(f"  Kaya Bot → {result['reply']}")


if __name__ == "__main__":
    main()
