"""Unit tests for WhatsApp 👍/👎 emoji-reaction feedback (no GPU/network).

Mirrors test_whatsapp_adapter.py: a stub responder + MockWahaClient drive the full
inbound→reply→reaction flow against a temp session dir.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.memory import KeyedSessionMemory
from src.chat.waha_client import MockWahaClient, extract_sent_id
from src.chat.whatsapp_adapter import (
    WhatsAppAdapter,
    parse_waha_reaction,
    reaction_rating,
)

BOT_JID = "351900000000@c.us"
ALICE = "351911111111@c.us"


def make_adapter(tmp_path, **overrides):
    config = {
        "whatsapp": {
            "bot_jid": BOT_JID,
            "send_seen": False,
            "whitelist": {"enabled": False},
            "feedback": {"enabled": True},
            **overrides,
        }
    }
    store = KeyedSessionMemory(base_dir=str(tmp_path / "sessions"), max_lines=10)
    client = MockWahaClient(echo=False)
    adapter = WhatsAppAdapter(lambda m, s, r: "Olá!", client, config, session_store=store)
    return adapter, client


def dm(text):
    return {
        "event": "message",
        "payload": {"id": "u1", "from": ALICE, "body": text, "notifyName": "Alice"},
    }


def reaction(target_id, emoji, sender=ALICE, from_me=False):
    return {
        "event": "message.reaction",
        "payload": {
            "from": sender,
            "participant": sender,
            "fromMe": from_me,
            "reaction": {"text": emoji, "messageId": target_id},
        },
    }


# ── emoji → rating ────────────────────────────────────────────────────────────
def test_reaction_rating_mapping():
    assert reaction_rating("👍") == "up"
    assert reaction_rating("👎") == "down"
    assert reaction_rating("👍🏽") == "up"  # skin-tone variant
    assert reaction_rating("❤️") is None
    assert reaction_rating("") is None  # removed reaction


# ── parsing ───────────────────────────────────────────────────────────────────
def test_parse_ignores_non_reaction():
    assert parse_waha_reaction({"event": "message", "payload": {}}) is None


def test_parse_reaction_fields():
    p = parse_waha_reaction(reaction("MSG1", "👍"))
    assert p["target_msg_id"] == "MSG1"
    assert p["emoji"] == "👍"
    assert p["chat_id"] == ALICE


def test_parse_reaction_flattened_payload():
    # Some engines flatten the reaction onto the payload instead of nesting it.
    event = {"event": "reaction", "payload": {"from": ALICE, "text": "👎", "messageId": "M2"}}
    p = parse_waha_reaction(event)
    assert p["emoji"] == "👎" and p["target_msg_id"] == "M2"


# ── sent-message tracking + attribution ──────────────────────────────────────
def test_extract_sent_id_shapes():
    assert extract_sent_id("abc") == "abc"
    assert extract_sent_id({"id": "abc"}) == "abc"
    assert extract_sent_id({"id": {"_serialized": "x@c.us_AAA"}}) == "x@c.us_AAA"
    assert extract_sent_id(None) == ""


def test_reply_is_tracked_then_reaction_attributed(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    adapter.handle_event(dm("oi"), system_prompt="sp")
    sent_id = next(iter(adapter.sent_messages))
    fb = adapter.handle_reaction(reaction(sent_id, "👍"))
    assert fb["rating"] == "up"
    assert fb["user_text"] == "oi"
    assert fb["reply"] == "Olá!"
    assert fb["is_group"] is False


def test_thumbs_down_attributed(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    adapter.handle_event(dm("oi"), system_prompt="sp")
    sent_id = next(iter(adapter.sent_messages))
    assert adapter.handle_reaction(reaction(sent_id, "👎"))["rating"] == "down"


def test_reaction_on_unknown_message_ignored(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    assert adapter.handle_reaction(reaction("NOPE", "👍")) is None


def test_removed_and_non_thumbs_ignored(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    adapter.handle_event(dm("oi"), system_prompt="sp")
    sent_id = next(iter(adapter.sent_messages))
    assert adapter.handle_reaction(reaction(sent_id, "")) is None
    assert adapter.handle_reaction(reaction(sent_id, "🎉")) is None


def test_bot_own_reaction_ignored(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    adapter.handle_event(dm("oi"), system_prompt="sp")
    sent_id = next(iter(adapter.sent_messages))
    assert adapter.handle_reaction(reaction(sent_id, "👍", from_me=True)) is None


def test_feedback_disabled_skips_tracking(tmp_path):
    adapter, _ = make_adapter(tmp_path, feedback={"enabled": False})
    adapter.handle_event(dm("oi"), system_prompt="sp")
    assert len(adapter.sent_messages) == 0
    assert adapter.handle_reaction(reaction("any", "👍")) is None


def test_sent_messages_bounded(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    adapter.sent_messages_max = 3
    for i in range(5):
        adapter._remember_sent(f"id-{i}", {"chat_id": ALICE})
    assert len(adapter.sent_messages) == 3
    assert list(adapter.sent_messages) == ["id-2", "id-3", "id-4"]
