"""Unit tests for the WhatsApp bridge — routing, gating, speaker, history.

No GPU/model/network: the engine is replaced by a stub ``responder`` and WAHA by
``MockWahaClient``, so this exercises the full inbound→reply logic the same way
``scripts/whatsapp_simulator.py`` does, against a temp session dir.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.memory import KeyedSessionMemory
from src.chat.waha_client import MockWahaClient
from src.chat.whatsapp_adapter import WhatsAppAdapter, parse_waha_message

BOT_JID = "351900000000@c.us"
GROUP = "12036300000000@g.us"
ALICE = "351911111111@c.us"


def make_adapter(tmp_path, **overrides):
    config = {
        "whatsapp": {
            "bot_jid": BOT_JID,
            "group": {"respond_on_mention": True, "respond_on_reply": True},
            "contacts": {"351911111111@c.us": "Alice"},
            "send_seen": False,
            "history_turns": 5,
            **overrides,
        }
    }
    store = KeyedSessionMemory(base_dir=str(tmp_path / "sessions"), max_lines=10)
    client = MockWahaClient(echo=False)

    def responder(message, speaker, recent_lines):
        return f"reply[{speaker}|{len(recent_lines)}]:{message}"

    adapter = WhatsAppAdapter(responder, client, config, session_store=store)
    return adapter, client


def dm_event(text, sender=ALICE, name="Alice", from_me=False):
    return {
        "event": "message",
        "payload": {"id": "x", "from": sender, "body": text, "notifyName": name, "fromMe": from_me},
    }


def group_event(text, sender=ALICE, name="Alice", mention=False, reply=False):
    payload = {
        "id": "g1",
        "from": GROUP,
        "participant": sender,
        "body": text,
        "notifyName": name,
        "mentionedIds": [BOT_JID] if mention else [],
    }
    if reply:
        payload["replyTo"] = {"participant": BOT_JID}
    return {"event": "message", "payload": payload}


# ── parsing ──────────────────────────────────────────────────────────────────
def test_parse_ignores_non_message():
    assert parse_waha_message({"event": "session.status", "payload": {}}) is None


def test_parse_dm_vs_group():
    dm = parse_waha_message(dm_event("hi"))
    assert dm.is_group is False and dm.sender_id == ALICE
    grp = parse_waha_message(group_event("hi", mention=True))
    assert grp.is_group is True
    assert grp.sender_id == ALICE  # participant, not the group id
    assert BOT_JID in grp.mentioned_ids


# ── DM routing: always answer ──────────────────────────────────────────────────
def test_dm_always_responds(tmp_path):
    adapter, client = make_adapter(tmp_path)
    result = adapter.handle_event(dm_event("olá"))
    assert result is not None
    assert len(client.sent) == 1
    assert client.sent[0]["chat_id"] == ALICE
    assert client.sent[0]["reply_to"] is None  # DMs are not quoted


def test_ignores_own_messages(tmp_path):
    adapter, client = make_adapter(tmp_path)
    assert adapter.handle_event(dm_event("echo", from_me=True)) is None
    assert client.sent == []


def test_ignores_empty_text(tmp_path):
    adapter, client = make_adapter(tmp_path)
    assert adapter.handle_event(dm_event("   ")) is None


# ── group routing: only when addressed ─────────────────────────────────────────
def test_group_silent_without_mention(tmp_path):
    adapter, client = make_adapter(tmp_path)
    assert adapter.handle_event(group_event("conversa random")) is None
    assert client.sent == []


def test_group_responds_on_mention(tmp_path):
    adapter, client = make_adapter(tmp_path)
    result = adapter.handle_event(group_event("@bot quem é o Rui?", mention=True))
    assert result is not None
    assert len(client.sent) == 1
    assert client.sent[0]["reply_to"] == "g1"  # group replies quote the asker


def test_group_responds_on_reply_to_bot(tmp_path):
    adapter, client = make_adapter(tmp_path)
    result = adapter.handle_event(group_event("e o Tó?", reply=True))
    assert result is not None
    assert len(client.sent) == 1


def test_group_mention_can_be_disabled(tmp_path):
    adapter, client = make_adapter(tmp_path, group={"respond_on_mention": False, "respond_on_reply": True})
    assert adapter.handle_event(group_event("oi", mention=True)) is None


# ── speaker resolution ────────────────────────────────────────────────────────
def test_speaker_from_contacts(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    result = adapter.handle_event(dm_event("oi", sender=ALICE, name="al"))
    assert "Alice" in result["reply"]  # mapped via contacts, not the push name


def test_speaker_falls_back_to_pushname(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    result = adapter.handle_event(dm_event("oi", sender="351999@c.us", name="Zé"))
    assert "Zé" in result["reply"]


def test_bot_mention_token_stripped(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    result = adapter.handle_event(group_event("@351900000000 quem ganhou?", mention=True))
    # the @<number> token is removed before reaching the model
    assert "@351900000000" not in result["reply"]
    assert "quem ganhou?" in result["reply"]


# ── history is per-chat and grows ──────────────────────────────────────────────
def test_history_accumulates_per_chat(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    adapter.handle_event(dm_event("primeira"))
    second = adapter.handle_event(dm_event("segunda"))
    # after one full exchange (2 lines), the second turn sees that history
    assert second["reply"].startswith("reply[Alice|2]")


BOT_LID = "111111111111111@lid"
USER_LID = "222222222222222@lid"
USER_PHONE = "351900000001"


def noweb_dm(text):
    """A NOWEB-shaped DM: @lid addressing, name+phone in _data."""
    return {
        "event": "message",
        "me": {"id": BOT_JID, "lid": BOT_LID},
        "payload": {
            "id": f"false_{USER_LID}_ABC",
            "from": USER_LID,
            "fromMe": False,
            "body": text,
            "_data": {
                "key": {"remoteJid": USER_LID, "remoteJidAlt": f"{USER_PHONE}@s.whatsapp.net"},
                "pushName": "Gustavo Abreu",
                "message": {"conversation": text},
            },
        },
    }


def noweb_group(text, mention_lid=None, reply_to_lid=None):
    """A NOWEB-shaped group message with nested contextInfo."""
    ext = {"text": text, "contextInfo": {}}
    if mention_lid:
        ext["contextInfo"]["mentionedJid"] = [mention_lid]
    if reply_to_lid:
        ext["contextInfo"]["participant"] = reply_to_lid
    return {
        "event": "message",
        "me": {"id": BOT_JID, "lid": BOT_LID},
        "payload": {
            "id": f"false_{GROUP}_XYZ_{USER_LID}",
            "from": GROUP,
            "participant": USER_LID,
            "fromMe": False,
            "body": text,
            "_data": {
                "key": {"participant": USER_LID, "participantAlt": f"{USER_PHONE}@s.whatsapp.net"},
                "pushName": "Gustavo Abreu",
                "message": {"extendedTextMessage": ext},
            },
        },
    }


def test_noweb_dm_parsed_and_named(tmp_path):
    adapter, client = make_adapter(tmp_path, contacts={f"{USER_PHONE}": "Gustavo"})
    result = adapter.handle_event(noweb_dm("Olá Kaya"))
    assert result is not None
    assert len(client.sent) == 1
    assert "Gustavo" in result["reply"]  # mapped via the real phone behind the @lid


def test_noweb_group_mention_by_lid(tmp_path):
    # bot_jid is the @c.us number, but NOWEB mentions the bot by its @lid (learned from me.lid)
    adapter, client = make_adapter(tmp_path)
    silent = adapter.handle_event(noweb_group("conversa qualquer"))
    assert silent is None
    hit = adapter.handle_event(noweb_group("@111111111111111 estás vivo?", mention_lid=BOT_LID))
    assert hit is not None
    assert "@111111111111111" not in hit["reply"]  # bot-lid token stripped


def test_noweb_group_reply_to_bot_lid(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    hit = adapter.handle_event(noweb_group("e depois?", reply_to_lid=BOT_LID))
    assert hit is not None


def test_ignores_stale_backlog(tmp_path):
    adapter, client = make_adapter(tmp_path)
    adapter.ignore_before_ts = 2000
    stale = dm_event("mensagem antiga")
    stale["payload"]["timestamp"] = 1000  # before the cutoff → dropped
    assert adapter.handle_event(stale) is None
    fresh = dm_event("mensagem nova")
    fresh["payload"]["timestamp"] = 3000  # after the cutoff → answered
    assert adapter.handle_event(fresh) is not None


def test_history_isolated_between_dm_and_group(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    adapter.handle_event(dm_event("dm message"))
    grp = adapter.handle_event(group_event("oi", mention=True))
    # the group's first turn must not see the DM history
    assert grp["reply"].startswith("reply[Alice|0]")
