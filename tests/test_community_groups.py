"""Kaya became a WhatsApp Community on 2026-09-29.

Two things follow, both pinned here. A group linked to the Kaya community is
group-wide memory without anybody editing a file, while any other group, or one
whose lookup failed, stays private. And a bare "@Kaya" is answered: dropping it
as empty text is how the bot looked asleep on the day of the switch.
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.memory import KeyedSessionMemory
from src.chat.scope import SHARED, scope_for_chat
from src.chat.waha_client import MockWahaClient
from src.chat.whatsapp_adapter import WhatsAppAdapter

BOT_JID = "351900000000@c.us"
COMMUNITY = "120363400000000001@g.us"
GENERAL = "351911111111-1585236524@g.us"
TRIP = "120363400000000002@g.us"
OTHER = "120363400000000003@g.us"
ALICE = "351911111111@c.us"

_seq = itertools.count(1)


class GroupsClient(MockWahaClient):
    def __init__(self, groups=None, fail=False):
        super().__init__(echo=False)
        self.groups = groups or {}
        self.fail = fail
        self.lookups = []

    def group_info(self, chat_id):
        self.lookups.append(chat_id)
        if self.fail:
            raise RuntimeError("WAHA is down")
        return self.groups.get(chat_id, {})


def make_adapter(tmp_path, client, shared_file=None, **overrides):
    config = {
        "whatsapp": {
            "bot_jid": BOT_JID,
            "group": {"respond_on_mention": True, "respond_on_reply": True},
            "contacts": {ALICE: "Alice"},
            "send_seen": False,
            "history_turns": 5,
            "message_log_dir": str(tmp_path / "live_messages"),
            "shared_chats": [GENERAL],
            "shared_communities": [COMMUNITY],
            "shared_chats_file": str(shared_file) if shared_file else None,
            **overrides,
        }
    }
    store = KeyedSessionMemory(base_dir=str(tmp_path / "sessions"), max_lines=10)
    seen = []

    def responder(message, speaker, recent_lines, scope=None, exclude_from=None):
        seen.append({"message": message, "scope": scope, "recent": list(recent_lines)})
        return f"reply:{message}"

    adapter = WhatsAppAdapter(responder, client, config, session_store=store)
    return adapter, seen


def event(chat_id, text, mention=False):
    return {
        "event": "message",
        "payload": {"id": f"c{next(_seq)}", "from": chat_id, "participant": ALICE,
                    "body": text, "notifyName": "Alice",
                    "mentionedIds": [BOT_JID] if mention else []},
    }


def logged_scopes(tmp_path):
    scopes = []
    for path in (tmp_path / "live_messages").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            scopes.append((json.loads(line)["chat_id"], json.loads(line)["scope"]))
    return scopes


def test_linked_group_becomes_shared_and_is_saved(tmp_path):
    shared_file = tmp_path / "whatsapp_shared_chats.json"
    shared_file.write_text(json.dumps({"_comment": "keep me", "shared_chats": [GENERAL],
                                       "shared_communities": [COMMUNITY]}))
    client = GroupsClient({TRIP: {"id": TRIP, "linkedParent": COMMUNITY}})
    adapter, _ = make_adapter(tmp_path, client, shared_file)

    adapter.handle_event(event(TRIP, "quem vem em fevereiro?"))

    assert TRIP in adapter.shared_chats
    assert (TRIP, SHARED) in logged_scopes(tmp_path)
    saved = json.loads(shared_file.read_text())
    assert set(saved["shared_chats"]) == {GENERAL, TRIP}
    assert saved["_comment"] == "keep me"
    assert saved["shared_communities"] == [COMMUNITY]


def test_lookup_happens_once_per_group(tmp_path):
    client = GroupsClient({TRIP: {"linkedParent": COMMUNITY}})
    adapter, _ = make_adapter(tmp_path, client)
    for _ in range(3):
        adapter.handle_event(event(TRIP, "mais uma"))
    adapter.handle_event(event(GENERAL, "no geral"))
    assert client.lookups == [TRIP]


def test_other_community_and_unlinked_groups_stay_private(tmp_path):
    client = GroupsClient({OTHER: {"linkedParent": "120363499999999999@g.us"}})
    adapter, _ = make_adapter(tmp_path, client)
    adapter.handle_event(event(OTHER, "grupo do trabalho"))
    adapter.handle_event(event(TRIP, "grupo sem pai"))
    assert OTHER not in adapter.shared_chats and TRIP not in adapter.shared_chats
    scopes = dict(logged_scopes(tmp_path))
    assert scopes[OTHER] == scope_for_chat(OTHER) != SHARED
    assert scopes[TRIP] == scope_for_chat(TRIP) != SHARED


def test_failed_lookup_stays_private_and_retries_later(tmp_path):
    shared_file = tmp_path / "whatsapp_shared_chats.json"
    shared_file.write_text(json.dumps({"shared_chats": [GENERAL]}))
    client = GroupsClient({TRIP: {"linkedParent": COMMUNITY}}, fail=True)
    adapter, _ = make_adapter(tmp_path, client, shared_file)

    adapter.handle_event(event(TRIP, "primeira"))
    adapter.handle_event(event(TRIP, "segunda"))
    assert TRIP not in adapter.shared_chats
    assert client.lookups == [TRIP]
    assert json.loads(shared_file.read_text())["shared_chats"] == [GENERAL]

    client.fail = False
    adapter._community_checked[TRIP] -= adapter.community_recheck_seconds + 1
    adapter.handle_event(event(TRIP, "terceira"))
    assert TRIP in adapter.shared_chats


def test_the_community_itself_is_shared(tmp_path):
    client = GroupsClient({COMMUNITY: {"isCommunity": True}})
    adapter, _ = make_adapter(tmp_path, client)
    adapter.handle_event(event(COMMUNITY, "aviso"))
    assert COMMUNITY in adapter.shared_chats


def test_dm_is_never_looked_up(tmp_path):
    client = GroupsClient()
    adapter, _ = make_adapter(tmp_path, client)
    adapter.handle_event(event(ALICE, "olá"))
    assert client.lookups == []


def test_bare_mention_is_answered_about_the_room(tmp_path):
    client = GroupsClient()
    adapter, seen = make_adapter(tmp_path, client, bare_mention_text="E tu, o que achas?")
    adapter.handle_event(event(GENERAL, "vamos criar uma comunidade"))
    result = adapter.handle_event(event(GENERAL, "@351900000000", mention=True))

    assert result is not None
    assert seen[-1]["message"] == "E tu, o que achas?"
    assert any("comunidade" in line for line in seen[-1]["recent"])
    assert not any(line.startswith("Alice:") and "o que achas" in line
                   for line in adapter.session_store.recent(GENERAL, 10))


def test_bare_text_without_mention_is_still_ignored(tmp_path):
    client = GroupsClient()
    adapter, seen = make_adapter(tmp_path, client)
    assert adapter.handle_event(event(GENERAL, "   ")) is None
    assert seen == []
