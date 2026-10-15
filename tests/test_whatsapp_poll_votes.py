"""The adapter's side of poll votes: recording, status, and the reminders.

No GPU/model/network: WAHA is ``MockWahaClient``, the registry and the message
log live in a temp dir, and the router's decision is faked by a responder that
returns a ``Reply`` with a ``Route``. This is the PC half of the feature; the
pure helpers are covered in ``test_poll_votes.py``.
"""
import datetime
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.memory import KeyedSessionMemory
from src.chat.poll_votes import PollRegistry
from src.chat.router import Route
from src.chat.waha_client import MockWahaClient
from src.chat.whatsapp_adapter import WhatsAppAdapter
from src.data.message_log import MessageLog

LISBON = ZoneInfo("Europe/Lisbon")
BOT_JID = "351900000000@c.us"
GROUP = "12036300000000@g.us"
ALICE = "351911111111@c.us"
BOB = "351922222222@c.us"
REPORT = "351999999999@c.us"
POLL_ID = "true_" + GROUP + "_key1"


class _Reply:
    """The shape ``complete()`` reads off a routed reply (src/chat/engine.py).

    A real ``Reply`` would do, but importing ``src.chat.engine`` pulls in the
    inference backends (requests, torch) which this stdlib-only test must not.
    """

    def __init__(self, route):
        self.text = ""
        self.route = route
        self.citation = ""
        self.telemetry = {}


def _vote_event(voter=ALICE, options=("sim",), poll_from_me=True,
                vote_from_me=False, timestamp=1_700_000_000, event="poll.vote"):
    return {
        "event": event,
        "payload": {
            "poll": {"id": POLL_ID, "fromMe": poll_from_me, "from": GROUP},
            "vote": {"participant": voter, "to": GROUP,
                     "selectedOptions": list(options), "fromMe": vote_from_me,
                     "timestamp": timestamp * 1000},
        },
    }


def make_adapter(tmp_path, **overrides):
    """An adapter wired for polls: a real registry, a real message log, two members."""
    config = {
        "whatsapp": {
            "bot_jid": BOT_JID,
            "group": {"respond_on_mention": True, "respond_on_reply": True},
            "contacts": {ALICE: "Alice", BOB: "Bob"},
            "member_aliases": {"alice": "Alice", "bob": "Bob"},
            "shared_chats": [GROUP],
            "send_seen": False,
            "history_turns": 5,
            "poll_votes": {"enabled": True, "store": "votes.json",
                           "electorate_days": 60, "status_max_age_hours": 168,
                           "remind": {"enabled": True, "dry_run": True,
                                      "interval_hours": 6, "stop_after_hours": 24,
                                      "min_gap_hours": 3,
                                      "send_between": ["00:00", "23:59"]}},
            **overrides,
        }
    }
    store = KeyedSessionMemory(base_dir=str(tmp_path / "sessions"), max_lines=10)
    client = MockWahaClient(echo=False)
    log = MessageLog(str(tmp_path / "log"))
    registry = PollRegistry(tmp_path / "votes.json")

    def responder(message, speaker, recent_lines, scope=None, exclude_from=None):
        return f"reply[{speaker}|{len(recent_lines)}]:{message}"

    adapter = WhatsAppAdapter(responder, client, config, session_store=store,
                              message_log=log, poll_registry=registry)
    return adapter, client, log, registry


def seed_log(log, name, jid, phone, ts):
    log.append(chat_id=GROUP, message_id=f"{name}{ts}", sender=name,
               text="olá", timestamp=ts, scope="shared",
               sender_id=jid, sender_phone=phone)


def test_vote_on_own_poll_is_recorded(tmp_path):
    adapter, client, log, registry = make_adapter(tmp_path)
    result = adapter.handle_poll_vote(_vote_event())
    assert result is not None
    assert result["recorded"] is True
    assert result["voter"] == "Alice"
    assert result["chat_id"] == GROUP
    record = registry.get("key1")
    assert record is not None
    assert record["votes"]["351911111111"]["name"] == "Alice"
    # a vote is bookkeeping, not a message: nothing reaches the log or the wire
    assert list(log.read("shared")) == []
    assert client.sent == []


def test_vote_on_member_poll_is_ignored(tmp_path):
    adapter, client, log, registry = make_adapter(tmp_path)
    result = adapter.handle_poll_vote(_vote_event(poll_from_me=False))
    assert result is not None
    assert result["recorded"] is False
    assert registry.get("key1") is None


def test_vote_never_reaches_message_log_or_session(tmp_path):
    adapter, client, log, registry = make_adapter(tmp_path)
    adapter.handle_poll_vote(_vote_event())
    assert list(log.read("shared")) == []
    assert adapter.session_store.recent(GROUP, 10) == []


def test_poll_vote_failed_is_acknowledged_not_raised(tmp_path):
    adapter, client, log, registry = make_adapter(tmp_path)
    result = adapter.handle_poll_vote(_vote_event(event="poll.vote.failed"))
    assert result == {"poll_vote_failed": True}
    assert registry.get("key1") is None


def test_disabled_registry_ignores_votes(tmp_path):
    adapter, client, log, registry = make_adapter(
        tmp_path, poll_votes={"enabled": False})
    # enabled is False, so the server would inject no registry at all
    adapter.poll_registry = None
    assert adapter.handle_poll_vote(_vote_event()) is None


def test_poll_status_command_answers_from_votes(tmp_path):
    adapter, client, log, registry = make_adapter(tmp_path)
    # the poll is young: status only answers for polls within status_max_age_hours
    created = int(time.time()) - 3600
    seed_log(log, "Alice", ALICE, "351911111111", created - 3600)
    seed_log(log, "Bob", BOB, "351922222222", created - 3500)
    registry.register(GROUP, "key1", ["sim", "não"], created, source="explicit")
    alice_vote = _vote_event(voter=ALICE, timestamp=created + 100)
    adapter.handle_poll_vote(alice_vote)

    def poll_status_responder(message, speaker, recent_lines, scope=None, exclude_from=None):
        return _Reply(Route(mode="factual", command="poll_status"))

    adapter.responder = poll_status_responder
    event = {"event": "message", "payload": {
        "id": "g1", "from": GROUP, "participant": BOB, "body": "quem está a ganhar?",
        "notifyName": "Bob", "mentionedIds": [BOT_JID]}}
    result = adapter.handle_event(event)
    assert result is not None
    assert result["command"] == "poll_status"
    assert "Alice" in result["reply"]
    assert "1 pessoas votaram" in result["reply"]
    assert "Falta votar: Bob" in result["reply"]


def test_poll_status_without_poll_says_so(tmp_path):
    adapter, client, log, registry = make_adapter(tmp_path)
    seed_log(log, "Alice", ALICE, "351911111111", 1_699_000_000)

    def poll_status_responder(message, speaker, recent_lines, scope=None, exclude_from=None):
        return _Reply(Route(mode="factual", command="poll_status"))

    adapter.responder = poll_status_responder
    event = {"event": "message", "payload": {
        "id": "g2", "from": GROUP, "participant": ALICE, "body": "como está a sondagem?",
        "notifyName": "Alice", "mentionedIds": [BOT_JID]}}
    result = adapter.handle_event(event)
    assert result["command"] == "poll_status"
    assert result["reply"] == adapter.command_replies["poll_none"]


def test_poll_status_reads_only_this_chat(tmp_path):
    adapter, client, log, registry = make_adapter(tmp_path)
    seed_log(log, "Alice", ALICE, "351911111111", 1_699_000_000)
    # a poll in a DIFFERENT chat must not be the answer here
    registry.register("999@g.us", "other", ["a"], 1_700_000_000, source="explicit")

    def poll_status_responder(message, speaker, recent_lines, scope=None, exclude_from=None):
        return _Reply(Route(mode="factual", command="poll_status"))

    adapter.responder = poll_status_responder
    event = {"event": "message", "payload": {
        "id": "g3", "from": GROUP, "participant": ALICE, "body": "quem está a ganhar?",
        "notifyName": "Alice", "mentionedIds": [BOT_JID]}}
    result = adapter.handle_event(event)
    assert result["reply"] == adapter.command_replies["poll_none"]


def test_due_reminder_mentions_missing_and_marks_slot(tmp_path):
    adapter, client, log, registry = make_adapter(tmp_path)
    seed_log(log, "Alice", ALICE, "351911111111", 1_699_000_000)
    seed_log(log, "Bob", BOB, "351922222222", 1_699_000_010)
    created = 1_700_000_000
    registry.register(GROUP, "key1", ["sim", "não"], created, source="explicit")
    adapter.handle_poll_vote(_vote_event(voter=ALICE, timestamp=1_700_000_100))

    # 7h after creation: the first 6h slot is due
    now = datetime.datetime.fromtimestamp(created + 7 * 3600, LISBON)
    due = adapter.due_poll_reminders(now)
    assert len(due) == 1
    poll, slot, missing = due[0]
    assert slot == created + 6 * 3600
    assert [v.name for v in missing] == ["Bob"]

    # a real run (dry_run off) @-mentions Bob in the group
    adapter.send_poll_reminder(poll, slot, missing, dry_run_to="")
    sent = client.sent[-1]
    assert sent["chat_id"] == GROUP
    assert sent.get("mentions") == [BOB]
    assert "@351922222222" in sent["text"]
    # the group sees the bot said it: the reminder is appended to the session store
    last_line = adapter.session_store.recent(GROUP, 1)[-1]
    assert last_line.startswith("Kaya Bot: ")
    assert "Bob" in last_line
    # the slot is marked, so the same slot is not due again
    assert slot in registry.get("key1")["reminded_slots"]
    assert adapter.due_poll_reminders(now) == []


def test_dry_run_reminder_goes_to_report_jid_without_mentions(tmp_path):
    adapter, client, log, registry = make_adapter(tmp_path)
    seed_log(log, "Alice", ALICE, "351911111111", 1_699_000_000)
    seed_log(log, "Bob", BOB, "351922222222", 1_699_000_010)
    created = 1_700_000_000
    registry.register(GROUP, "key1", ["sim", "não"], created, source="explicit")
    adapter.handle_poll_vote(_vote_event(voter=ALICE, timestamp=1_700_000_100))

    now = datetime.datetime.fromtimestamp(created + 7 * 3600, LISBON)
    poll, slot, missing = adapter.due_poll_reminders(now)[0]
    result = adapter.send_poll_reminder(poll, slot, missing, dry_run_to=REPORT)
    assert result["dry_run"] is True
    sent = client.sent[-1]
    assert sent["chat_id"] == REPORT
    assert "mentions" not in sent
    assert "[ensaio, para " + GROUP + "]" in sent["text"]
    assert "Bob" in sent["text"]


def test_everyone_voted_closes_without_message(tmp_path):
    adapter, client, log, registry = make_adapter(tmp_path)
    seed_log(log, "Alice", ALICE, "351911111111", 1_699_000_000)
    seed_log(log, "Bob", BOB, "351922222222", 1_699_000_010)
    created = 1_700_000_000
    registry.register(GROUP, "key1", ["sim", "não"], created, source="explicit")
    adapter.handle_poll_vote(_vote_event(voter=ALICE, timestamp=1_700_000_100))
    adapter.handle_poll_vote(_vote_event(voter=BOB, timestamp=1_700_000_200))

    now = datetime.datetime.fromtimestamp(created + 7 * 3600, LISBON)
    assert adapter.due_poll_reminders(now) == []
    assert registry.get("key1")["closed"] == "all_voted"
    assert client.sent == []


def test_empty_electorate_skips_the_slot_but_does_not_close(tmp_path):
    # nobody has spoken in the chat yet: there is no electorate to remind, so the
    # slot is marked reminded and skipped (not closed) — a poll auto-registered
    # from its first vote can still get reminders on later slots once people
    # start writing.
    adapter, client, log, registry = make_adapter(tmp_path)
    created = 1_700_000_000
    registry.register(GROUP, "key1", ["sim", "não"], created, source="explicit")
    now = datetime.datetime.fromtimestamp(created + 7 * 3600, LISBON)
    assert adapter.due_poll_reminders(now) == []
    # the slot is marked reminded, but the poll is NOT closed
    assert (created + 6 * 3600) in registry.get("key1")["reminded_slots"]
    assert registry.get("key1")["closed"] is None
    assert client.sent == []

    # once someone writes, a later slot can still remind them
    seed_log(log, "Alice", ALICE, "351911111111", created + 7 * 3600)
    later = datetime.datetime.fromtimestamp(created + 13 * 3600, LISBON)
    due = adapter.due_poll_reminders(later)
    assert len(due) == 1
    poll, slot, missing = due[0]
    assert slot == created + 12 * 3600
    assert [v.name for v in missing] == ["Alice"]
