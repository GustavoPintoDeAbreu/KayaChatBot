"""The WhatsApp side of a poll: the draft the engine returns becomes a native
WhatsApp poll, or a fixed line when it cannot.

The engine's part (reading the log, drafting) is covered by test_engine_poll.py;
here the responder hands back a Reply whose ``poll`` is already a PollDraft, and
the adapter either sends it or says plainly why it did not. A sent poll is also
recorded in the poll store, so a later pass can track who voted what.
"""
import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import metrics, polls
from src.chat.memory import KeyedSessionMemory
from src.chat.waha_client import MockWahaClient, WahaClient
from src.chat.whatsapp_adapter import WhatsAppAdapter

BOT_JID = "351900000000@c.us"
GROUP = "12036300000000@g.us"
ALICE = "351911111111@c.us"


class PollReply:
    """Mimics ``engine.Reply`` for a poll turn: empty text, a poll draft."""

    def __init__(self, draft, telemetry=None):
        self.text = ""
        self.citation = ""
        self.poll = draft
        self.route = type("Route", (), {"mode": "factual", "command": "poll"})()
        self.telemetry = telemetry or {}


def _draft(**over):
    draft = polls.PollDraft(
        question="Onde é que vamos a jantar?",
        options=["O Botequim do Zé", "A Tasca da Rita"],
        dropped=[("O Sushi Bar", "rejeitado por todos")],
        messages_read=3,
    )
    for key, value in over.items():
        setattr(draft, key, value)
    return draft


def make_poll_adapter(tmp_path, reply, **overrides):
    """Adapter whose responder returns a poll Reply, with the poll store pointed
    at a temp file so a test never writes to the repo's data/."""
    config = {
        "whatsapp": {
            "bot_jid": BOT_JID,
            "contacts": {"351911111111@c.us": "Alice"},
            "send_seen": False,
            **overrides,
        },
        "chat": {"poll": {"store": str(tmp_path / "polls.jsonl")}},
    }
    client = MockWahaClient(echo=False)
    adapter = WhatsAppAdapter(
        responder=lambda message, speaker, recent_lines, **kw: reply,
        waha_client=client,
        config=config,
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "sessions")),
    )
    return adapter, client


def make_capturing_adapter(tmp_path, reply, responder):
    """Like make_poll_adapter but with a caller-supplied responder, so a test
    can check what the adapter passed it (e.g. the chat_id)."""
    config = {
        "whatsapp": {
            "bot_jid": BOT_JID,
            "contacts": {"351911111111@c.us": "Alice"},
            "send_seen": False,
        },
        "chat": {"poll": {"store": str(tmp_path / "polls.jsonl")}},
    }
    client = MockWahaClient(echo=False)
    adapter = WhatsAppAdapter(
        responder=responder,
        waha_client=client,
        config=config,
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "sessions")),
    )
    return adapter, client


# Real WhatsApp ids are unique per message, and the adapter ignores a repeated
# id as a replay — so a fixture must not reuse one across events.
_event_seq = itertools.count(1)


def group_event(text, sender=ALICE, name="Alice", mention=False, message_id=None):
    return {
        "event": "message",
        "payload": {
            "id": message_id or f"g{next(_event_seq)}",
            "from": GROUP,
            "participant": sender,
            "body": text,
            "notifyName": name,
            "mentionedIds": [BOT_JID] if mention else [],
        },
    }


def test_a_two_option_draft_becomes_a_native_poll(tmp_path):
    adapter, client = make_poll_adapter(tmp_path, PollReply(_draft()))

    result = adapter.handle_event(group_event("faz uma sondagem", mention=True))

    assert result is not None
    assert result["command"] == "poll"
    assert result["poll_id"]
    # The poll went out as a poll, not as text.
    poll_sends = [s for s in client.sent if "poll" in s]
    assert len(poll_sends) == 1
    assert poll_sends[0]["poll"]["name"] == "Onde é que vamos a jantar?"
    assert poll_sends[0]["poll"]["options"] == ["O Botequim do Zé", "A Tasca da Rita"]
    assert poll_sends[0]["poll"]["multipleAnswers"] is True
    # A dropped option is named in a follow-up text line. The [Sondagem: …]
    # summary is NOT sent to WhatsApp — it only goes into the session window,
    # so the next turn knows a poll exists.
    texts = [s["text"] for s in client.sent if "text" in s]
    assert any("O Sushi Bar" in t for t in texts)
    assert not any("[Sondagem:" in t for t in texts)
    assert any("[Sondagem: Onde é que vamos a jantar?:" in line
               for line in adapter.session_store.recent(GROUP))


def test_a_sent_poll_is_recorded_in_the_store(tmp_path):
    adapter, client = make_poll_adapter(tmp_path, PollReply(_draft()))

    result = adapter.handle_event(group_event("faz uma sondagem", mention=True))

    entries = polls.PollStore(tmp_path / "polls.jsonl").load()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["poll_id"] == result["poll_id"]
    assert entry["chat_id"] == GROUP
    assert entry["question"] == "Onde é que vamos a jantar?"
    assert entry["options"] == ["O Botequim do Zé", "A Tasca da Rita"]
    assert entry["dropped"] == [{"option": "O Sushi Bar", "reason": "rejeitado por todos"}]
    assert entry["multiple_answers"] is True
    assert entry["asked_by"] == "Alice"
    assert entry["messages_read"] == 3


def test_one_option_is_a_line_not_a_poll(tmp_path):
    adapter, client = make_poll_adapter(tmp_path, PollReply(_draft(options=["O Botequim do Zé"])))

    result = adapter.handle_event(group_event("faz uma sondagem", mention=True))

    assert result["command"] == "poll"
    assert not [s for s in client.sent if "poll" in s]
    line = client.sent[0]["text"]
    assert "O Botequim do Zé" in line
    # Nothing was sent, so nothing is recorded.
    assert polls.PollStore(tmp_path / "polls.jsonl").load() == []


def test_no_options_is_a_line_not_a_poll(tmp_path):
    adapter, client = make_poll_adapter(tmp_path, PollReply(_draft(options=[])))

    result = adapter.handle_event(group_event("faz uma sondagem", mention=True))

    assert result["command"] == "poll"
    assert not [s for s in client.sent if "poll" in s]
    assert "duas opções" in client.sent[0]["text"]


def test_a_disabled_draft_gets_the_disabled_line(tmp_path):
    adapter, client = make_poll_adapter(
        tmp_path, PollReply(_draft(status=polls.STATUS_DISABLED)))

    result = adapter.handle_event(group_event("faz uma sondagem", mention=True))

    assert result["command"] == "poll"
    assert not [s for s in client.sent if "poll" in s]
    assert client.sent[0]["text"] == adapter.command_replies["poll_disabled"]


def test_an_error_draft_gets_the_unreadable_line(tmp_path):
    adapter, client = make_poll_adapter(
        tmp_path, PollReply(_draft(status=polls.STATUS_ERROR)))

    result = adapter.handle_event(group_event("faz uma sondagem", mention=True))

    assert result["command"] == "poll"
    assert not [s for s in client.sent if "poll" in s]
    assert client.sent[0]["text"] == adapter.command_replies["poll_unreadable"]


def test_a_none_draft_gets_the_unreadable_line(tmp_path):
    adapter, client = make_poll_adapter(tmp_path, PollReply(None))

    result = adapter.handle_event(group_event("faz uma sondagem", mention=True))

    assert result["command"] == "poll"
    assert client.sent[0]["text"] == adapter.command_replies["poll_unreadable"]


def test_a_failed_send_says_what_was_found(tmp_path):
    adapter, client = make_poll_adapter(tmp_path, PollReply(_draft()))

    def boom(chat_id, name, options, multiple_answers=True):
        raise RuntimeError("WAHA down")

    client.send_poll = boom

    result = adapter.handle_event(group_event("faz uma sondagem", mention=True))

    assert result["command"] == "poll"
    line = client.sent[0]["text"]
    assert "Não consegui criar a sondagem" in line
    assert "O Botequim do Zé, A Tasca da Rita" in line
    # The poll never went out, so it is not recorded.
    assert polls.PollStore(tmp_path / "polls.jsonl").load() == []


def test_a_failed_poll_line_quotes_the_asker_in_a_group(tmp_path):
    """The fixed failure lines quote the asker's message; the poll itself and
    the left-out line do not — only the failure lines carry a reply_to."""
    adapter, client = make_poll_adapter(tmp_path, PollReply(_draft(options=[])))
    event = group_event("faz uma sondagem", mention=True)

    adapter.handle_event(event)

    texts = [s for s in client.sent if "text" in s]
    assert len(texts) == 1
    assert texts[0].get("reply_to") == event["payload"]["id"]


def test_a_sent_poll_does_not_quote(tmp_path):
    """On success the poll and the left-out line are both sent without a
    reply_to: the poll is its own message, and the left-out line follows it."""
    adapter, client = make_poll_adapter(tmp_path, PollReply(_draft()))
    event = group_event("faz uma sondagem", mention=True)

    adapter.handle_event(event)

    for send in client.sent:
        assert send.get("reply_to") is None


def test_the_poll_lines_carry_no_dash(tmp_path):
    """Everything the bot sends is read aloud or rendered on a phone, so the
    fixed poll lines follow the same no-dash rule as the rest of command_replies."""
    adapter, _ = make_poll_adapter(tmp_path, PollReply(_draft()))
    for key in ("poll_not_enough", "poll_one_option", "poll_unreadable",
                "poll_disabled", "poll_send_failed", "poll_left_out"):
        line = adapter.command_replies[key]
        assert "—" not in line and "–" not in line and " - " not in line


def test_the_poll_enters_the_verbatim_window(tmp_path):
    """The [Sondagem: …] line is appended to the session store, so the next turn
    (and the rolling summary) knows a poll exists in this chat."""
    adapter, _ = make_poll_adapter(tmp_path, PollReply(_draft()))

    adapter.handle_event(group_event("faz uma sondagem", mention=True))

    lines = adapter.session_store.recent(GROUP)
    assert any(line.startswith("Kaya Bot: [Sondagem:") for line in lines)
    assert any("O Botequim do Zé" in line and "A Tasca da Rita" in line
               for line in lines)


def test_the_responder_is_given_the_chat_id(tmp_path):
    """The engine needs the chat_id to read the right log, so the adapter passes
    it through when the responder accepts it."""
    seen = {}

    def responder(message, speaker, recent_lines, **kw):
        seen.update(kw)
        return PollReply(_draft())

    adapter, _ = make_capturing_adapter(tmp_path, None, responder)

    adapter.handle_event(group_event("faz uma sondagem", mention=True))

    assert seen.get("chat_id") == GROUP


def test_a_responder_without_chat_id_still_works(tmp_path):
    """A responder that does not accept chat_id must not break: the adapter
    detects the missing kwarg and simply does not pass it."""
    def responder(message, speaker, recent_lines, scope=None, exclude_from=None):
        return PollReply(_draft())

    adapter, client = make_capturing_adapter(tmp_path, None, responder)

    result = adapter.handle_event(group_event("faz uma sondagem", mention=True))

    assert result is not None
    assert result["command"] == "poll"
    assert [s for s in client.sent if "poll" in s]


def test_a_poll_turn_is_interaction_logged(tmp_path):
    """A poll is a real turn, not bookkeeping: it goes into the interaction log
    so the telemetry (options, dropped, invented) is measurable."""
    adapter, _ = make_poll_adapter(tmp_path, PollReply(_draft()))

    result = adapter.handle_event(group_event("faz uma sondagem", mention=True))

    assert metrics.should_log(result) is True


def test_waha_client_sendpoll_body():
    """The real WahaClient posts to /api/sendPoll with the documented body.
    There is no reply_to on this endpoint: the poll is its own message."""
    client = WahaClient.__new__(WahaClient)
    client.session = "default"
    posts = []

    class _FakeHttp:
        def post(self, path, json=None):
            posts.append((path, json))

            class _Resp:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"id": "waha-poll-1"}

            return _Resp()

    client._client = _FakeHttp()

    result = client.send_poll(GROUP, "Onde jantar?", ["A", "B"], multiple_answers=True)

    assert len(posts) == 1
    path, body = posts[0]
    assert path == "/api/sendPoll"
    assert body == {
        "session": "default",
        "chatId": GROUP,
        "poll": {"name": "Onde jantar?", "options": ["A", "B"],
                 "multipleAnswers": True},
    }
    assert "reply_to" not in body
    assert result == {"id": "waha-poll-1"}
