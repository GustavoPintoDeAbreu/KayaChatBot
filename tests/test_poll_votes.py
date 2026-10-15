"""Votes on Kaya's polls: parsing, the registry, the electorate, and the slots."""
import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import poll_votes

LISBON = ZoneInfo("Europe/Lisbon")
CHAT = "120363@ g.us".replace(" ", "")
BOT = "351900000000@c.us"


def vote_event(poll_id="true_" + CHAT + "_key1", voter="351911111111@c.us",
               options=("sim",), poll_from_me=True, vote_from_me=False,
               timestamp=1_700_000_000):
    return {
        "event": "poll.vote",
        "payload": {
            "poll": {"id": poll_id, "fromMe": poll_from_me, "from": CHAT},
            "vote": {"participant": voter, "to": CHAT, "selectedOptions": list(options),
                     "fromMe": vote_from_me, "timestamp": timestamp * 1000},
        },
    }


def test_parse_waha_poll_vote_reads_the_voter_and_options():
    vote = poll_votes.parse_waha_poll_vote(vote_event())
    assert vote is not None
    assert vote.poll_id == "key1"
    assert vote.chat_id == CHAT
    assert vote.voter_id == "351911111111@c.us"
    assert vote.options == ["sim"]
    assert vote.poll_from_me is True
    assert vote.vote_from_me is False
    assert vote.timestamp == 1_700_000_000


def test_parse_waha_poll_vote_rejects_a_vote_from_the_bot():
    vote = poll_votes.parse_waha_poll_vote(vote_event(vote_from_me=True))
    assert vote is not None
    assert vote.vote_from_me is True


def test_parse_waha_poll_vote_rejects_a_group_as_voter():
    vote = poll_votes.parse_waha_poll_vote(vote_event(voter=CHAT))
    assert vote is None


def test_parse_waha_poll_vote_ignores_the_failed_twin():
    event = vote_event()
    event["event"] = "poll.vote.failed"
    assert poll_votes.parse_waha_poll_vote(event) is None


def test_parse_waha_poll_vote_dm_uses_the_voter_as_chat():
    # a poll in a DM has no group JID anywhere: the chat is the voter
    event = vote_event(poll_id="true_351911111111@c.us_key1")
    event["payload"]["poll"]["from"] = "351900000000@c.us"
    event["payload"]["vote"]["to"] = "351900000000@c.us"
    vote = poll_votes.parse_waha_poll_vote(event)
    assert vote is not None
    assert vote.chat_id == "351911111111@c.us"
    assert vote.voter_id == "351911111111@c.us"


def test_parse_waha_poll_vote_ignores_other_events_and_missing_poll():
    # a plain message is not a vote
    message = {"event": "message", "payload": {"vote": {"participant": "351911111111@c.us"},
                                                "poll": {"id": "true_" + CHAT + "_key1"}}}
    assert poll_votes.parse_waha_poll_vote(message) is None
    # a poll.vote with no poll object is not a vote either
    no_poll = vote_event()
    del no_poll["payload"]["poll"]
    assert poll_votes.parse_waha_poll_vote(no_poll) is None


def test_poll_key_normalises_serialised_and_bare_ids():
    # the four shapes a poll id can arrive in, all the same poll
    assert poll_votes.poll_key("true_" + CHAT + "_key1") == "key1"
    assert poll_votes.poll_key("false_" + CHAT + "_key1") == "key1"
    assert poll_votes.poll_key("true_" + CHAT + "_key1_351911111111") == "key1"
    assert poll_votes.poll_key("key1") == "key1"
    assert len({poll_votes.poll_key(s) for s in (
        "true_" + CHAT + "_key1", "false_" + CHAT + "_key1",
        "true_" + CHAT + "_key1_351911111111", "key1")}) == 1


def test_registry_records_votes_and_ignores_the_bot(tmp_path):
    registry = poll_votes.PollRegistry(tmp_path / "votes.json")
    registry.register(CHAT, "key1", ["sim", "não"], 1_700_000_000, source="explicit")
    alice = poll_votes.parse_waha_poll_vote(vote_event())
    assert registry.record_vote(alice, "Alice", 1_700_000_100) is True
    # the bot's own tap is not a vote to record
    bot_vote = poll_votes.parse_waha_poll_vote(vote_event(vote_from_me=True))
    assert registry.record_vote(bot_vote, "Kaya Bot", 1_700_000_200) is False
    record = registry.get("key1")
    assert set(record["votes"]) == {"351911111111"}
    assert record["votes"]["351911111111"]["name"] == "Alice"


def test_registry_keeps_the_newer_vote_and_a_retraction(tmp_path):
    registry = poll_votes.PollRegistry(tmp_path / "votes.json")
    registry.register(CHAT, "key1", ["sim", "não"], 1_700_000_000, source="explicit")
    alice = poll_votes.parse_waha_poll_vote(
        vote_event(options=("sim",), timestamp=1_700_000_100))
    assert registry.record_vote(alice, "Alice", 1_700_000_100) is True
    # a stale duplicate (older timestamp) does not overwrite the newer vote
    stale = poll_votes.parse_waha_poll_vote(
        vote_event(options=("não",), timestamp=1_700_000_050))
    assert registry.record_vote(stale, "Alice", 1_700_000_050) is False
    assert registry.get("key1")["votes"]["351911111111"]["options"] == ["sim"]
    # a retraction (no selected options) clears the vote
    retract = poll_votes.parse_waha_poll_vote(
        vote_event(options=(), timestamp=1_700_000_200))
    assert registry.record_vote(retract, "Alice", 1_700_000_200) is True
    assert registry.get("key1")["votes"]["351911111111"]["options"] == []


def test_registry_auto_registers_a_vote_on_an_unseen_own_poll(tmp_path):
    registry = poll_votes.PollRegistry(tmp_path / "votes.json")
    vote = poll_votes.parse_waha_poll_vote(vote_event(poll_from_me=True))
    assert registry.record_vote(vote, "Alice", 1_700_000_100) is True
    assert registry.get("key1") is not None
    # a vote on somebody else's poll (a different key, never registered) is ignored
    other = poll_votes.parse_waha_poll_vote(
        vote_event(poll_id="true_" + CHAT + "_someoneelse", poll_from_me=False))
    assert registry.record_vote(other, "Bob", 1_700_000_200) is False
    assert registry.get("someoneelse") is None


def test_explicit_register_after_auto_keeps_votes(tmp_path):
    # a poll learned from its first vote, then registered explicitly (the
    # conversation-poll hook): the votes survive, and the explicit registration
    # wins the source and created_at.
    registry = poll_votes.PollRegistry(tmp_path / "votes.json")
    vote = poll_votes.parse_waha_poll_vote(vote_event(poll_from_me=True,
                                                      timestamp=1_700_000_100))
    assert registry.record_vote(vote, "Alice", 1_700_000_100) is True
    record = registry.get("key1")
    assert record["source"] == "vote"
    assert record["created_at"] == 1_700_000_100
    registry.register(CHAT, "key1", ["sim", "não"], 1_700_000_000, source="explicit")
    record = registry.get("key1")
    assert record["source"] == "explicit"
    assert record["created_at"] == 1_700_000_000
    assert record["votes"]["351911111111"]["name"] == "Alice"
    assert record["votes"]["351911111111"]["options"] == ["sim"]


def test_registry_open_polls_close_the_expired(tmp_path):
    registry = poll_votes.PollRegistry(tmp_path / "votes.json")
    registry.register(CHAT, "old", ["a"], 1_700_000_000 - 30 * 3600)
    registry.register(CHAT, "fresh", ["a"], 1_700_000_000)
    open_polls = registry.open_polls(1_700_000_000, stop_after_hours=24)
    assert [poll["id"] for poll in open_polls] == ["fresh"]
    assert registry.get("old")["closed"] == "expired"


def test_electorate_groups_members_and_drops_the_bot():
    records = [
        {"chat_id": CHAT, "sender": "Alice", "sender_id": "351911111111@c.us",
         "sender_phone": "351911111111", "timestamp": 1_700_000_000},
        {"chat_id": CHAT, "sender": "Alice", "sender_id": "alice@lid",
         "sender_phone": "351911111111", "timestamp": 1_700_000_010},
        # the bot's own line, under a display name that is NOT "Kaya Bot": it is
        # dropped by the bot_ids filter (local parts), not by the name check.
        {"chat_id": CHAT, "sender": "Kaya", "sender_id": BOT,
         "sender_phone": "351900000000", "timestamp": 1_700_000_020},
        {"chat_id": "other@ g.us".replace(" ", ""), "sender": "Bob",
         "sender_id": "351922222222@c.us", "sender_phone": "351922222222",
         "timestamp": 1_700_000_030},
    ]
    voters = poll_votes.electorate(records, CHAT, 0, {"351900000000"},
                                   is_member=lambda name: name == "Alice")
    assert len(voters) == 1
    assert voters[0].name == "Alice"
    assert voters[0].mention_jid == "351911111111@c.us"
    assert {"351911111111", "alice"} <= voters[0].ids


def test_electorate_a_lid_sender_without_a_phone():
    # most of this group's contacts are keyed by @lid with no phone: the
    # mention must be the lid JID itself, not a <lid>@c.us that tags nobody, and
    # the lid stays in ids (as an id) so a vote keyed by it still matches.
    records = [
        {"chat_id": CHAT, "sender": "Bernardo", "sender_id": "bernardo@lid",
         "sender_phone": "", "timestamp": 1_700_000_000},
    ]
    voters = poll_votes.electorate(records, CHAT, 0, {"351900000000"},
                                   is_member=lambda name: name == "Bernardo")
    assert len(voters) == 1
    voter = voters[0]
    assert voter.name == "Bernardo"
    # the lid JID, not a fake phone: a <lid>@c.us does not exist
    assert voter.mention_jid == "bernardo@lid"
    assert "@c.us" not in voter.mention_jid
    # the lid local part is in ids (it is how a vote is keyed), but there is no
    # phone in there
    assert "bernardo" in voter.ids
    assert "351900000000" not in voter.ids
    # and a vote recorded under the lid still counts as having voted
    poll = {"votes": {"bernardo": {"name": "Bernardo", "options": ["sim"], "at": 1}}}
    assert poll_votes.has_voted(poll, voter) is True


def test_electorate_upgrades_the_mention_when_a_phone_turns_up():
    # a member whose first record is only an @lid id gets a lid mention; a later
    # record that carries a phone upgrades it to the phone JID, which is the form
    # WAHA is known to render.
    records = [
        {"chat_id": CHAT, "sender": "Bernardo", "sender_id": "bernardo@lid",
         "sender_phone": "", "timestamp": 1_700_000_000},
        {"chat_id": CHAT, "sender": "Bernardo", "sender_id": "351911111111@c.us",
         "sender_phone": "351911111111", "timestamp": 1_700_000_010},
    ]
    voters = poll_votes.electorate(records, CHAT, 0, {"351900000000"},
                                   is_member=lambda name: name == "Bernardo")
    assert len(voters) == 1
    voter = voters[0]
    # the phone, once known, is the mention target
    assert voter.mention_jid == "351911111111@c.us"
    # both ids stay, so a vote keyed by either the lid or the phone matches
    assert {"bernardo", "351911111111"} <= voter.ids


def test_electorate_drops_records_older_than_since():
    # the electorate is bounded below by since_ts (created_at - electorate_days):
    # a message written before the window is not a voter, even in the same chat.
    since = 1_700_000_000
    records = [
        {"chat_id": CHAT, "sender": "Alice", "sender_id": "351911111111@c.us",
         "sender_phone": "351911111111", "timestamp": since - 3600},  # too old
        {"chat_id": CHAT, "sender": "Alice", "sender_id": "351911111111@c.us",
         "sender_phone": "351911111111", "timestamp": since + 3600},  # in window
        {"chat_id": CHAT, "sender": "Bob", "sender_id": "351922222222@c.us",
         "sender_phone": "351922222222", "timestamp": since - 7200},  # too old, and only one
    ]
    voters = poll_votes.electorate(records, CHAT, since, {BOT},
                                   is_member=lambda name: name in ("Alice", "Bob"))
    # only Alice has a record inside the window; Bob's only line is too old
    assert [voter.name for voter in voters] == ["Alice"]


def test_missing_voters_and_has_voted():
    poll = {"votes": {"351911111111": {"name": "Alice", "options": ["sim"],
                                       "at": 1}}}
    alice = poll_votes.Voter(name="Alice", ids={"351911111111"})
    bob = poll_votes.Voter(name="Bob", ids={"351922222222"})
    assert poll_votes.has_voted(poll, alice) is True
    assert poll_votes.has_voted(poll, bob) is False
    assert poll_votes.missing_voters(poll, [alice, bob]) == [bob]


def test_a_retracted_vote_leaves_the_person_in_missing_voters():
    # a retraction (empty options) is kept as "not voted", so the person is
    # back in Falta votar and gets reminded again — not counted as having voted.
    poll = {"votes": {"351911111111": {"name": "Alice", "options": [], "at": 2}}}
    alice = poll_votes.Voter(name="Alice", ids={"351911111111"})
    assert poll_votes.has_voted(poll, alice) is False
    assert poll_votes.missing_voters(poll, [alice]) == [alice]
    # and a retraction does not count toward "N pessoas votaram"
    text = poll_votes.status_text(
        {"options": ["sim"], "created_at": 1_700_000_000,
         "votes": {"351911111111": {"name": "Alice", "options": [], "at": 2}}},
        [alice])
    assert "0 pessoas votaram" in text


def test_tally_orders_by_votes_and_keeps_zero_options():
    poll = {"options": ["sim", "não", "talvez"],
            "votes": {"a": {"name": "Alice", "options": ["sim"], "at": 1},
                      "b": {"name": "Bob", "options": ["sim"], "at": 2},
                      "c": {"name": "Carol", "options": ["não"], "at": 3}}}
    result = poll_votes.tally(poll)
    assert result[0][0] == "sim" and len(result[0][1]) == 2
    assert result[1][0] == "não"
    assert result[2] == ("talvez", [])


def test_in_window_handles_a_wrapping_window():
    now = datetime.datetime(2026, 10, 14, 23, 0, tzinfo=LISBON)
    assert poll_votes.in_window(now, ["22:00", "06:00"]) is True
    midday = datetime.datetime(2026, 10, 14, 12, 0, tzinfo=LISBON)
    assert poll_votes.in_window(midday, ["22:00", "06:00"]) is False
    assert poll_votes.in_window(midday, ["10:00", "22:00"]) is True


def _remind_cfg(**overrides):
    cfg = {"enabled": True, "dry_run": True, "interval_hours": 6,
           "stop_after_hours": 24, "min_gap_hours": 3,
           "send_between": ["00:00", "23:59"]}
    cfg.update(overrides)
    return cfg


def test_due_slot_fires_once_per_interval_and_stops_after():
    created = 1_700_000_000
    poll = {"created_at": created, "reminded_slots": [], "last_reminder_at": None}
    # 3 hours in: not yet a full interval
    early = datetime.datetime.fromtimestamp(created + 3 * 3600, LISBON)
    assert poll_votes.due_slot(poll, early, _remind_cfg()) is None
    # 7 hours in: the first slot (created + 6h) is due
    due = datetime.datetime.fromtimestamp(created + 7 * 3600, LISBON)
    slot = poll_votes.due_slot(poll, due, _remind_cfg())
    assert slot == created + 6 * 3600
    # already reminded in that slot: nothing
    poll["reminded_slots"] = [slot]
    assert poll_votes.due_slot(poll, due, _remind_cfg()) is None
    # 30 hours in: past the stop
    late = datetime.datetime.fromtimestamp(created + 30 * 3600, LISBON)
    assert poll_votes.due_slot(poll, late, _remind_cfg()) is None


def test_due_slot_quiet_hours_sends_latest_once():
    # two slots fall in quiet hours (02:00 and 08:00); at 10:00 the latest one
    # (08:00) is sent, and at 10:05 nothing more (min gap + already reminded).
    created = 1_700_000_000
    poll = {"created_at": created, "reminded_slots": [], "last_reminder_at": None}
    cfg = _remind_cfg(send_between=["10:00", "22:00"])
    # a moment 7h after creation that is 03:00 local: both the 6h and 12h slots
    # are in the past, but 03:00 is outside the window -> nothing
    target = datetime.datetime.fromtimestamp(created + 7 * 3600, LISBON)
    poll["created_at"] = int(target.replace(hour=3, minute=0, second=0).timestamp()) \
        - 7 * 3600
    at_3am = datetime.datetime.fromtimestamp(poll["created_at"] + 7 * 3600, LISBON)
    assert at_3am.hour == 3
    assert poll_votes.due_slot(poll, at_3am, cfg) is None
    # 10:00 local the same day: inside the window, the latest slot (12h) is due
    at_10 = datetime.datetime.fromtimestamp(poll["created_at"] + 7 * 3600, LISBON)
    at_10 = at_10.replace(hour=10, minute=0, second=0)
    slot = poll_votes.due_slot(poll, at_10, cfg)
    assert slot == poll["created_at"] + 12 * 3600
    # mark it reminded: at 10:05 the min gap holds it back
    poll["reminded_slots"] = [slot]
    poll["last_reminder_at"] = int(at_10.timestamp())
    at_1005 = at_10.replace(minute=5)
    assert poll_votes.due_slot(poll, at_1005, cfg) is None


def test_due_slot_honours_the_min_gap_between_reminders():
    created = 1_700_000_000
    poll = {"created_at": created, "reminded_slots": [], "last_reminder_at": None}
    # 13h in: the second slot (created + 12h) is due
    at_13h = datetime.datetime.fromtimestamp(created + 13 * 3600, LISBON)
    assert poll_votes.due_slot(poll, at_13h, _remind_cfg()) == created + 12 * 3600
    # a reminder went out 1h ago: the min gap (3h) holds the next one back
    poll["last_reminder_at"] = created + 12 * 3600
    at_13h_again = datetime.datetime.fromtimestamp(created + 13 * 3600, LISBON)
    assert poll_votes.due_slot(poll, at_13h_again, _remind_cfg()) is None
    # 4h after that reminder the gap has passed: the slot is due again
    at_16h = datetime.datetime.fromtimestamp(created + 16 * 3600, LISBON)
    assert poll_votes.due_slot(poll, at_16h, _remind_cfg()) == created + 12 * 3600


def test_due_slot_returns_only_the_latest_slot():
    created = 1_700_000_000
    poll = {"created_at": created, "reminded_slots": [], "last_reminder_at": None}
    # 13h in: the 6h and 12h slots have both passed; only the newest (12h) is due
    at_13h = datetime.datetime.fromtimestamp(created + 13 * 3600, LISBON)
    assert poll_votes.due_slot(poll, at_13h, _remind_cfg()) == created + 12 * 3600
    # mark it reminded: the next pass finds nothing (the older 6h slot is not replayed)
    poll["reminded_slots"] = [created + 12 * 3600]
    poll["last_reminder_at"] = created + 13 * 3600
    assert poll_votes.due_slot(poll, at_13h, _remind_cfg()) is None


def test_reminder_text_mentions_the_missing_voters():
    poll = {"options": ["sim", "não"],
            "votes": {"351911111111": {"name": "Alice", "options": ["sim"], "at": 1}}}
    bob = poll_votes.Voter(name="Bob", ids={"351922222222"},
                           mention_jid="351922222222@c.us")
    text_with_tokens, text_with_names, jids = poll_votes.reminder_text(poll, [bob])
    assert "@351922222222" in text_with_tokens
    assert "Bob" in text_with_names
    assert jids == ["351922222222@c.us"]
    assert "Vai à frente: sim (1)." in text_with_tokens


def test_status_text_counts_without_inventing():
    poll = {"options": ["sim", "não"], "created_at": 1_700_000_000,
            "votes": {"351911111111": {"name": "Alice", "options": ["sim"], "at": 1}}}
    bob = poll_votes.Voter(name="Bob", ids={"351922222222"})
    text = poll_votes.status_text(poll, [bob])
    assert "1 pessoas votaram" in text
    assert "sim: 1 (Alice)" in text
    assert "não: 0" in text
    assert "Falta votar: Bob." in text


def test_status_text_has_no_dashes():
    # the group's rule: no —, –, or " - " anywhere in a canned string.
    poll = {"options": ["sim", "não"], "created_at": 1_700_000_000,
            "votes": {"351911111111": {"name": "Alice", "options": ["sim"], "at": 1}}}
    bob = poll_votes.Voter(name="Bob", ids={"351922222222"})
    text = poll_votes.status_text(poll, [bob])
    assert "—" not in text
    assert "–" not in text
    assert " - " not in text
    tokens, names, _ = poll_votes.reminder_text(poll, [bob])
    for line in (tokens, names):
        assert "—" not in line and "–" not in line and " - " not in line
