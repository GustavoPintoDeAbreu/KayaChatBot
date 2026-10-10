"""The Pi's birthday line at midnight, and the noon nudge only when nobody said parabéns."""
import datetime
import random
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.app import Gateway, GatewaySettings
from src.gateway.birthdays import BirthdayGreeter, tag_for
from src.gateway.journal import Journal
from src.gateway.monitor import PcMonitor, PcState

LISBON = ZoneInfo("Europe/Lisbon")
GENERAL = "120363000000000001@g.us"
TRIPS = "120363000000000002@g.us"
BOT = "351900000000@c.us"
ALICE = "351911111111@c.us"
CONFIG = {"chat": {"birthdays": {
    "enabled": True,
    "windows": {"midnight": ["00:00", "11:00"], "noon": ["12:00", "18:00"]},
    "messages": {"midnight": ["Parabéns {tag}!"], "noon": ["Ninguém deu os parabéns ao {tag}?"]},
}}}
ROSTER = {
    "Fred": {"date": "06-29", "aliases": ["fred", "frederico"], "jid": "111@lid"},
    "Gil": {"date": "02-06", "aliases": ["gil"], "jid": ""},
}


def _at(year, month, day, hour, minute=0):
    return datetime.datetime(year, month, day, hour, minute, tzinfo=LISBON).timestamp()


class Clock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def _message(message_id, body, *, chat=GENERAL, timestamp, mentions=(), from_me=False):
    return {"event": "message", "me": {"id": BOT}, "payload": {
        "id": message_id, "from": chat, "participant": ALICE, "body": body,
        "timestamp": int(timestamp), "notifyName": "Alice", "fromMe": from_me,
        "mentionedIds": list(mentions)}}


@pytest.fixture
def greeter(tmp_path):
    journal = Journal(str(tmp_path / "journal.sqlite3"), str(tmp_path / "media"))
    clock = Clock(_at(2026, 6, 29, 0, 1))
    sent, roster = [], {"value": dict(ROSTER)}

    def send(chat, text, mentions):
        sent.append((chat, text, mentions))

    def fetch():
        if roster["value"] is None:
            raise OSError("PC is off")
        return roster["value"]

    instance = BirthdayGreeter(chat_id=GENERAL, data_dir=str(tmp_path), config=CONFIG,
                               journal=journal, send_text=send, fetch_roster=fetch,
                               timezone=LISBON, now=clock, rng=random.Random(0))
    instance.refresh_roster(force=True)
    return instance, clock, sent, journal, roster


def _journal(journal, event, received_at):
    payload = event["payload"]
    journal.append(event, dedup_key=f"message:{payload['id']}", event_type="message",
                   chat_id=payload["from"], sender_id=payload["participant"],
                   message_id=payload["id"], wa_ts=payload["timestamp"],
                   received_at=received_at, addressed=False, backlog=False)


def test_the_midnight_line_tags_the_member_once(greeter):
    instance, clock, sent, _, _ = greeter
    assert instance.tick() == ["Fred:midnight sent"]
    assert sent == [(GENERAL, "Parabéns @111!", ["111@lid"])]
    clock.value = _at(2026, 6, 29, 0, 2)
    assert instance.tick() == []
    assert len(sent) == 1


def test_a_member_with_no_id_is_named_not_tagged():
    assert tag_for("Gil", {"jid": ""}) == ("Gil", [])
    assert tag_for("Fred", {"jid": "111@lid"}) == ("@111", ["111@lid"])


def test_midnight_catches_up_until_eleven_then_gives_up(greeter):
    instance, clock, sent, _, _ = greeter
    clock.value = _at(2026, 6, 29, 10, 59)
    assert instance.tick() == ["Fred:midnight sent"]
    instance.state.path.unlink()
    clock.value = _at(2026, 6, 29, 11, 0)
    assert instance.tick() == []


def test_noon_is_skipped_when_somebody_congratulated(greeter):
    instance, clock, sent, journal, _ = greeter
    instance.tick()
    event = _message("m1", "Parabéns fred!! 🎂", timestamp=_at(2026, 6, 29, 9))
    _journal(journal, event, _at(2026, 6, 29, 9))
    clock.value = _at(2026, 6, 29, 12, 0)
    assert instance.tick() == ["Fred:noon skipped"]
    assert len(sent) == 1


def test_noon_nudges_when_nobody_did(greeter):
    instance, clock, sent, journal, _ = greeter
    instance.tick()
    _journal(journal, _message("m1", "bom dia malta", timestamp=_at(2026, 6, 29, 9)),
             _at(2026, 6, 29, 9))
    _journal(journal, _message("m0", "parabéns fred", timestamp=_at(2026, 6, 28, 23)),
             _at(2026, 6, 28, 23))
    clock.value = _at(2026, 6, 29, 12, 30)
    assert instance.tick() == ["Fred:noon sent"]
    assert sent[-1] == (GENERAL, "Ninguém deu os parabéns ao @111?", ["111@lid"])


def test_the_bots_own_parabens_does_not_count(greeter):
    instance, clock, sent, journal, _ = greeter
    _journal(journal, _message("m1", "Parabéns @111!", timestamp=_at(2026, 6, 29, 0, 1),
                               from_me=True), _at(2026, 6, 29, 0, 1))
    clock.value = _at(2026, 6, 29, 12, 0)
    assert "Fred:noon sent" in instance.tick()


def test_a_parabens_in_another_sub_group_counts(greeter):
    instance, clock, _, journal, _ = greeter
    _journal(journal, _message("m1", "parabéns!!", chat=TRIPS, timestamp=_at(2026, 6, 29, 8)),
             _at(2026, 6, 29, 8))
    clock.value = _at(2026, 6, 29, 12, 0)
    assert "Fred:noon skipped" in instance.tick()


def test_two_birthdays_need_the_parabens_to_name_the_member(greeter):
    instance, clock, sent, journal, roster = greeter
    roster["value"]["Gil"] = {"date": "06-29", "aliases": ["gil"], "jid": "222@lid"}
    instance.refresh_roster(force=True)
    _journal(journal, _message("m1", "parabéns gil 🎉", timestamp=_at(2026, 6, 29, 9)),
             _at(2026, 6, 29, 9))
    clock.value = _at(2026, 6, 29, 12, 0)
    assert sorted(instance.tick()) == ["Fred:noon sent", "Gil:noon skipped"]


def test_a_tag_counts_as_naming(greeter):
    instance, clock, _, journal, roster = greeter
    roster["value"]["Gil"] = {"date": "06-29", "aliases": ["gil"], "jid": "222@lid"}
    instance.refresh_roster(force=True)
    _journal(journal, _message("m1", "parabéns @111", mentions=["111@lid"],
                               timestamp=_at(2026, 6, 29, 9)), _at(2026, 6, 29, 9))
    clock.value = _at(2026, 6, 29, 12, 0)
    assert "Fred:noon skipped" in instance.tick()


def test_leap_day_birthday_is_greeted_on_the_28th(greeter):
    instance, clock, sent, _, roster = greeter
    roster["value"] = {"Leap": {"date": "02-29", "aliases": [], "jid": ""}}
    instance.refresh_roster(force=True)
    clock.value = _at(2027, 2, 28, 0, 5)
    assert instance.tick() == ["Leap:midnight sent"]


def test_a_failed_send_is_retried(greeter):
    instance, clock, sent, _, _ = greeter
    calls = {"n": 0}

    def flaky(chat, text, mentions):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("WAHA down")
        sent.append((chat, text, mentions))

    instance._send_text = flaky
    assert instance.tick() == []
    clock.value = _at(2026, 6, 29, 0, 3)
    assert instance.tick() == ["Fred:midnight sent"]


def test_the_roster_survives_the_pc_being_off(greeter, tmp_path):
    instance, clock, sent, _, roster = greeter
    roster["value"] = None
    assert instance.refresh_roster(force=True) is False
    assert instance.tick() == ["Fred:midnight sent"]


def test_state_survives_a_restart(greeter, tmp_path):
    instance, clock, sent, journal, _ = greeter
    instance.tick()
    again = BirthdayGreeter(chat_id=GENERAL, data_dir=str(tmp_path), config=CONFIG,
                            journal=journal, send_text=lambda *a: sent.append(a),
                            fetch_roster=lambda: None, timezone=LISBON, now=clock)
    assert again.tick() == []


def test_the_gateway_runs_it_only_with_a_target(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({**CONFIG, "power": {"timezone": "Europe/Lisbon"}}))
    env = {"GATEWAY_DATA_DIR": str(tmp_path / "data"), "GATEWAY_CONFIG": str(config),
           "GATEWAY_WHITELIST": str(tmp_path / "none.json"), "KAYA_RELAY_TOKEN": "relay"}
    monitor = PcMonitor("http://pc", probe=lambda: PcState.ONLINE)
    without = Gateway(GatewaySettings.from_env(env), monitor=monitor,
                      send_text=lambda *a, **k: None)
    assert without.birthdays is None
    sent = []
    gateway = Gateway(GatewaySettings.from_env({**env, "GATEWAY_BIRTHDAY_CHAT": GENERAL}),
                      monitor=monitor, send_text=lambda *a, **k: None,
                      send_mentions=lambda *a: sent.append(a),
                      fetch_birthdays=lambda: ROSTER)
    assert gateway.birthdays is not None
    gateway.birthdays.refresh_roster(force=True)
    status = TestClient(gateway.internal_app).get("/status").json()
    assert status["birthdays"]["members"] == 2
    public = TestClient(gateway.public_app).get("/status").json()
    assert "birthdays" not in public
