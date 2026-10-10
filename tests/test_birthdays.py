"""Birthdays: reading a date, trusting the right source, greeting once, mining bursts."""
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import birthdays
from src.chat.memory import KeyedSessionMemory
from src.chat.response_utils import build_member_prompt_suffix
from src.chat.waha_client import MockWahaClient
from src.chat.whatsapp_adapter import WhatsAppAdapter
from src.data.message_log import MessageLog


@pytest.mark.parametrize("text,expected", [
    ("8/9", "09-08"),
    ("08-09", "09-08"),
    ("8.9.1994", "09-08"),
    ("8 de setembro", "09-08"),
    ("faço anos a 29 de fevereiro", "02-29"),
    ("1 março", "03-01"),
    ("September 8th", "09-08"),
    ("31/2", None),
    ("amanhã", None),
])
def test_parse_date(text, expected):
    assert birthdays.parse_date(text) == expected


def test_spoken_date():
    assert birthdays.spoken_date("09-08") == "8 de setembro"


def test_leap_day_is_kept_on_the_28th():
    assert birthdays.falls_on("02-29", date(2027, 2, 28))
    assert not birthdays.falls_on("02-29", date(2028, 2, 28))
    assert birthdays.falls_on("02-29", date(2028, 2, 29))


def test_self_beats_profile_beats_mined():
    store = {"Pedro": {"mined": {"date": "07-14"}, "self": {"date": "07-15"}},
             "Gil": {"mined": {"date": "01-01"}}}
    members = {"members": [{"name": "Gil", "birthday": "2 de janeiro"},
                           {"name": "Rafa", "birthday": ""}]}
    resolved = birthdays.resolved_dates(store, members)
    assert resolved["Pedro"] == ("07-15", "self")
    assert resolved["Gil"] == ("01-02", "profile")
    assert "Rafa" not in resolved


def test_store_round_trip(tmp_path):
    store = birthdays.BirthdayStore(tmp_path / "b.json")
    store.set("Pedro", "07-15", "self")
    assert store.load()["Pedro"]["self"]["date"] == "07-15"
    with pytest.raises(ValueError):
        store.set("Pedro", "07-15", "guess")


def test_due_respects_window_and_sends_once_per_slot(tmp_path):
    dates = {"Fred": ("06-29", "self"), "Gil": ("01-02", "self")}
    state = birthdays.GreetingState(tmp_path / "state.json")
    noon = ("12:00", "18:00")
    assert birthdays.due(dates, state, datetime(2026, 6, 29, 11, 59), "noon", *noon) == []
    assert birthdays.due(dates, state, datetime(2026, 6, 29, 12, 0), "noon", *noon) == ["Fred"]
    state.mark(2026, "Fred:noon", "group")
    assert birthdays.due(dates, state, datetime(2026, 6, 29, 15, 0), "noon", *noon) == []
    assert birthdays.due(dates, state, datetime(2026, 6, 29, 0, 5), "midnight",
                         "00:00", "11:00") == ["Fred"]
    assert birthdays.due(dates, state, datetime(2027, 6, 29, 17, 59), "noon", *noon) == ["Fred"]
    assert birthdays.due(dates, state, datetime(2027, 6, 29, 18, 0), "noon", *noon) == []


def test_settings_carry_both_slots_with_defaults():
    cfg = birthdays.settings({"chat": {"birthdays": {"enabled": True}}})
    assert set(cfg["messages"]) == {"midnight", "noon"}
    assert all("{tag}" in line for lines in cfg["messages"].values() for line in lines)
    assert cfg["windows"]["midnight"] == ("00:00", "11:00")


def test_the_shipped_messages_all_tag_the_member():
    from src.config_loader import load_config

    cfg = birthdays.settings(load_config("config.yaml"))
    for slot in birthdays.SLOTS:
        assert cfg["messages"][slot]
        for line in cfg["messages"][slot]:
            assert line.format(tag="@X").count("@X") == 1


def test_roster_prefers_the_lid_and_falls_back_to_the_phone(tmp_path, monkeypatch):
    store = tmp_path / "b.json"
    birthdays.BirthdayStore(store).set("Pedro", "07-15", "self")
    birthdays.BirthdayStore(store).set("Gil", "02-06", "self")
    birthdays.BirthdayStore(store).set("Rafa", "04-24", "self")
    members = tmp_path / "members.json"
    members.write_text('{"members": [{"name": "Pedro", "aliases": ["pedrito"]}]}')
    config = {"chat": {"birthdays": {"enabled": True, "store": str(store)}},
              "data": {"group_members_file": str(members)}}
    contacts = {"351911111111": "Pedro", "123@lid": "Pedro", "351922222222": "Gil"}
    result = birthdays.roster(config, contacts=contacts)
    assert result["Pedro"] == {"date": "07-15", "aliases": ["pedrito"], "jid": "123@lid"}
    assert result["Gil"]["jid"] == "351922222222@c.us"
    assert result["Rafa"]["jid"] == ""


def test_today_line():
    dates = {"Fred": ("06-29", "self"), "Peter": ("06-29", "mined")}
    assert birthdays.today_line(dates, date(2026, 6, 29)) == "Hoje faz anos: Fred, Peter."
    assert birthdays.today_line(dates, date(2026, 6, 30)) == ""


ALIASES = {"Frederico": ["Fred", "Fredji"], "Peter": [], "Rafa": [], "Gil": []}


def rows_for(day, lines):
    return [(datetime(day.year, day.month, day.day, 10, index), sender, text)
            for index, (sender, text) in enumerate(lines)]


def test_mine_finds_the_target_and_recurs_across_years():
    rows = []
    for year in (2024, 2025):
        rows += rows_for(date(year, 6, 29), [
            ("Peter", "Happy birthday Fredjiiii"),
            ("Rafa", "Parabéns Fred!!"),
            ("Gil", "parabéns 🎉"),
            ("Frederico", "obrigado malta"),
        ])
    candidates = birthdays.mine(rows, ALIASES)
    assert [(c.member, c.month_day) for c in candidates] == [("Frederico", "06-29")]
    assert candidates[0].confidence == "high"


def test_mine_needs_three_senders_and_a_target():
    two = rows_for(date(2025, 6, 29), [("Peter", "parabéns Fred"), ("Rafa", "parabéns Fred")])
    nobody = rows_for(date(2025, 6, 30), [("Peter", "parabéns"), ("Rafa", "parabéns"),
                                          ("Gil", "parabéns")])
    assert birthdays.mine(two + nobody, ALIASES) == []


def test_sender_naming_themselves_does_not_count():
    rows = rows_for(date(2025, 3, 6), [
        ("Peter", "Obrigado Peter aqui, parabéns a mim"),
        ("Rafa", "parabéns Gil"), ("Frederico", "parabéns Gil"), ("Gil", "parabéns Peter")])
    candidates = birthdays.mine(rows, ALIASES)
    assert candidates[0].member == "Gil"


def test_adjacent_days_merge_into_the_earlier():
    rows = rows_for(date(2024, 9, 8), [("Peter", "parabéns Gil"), ("Rafa", "parabéns Gil"),
                                        ("Frederico", "parabéns Gil")])
    rows += rows_for(date(2025, 9, 9), [("Peter", "parabéns Gil"), ("Rafa", "parabéns Gil"),
                                         ("Frederico", "parabéns Gil")])
    candidates = birthdays.mine(rows, ALIASES)
    assert [(c.month_day, sorted(c.years)) for c in candidates] == [("09-08", [2024, 2025])]


def test_birthday_line_in_member_suffix():
    members = {"members": [{"name": "Pedro", "aliases": [], "key_facts": ["Tem uma startup."]}]}
    suffix = build_member_prompt_suffix(members, birthdays={"Pedro": "07-15"})
    assert "Tem uma startup. Faz anos a 15 de julho." in suffix


GROUP = "12036300000000@g.us"
BOT = "351900000000@c.us"
ALICE = "351911111111@c.us"


class Members:
    def is_member(self, name):
        return name == "Alice"

    def resolve(self, raw):
        return raw


def birthday_event(text, message_id):
    return {"event": "message", "payload": {
        "id": message_id, "from": GROUP, "participant": ALICE, "body": text,
        "notifyName": "Alice", "mentionedIds": [BOT]}}


def make_adapter(tmp_path, resolver):
    config = {"whatsapp": {"bot_jid": BOT, "send_seen": False,
                           "contacts": {ALICE: "Alice"}},
              "chat": {"birthdays": {"store": str(tmp_path / "birthdays.json")}}}
    client = MockWahaClient(echo=False)
    adapter = WhatsAppAdapter(
        lambda *args, **kwargs: "ok", client, config,
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "s"), max_lines=10),
        message_log=MessageLog(str(tmp_path / "log")), sender_resolver=resolver)
    return adapter, client


def test_aniversario_command_stores_the_speakers_date(tmp_path):
    adapter, client = make_adapter(tmp_path, Members())
    result = adapter.handle_event(birthday_event("@bot /aniversario 8/9", "b1"))
    assert result["command"] == "birthday"
    assert "8 de setembro" in client.sent[-1]["text"]
    stored = birthdays.BirthdayStore(tmp_path / "birthdays.json").load()
    assert stored["Alice"]["self"]["date"] == "09-08"
    assert not list((tmp_path / "log").glob("*.jsonl")) or all(
        "/aniversario" not in path.read_text() for path in (tmp_path / "log").glob("*.jsonl"))


def test_aniversario_without_a_date_gets_usage(tmp_path):
    adapter, client = make_adapter(tmp_path, Members())
    adapter.handle_event(birthday_event("@bot /aniversario", "b2"))
    assert "Exemplo" in client.sent[-1]["text"]
    assert not (tmp_path / "birthdays.json").exists()


def test_aniversario_from_a_non_member_is_refused(tmp_path):
    class Nobody(Members):
        def is_member(self, name):
            return False

    adapter, client = make_adapter(tmp_path, Nobody())
    adapter.handle_event(birthday_event("@bot /aniversario 8/9", "b3"))
    assert "quem é do grupo" in client.sent[-1]["text"]
    assert not (tmp_path / "birthdays.json").exists()
