"""The command registry (lab.yaml `commands:`, src/chat/commands.py) and the parsers agree.

2026-11-25: a wrong /command got "Tenho /bug, /feedback, /aniversario e /clear" in
every chat, /help worked only for the owner, and the /help text was hand-written.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import commands as registry

ROOT = Path(__file__).parent.parent
LISTED = registry.load_lab_yaml(ROOT / "lab.yaml")


def _parsed_tokens():
    adapter = (ROOT / "src/chat/whatsapp_adapter.py").read_text()
    tokens = set()
    for name in ("clear_commands", "bug_commands", "feedback_commands", "birthday_commands",
                 "help_commands"):
        body = re.search(rf"self\.{name} = \{{([^}}]*)\}}", adapter).group(1)
        tokens |= set(re.findall(r'"(/[^"]+)"', body))
    homelab = (ROOT / "src/gateway/homelab.py").read_text()
    tokens |= set(re.findall(r'first == "(/[a-z_]+)"', homelab))
    from src.gateway.help_tap import HELP_COMMANDS
    return tokens | set(HELP_COMMANDS)


def test_every_command_kaya_parses_is_in_the_registry_and_nothing_else():
    """A new command without a lab.yaml line (or a line with no parser) fails here."""
    assert _parsed_tokens() == registry.tokens(LISTED)


def test_the_registry_is_valid():
    assert LISTED and all(c.summary for c in LISTED)
    assert all(c.product == "kaya" for c in LISTED)


def test_a_group_never_sees_owner_commands_even_when_he_asks():
    text = registry.help_text(LISTED, "group", is_owner=True)
    assert "/bug" in text and "faz uma sondagem" in text
    assert "/homelab" not in text and "/gamingoff" not in text


def test_the_owners_dm_sees_everything():
    text = registry.help_text(LISTED, "dm", is_owner=True)
    assert "No grupo:" in text and "Por DM (só tu):" in text
    for command in ("/bug", "/homelaboff", "/homelabon", "/gamingoff", "/homelabrc", "/homelabqc"):
        assert command in text


def test_a_members_dm_sees_no_owner_commands_and_no_group_only_ones():
    text = registry.help_text(LISTED, "dm", is_owner=False)
    assert "/bug" in text and "/homelab" not in text and "sondagem" not in text


def test_suggestions_only_offer_what_the_chat_can_use():
    assert registry.suggest("/hepl", LISTED, "group", False) == "/help"
    assert registry.suggest("/homelabof", LISTED, "dm", True) == "/homelaboff"
    assert registry.suggest("/homelabof", LISTED, "group", True) is None
    assert registry.suggest("/xyzzy", LISTED, "dm", True) is None


def test_the_wrong_command_line_always_points_at_help():
    assert "/help" in registry.unknown_reply("/xyzzy", LISTED, "group", False)
    assert "Querias dizer /aniversario?" in registry.unknown_reply("/aniversaro", LISTED, "dm", False)


def test_an_entry_from_another_repo_merges_in_and_a_bad_row_is_skipped():
    other = registry.from_entries([
        {"product": "idea-pipeline", "cmd": "/idea", "args": "<texto>", "where": ["dm"],
         "who": "owner", "summary": "uma ideia"},
        {"cmd": "no-slash", "summary": "x"}, {"summary": "nothing"}, "junk",
    ])
    merged = registry.merge(other, LISTED)
    assert [c.cmd for c in other] == ["/idea"]
    assert "/idea <texto>" in registry.help_text(merged, "dm", is_owner=True)
    assert "/idea" not in registry.help_text(merged, "group", is_owner=True)


def test_the_guide_mentions_every_command():
    guide = (ROOT / "docs/how-to-use-kaya.md").read_text()
    missing = [c.cmd for c in LISTED if c.cmd and f"`{c.cmd}" not in guide]
    assert not missing, f"docs/how-to-use-kaya.md does not mention {missing}"


def test_one_command_with_several_argument_rows_keeps_every_row():
    rows = registry.from_entries([
        {"cmd": "/idea", "args": "<texto>", "where": ["dm"], "who": "owner", "summary": "uma ideia"},
        {"cmd": "/idea", "args": "list", "where": ["dm"], "who": "owner", "summary": "o estado"},
    ])
    merged = registry.merge(rows, rows, LISTED)
    assert [c.args for c in merged if c.cmd == "/idea"] == ["<texto>", "list"]
    assert "/idea list" in registry.help_text(merged, "dm", is_owner=True)
