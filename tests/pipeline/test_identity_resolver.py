"""Unit tests for src/data/identity_resolver.py."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.identity_resolver import SenderResolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MEMBERS = [
    {"name": "Gustavo", "aliases": ["gustavo"]},
    {"name": "Gil", "aliases": ["gil", "gilão", "gilao"]},
    {"name": "Peter", "aliases": ["peter", "piteru"]},
    {"name": "Murgeiro", "aliases": ["murgeiro", "joao murgeiro"]},
    {"name": "Frederico", "aliases": ["frederico", "fred"]},
]

SENDER_ALIASES = {
    "O Pedro do Costume": "Peter",
    "joao_murgeiro": "Murgeiro",
    "Fred NL": "Frederico",
}


@pytest.fixture()
def members_file(tmp_path: Path) -> Path:
    path = tmp_path / "group_members.json"
    path.write_text(json.dumps(MEMBERS), encoding="utf-8")
    return path


@pytest.fixture()
def resolver(members_file: Path) -> SenderResolver:
    return SenderResolver(members_file, sender_aliases=SENDER_ALIASES)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_config_override_exact_key(resolver: SenderResolver) -> None:
    """Config overrides are resolved first, regardless of alias matching."""
    assert resolver.resolve("O Pedro do Costume") == "Peter"
    assert resolver.resolve("joao_murgeiro") == "Murgeiro"
    assert resolver.resolve("Fred NL") == "Frederico"


def test_exact_alias_match(resolver: SenderResolver) -> None:
    """An alias that exactly matches a member name resolves correctly."""
    assert resolver.resolve("gilão") == "Gil"
    assert resolver.resolve("piteru") == "Peter"
    assert resolver.resolve("fred") == "Frederico"


def test_exact_alias_case_insensitive(resolver: SenderResolver) -> None:
    """Alias matching is case-insensitive."""
    assert resolver.resolve("GILÃO") == "Gil"
    assert resolver.resolve("Piteru") == "Peter"
    assert resolver.resolve("FRED") == "Frederico"


def test_token_match_full_name(resolver: SenderResolver) -> None:
    """A token in a full name matches the member via alias list."""
    # "Gil" is an alias for the member Gil
    result = resolver.resolve("João Gil")
    assert result == "Gil"


def test_resolve_always_returns_str(resolver: SenderResolver) -> None:
    """resolve() never returns None — unknown senders pass through as-is."""
    assert resolver.resolve("") == ""
    assert isinstance(resolver.resolve("Someone Unknown"), str)


def test_non_member_preserves_name(resolver: SenderResolver) -> None:
    """A sender not in any member list is returned as-is."""
    result = resolver.resolve("Maria Costa")
    assert result == "Maria Costa"


def test_is_member_true(resolver: SenderResolver) -> None:
    """is_member() returns True for canonical member names."""
    assert resolver.is_member("Gustavo") is True
    assert resolver.is_member("gustavo") is True


def test_is_member_false(resolver: SenderResolver) -> None:
    """is_member() returns False for non-members."""
    assert resolver.is_member("RandomPerson") is False


# ── the real Kaya roster, and the ambiguity in it (2026-08-13) ───────────────
# Two members are called Ricardo: Ricardo Romano (the member "Romano") and
# Ricardo Alberto ("Ricky"). Bare "Ricardo" therefore identifies nobody, and
# guessing between two real people is worse than declining.
KAYA = [
    {"name": "Gil", "aliases": ["gil", "gilão"]},
    {"name": "Murgeiro", "aliases": ["murgeiro", "joão", "jao"]},
    {"name": "Carnall", "aliases": ["carnall", "tomás", "loirinho"]},
    {"name": "Frederico", "aliases": ["frederico", "fred"]},
    {"name": "David", "aliases": ["david", "raminhos"]},
    {"name": "Romano", "aliases": ["romano", "ricardo romano"]},
    {"name": "Ricky", "aliases": ["ricky", "ricardo alberto"]},
    {"name": "Daniel", "aliases": ["daniel", "carmelino", "caramelo", "parceiro",
                                   "daniel carmelino"]},
]
KAYA_OVERRIDES = {"fredericop167": "Frederico", "Gil João": "Gil"}


@pytest.fixture
def kaya(tmp_path):
    path = tmp_path / "group_members.json"
    path.write_text(json.dumps({"members": KAYA}), encoding="utf-8")
    return SenderResolver(path, KAYA_OVERRIDES)


@pytest.mark.parametrize("raw,expected", [
    # The live bug: only the FIRST token used to be tried, and it is "tomas"
    # while the alias is "tomás". The second token is an exact alias.
    ("Tomas Carnall", "Carnall"),
    # A display name that is a username and matches nothing — sender_aliases.
    ("fredericop167", "Frederico"),
    # Private references the group actually uses.
    ("Raminhos", "David"),
    ("Caramelo", "Daniel"),
    ("Parceiro", "Daniel"),
    ("Daniel Carmelino", "Daniel"),
    # Both Ricardos, named in full.
    ("Ricardo Romano", "Romano"),
    ("Ricardo Alberto", "Ricky"),
    ("Ricky", "Ricky"),
])
def test_the_real_senders_resolve(kaya, raw, expected):
    assert kaya.resolve(raw) == expected


def test_a_bare_ricardo_resolves_to_nobody(kaya):
    """The property that matters most. Two real people answer to it, so the
    resolver must decline rather than pick whichever it saw first."""
    assert kaya.is_member(kaya.resolve("Ricardo")) is False


def test_adding_ricky_did_not_break_romano(kaya):
    """"ricardo alberto" is a two-word alias, matched by the exact-match step
    before token matching runs, so it adds no new single token. A bare "ricardo"
    alias would have made "Ricardo Romano" match two members."""
    assert kaya.resolve("Ricardo Romano") == "Romano"


def test_an_ambiguous_display_name_needs_an_override(kaya):
    """"Gil João": "gil" is Gil and "joão" is Murgeiro, so both tokens match but
    different people. Only an exact override can settle it."""
    assert kaya.resolve("Gil João") == "Gil"


def test_a_display_name_whose_tokens_agree_needs_no_help(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"members": KAYA}), encoding="utf-8")

    assert SenderResolver(path).resolve("João Murgeiro") == "Murgeiro"


def test_a_genuine_outsider_is_preserved(kaya):
    assert kaya.resolve("Iñaki") == "Iñaki"
    assert kaya.is_member("Iñaki") is False


# ── reporting who could not be placed ────────────────────────────────────────
def test_unresolved_senders_are_reported(kaya, tmp_path):
    """Silent mis-attribution is how "Tomas Carnall" went unnoticed for months."""
    logs = tmp_path / "live_messages"
    logs.mkdir()
    (logs / "shared.jsonl").write_text("\n".join(
        json.dumps({"sender": s, "text": "olá"}) for s in
        ["Gil", "Tomas Carnall", "Ricardo", "Ricardo", "Iñaki"]), encoding="utf-8")

    unresolved = kaya.unresolved_senders(logs)

    assert unresolved == {"Ricardo": 2, "Iñaki": 1}


def test_a_missing_log_directory_is_not_an_error(kaya, tmp_path):
    assert kaya.unresolved_senders(tmp_path / "nope") == {}


def test_a_corrupt_log_does_not_stop_startup(kaya, tmp_path):
    logs = tmp_path / "live_messages"
    logs.mkdir()
    (logs / "good.jsonl").write_text(json.dumps({"sender": "Ricardo"}), encoding="utf-8")
    (logs / "bad.jsonl").write_text("{not json", encoding="utf-8")

    assert kaya.unresolved_senders(logs) == {"Ricardo": 1}
