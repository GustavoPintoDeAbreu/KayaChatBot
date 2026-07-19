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
