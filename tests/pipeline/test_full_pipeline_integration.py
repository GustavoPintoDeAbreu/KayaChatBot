"""End-to-end integration tests for the WhatsApp extraction pipeline.

These tests run the full extraction path against synthetic fixture files that
mimic the real data format, verifying that the pipeline components work together
correctly from raw source data to cleaned, deduplicated JSONL output.
"""

import json
from pathlib import Path
from typing import List

import pytest

from src.data.extract_all_messages import MessageExtractor
from src.data.identity_resolver import SenderResolver


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

MEMBERS = [
    {"name": "Gustavo", "aliases": ["gustavo"]},
    {"name": "Gil", "aliases": ["gil", "gilão", "gilao"]},
    {"name": "Peter", "aliases": ["peter", "piteru"]},
    {"name": "Murgeiro", "aliases": ["murgeiro", "joao murgeiro"]},
    {"name": "Frederico", "aliases": ["frederico", "fred"]},
]

# Raw WhatsApp sender strings that need a manual override to resolve
SENDER_ALIASES = {
    "O Pedro do Costume": "Peter",
    "Fred NL": "Frederico",
}


def _make_whatsapp_file(tmp_path: Path, lines: List[str]) -> Path:
    p = tmp_path / "chat.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


@pytest.fixture()
def members_file(tmp_path: Path) -> Path:
    p = tmp_path / "group_members.json"
    p.write_text(json.dumps(MEMBERS), encoding="utf-8")
    return p


@pytest.fixture()
def resolver(members_file: Path) -> SenderResolver:
    return SenderResolver(members_file, sender_aliases=SENDER_ALIASES)


@pytest.fixture()
def extractor(resolver: SenderResolver) -> MessageExtractor:
    ex = MessageExtractor()
    ex._resolver = resolver
    return ex


# ---------------------------------------------------------------------------
# Integration tests: Identity resolver ↔ MessageExtractor
# ---------------------------------------------------------------------------


def test_config_override_resolves_in_extraction(
    extractor: MessageExtractor, tmp_path: Path
) -> None:
    """An opaque raw sender resolves to the member via config override."""
    path = _make_whatsapp_file(tmp_path, [
        "3/26/20, 15:28 - O Pedro do Costume: Bom dia a todos",
    ])
    msgs = extractor.extract_whatsapp(path)
    assert len(msgs) == 1
    assert msgs[0]["sender"] == "Peter"


def test_full_name_resolves_via_token_match(
    extractor: MessageExtractor, tmp_path: Path
) -> None:
    """A full sender name like 'João Gil' resolves to Gil via token match."""
    path = _make_whatsapp_file(tmp_path, [
        "3/26/20, 15:28 - João Gil: Olá pessoal como estão",
    ])
    msgs = extractor.extract_whatsapp(path)
    assert len(msgs) == 1
    assert msgs[0]["sender"] == "Gil"


def test_non_member_name_preserved(extractor: MessageExtractor, tmp_path: Path) -> None:
    """A non-member sender passes through with their name intact."""
    path = _make_whatsapp_file(tmp_path, [
        "3/26/20, 15:28 - Maria Costa: Bem-vindos ao grupo pessoal",
    ])
    msgs = extractor.extract_whatsapp(path)
    assert len(msgs) == 1
    assert msgs[0]["sender"] == "Maria Costa"


def test_system_messages_filtered(extractor: MessageExtractor, tmp_path: Path) -> None:
    """System lines (no 'Sender:' part) and media are filtered out."""
    path = _make_whatsapp_file(tmp_path, [
        "3/26/20, 15:28 - Gil João added you",
        "3/26/20, 15:29 - Gustavo: <Media omitted>",
        "3/26/20, 15:30 - Gustavo: Real message here",
    ])
    msgs = extractor.extract_whatsapp(path)
    assert len(msgs) == 1
    assert "Real message here" in msgs[0]["text"]


def test_source_label_is_whatsapp(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Extracted messages carry source='whatsapp'."""
    path = _make_whatsapp_file(tmp_path, [
        "3/26/20, 15:28 - Gustavo: Uma mensagem de teste",
    ])
    msgs = extractor.extract_whatsapp(path)
    assert msgs and all(m["source"] == "whatsapp" for m in msgs)


def test_all_required_fields_present(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Every extracted message has timestamp, sender, text, source."""
    path = _make_whatsapp_file(tmp_path, [
        "3/26/20, 15:28 - Gustavo: Hello there friends",
        "3/26/20, 15:40 - João Gil: World is big today",
    ])
    msgs = extractor.extract_whatsapp(path)
    assert len(msgs) == 2
    required_keys = {"timestamp", "sender", "text", "source"}
    for msg in msgs:
        assert required_keys.issubset(msg.keys()), f"Missing keys in: {msg}"


def test_timestamp_is_iso8601(extractor: MessageExtractor, tmp_path: Path) -> None:
    """The 'timestamp' field parses as ISO-8601."""
    from datetime import datetime
    path = _make_whatsapp_file(tmp_path, [
        "3/26/20, 15:28 - Gustavo: Data e hora corretas",
    ])
    msgs = extractor.extract_whatsapp(path)
    assert len(msgs) == 1
    dt = datetime.fromisoformat(msgs[0]["timestamp"])
    assert dt.year == 2020
