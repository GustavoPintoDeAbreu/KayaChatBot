"""End-to-end integration tests for the Instagram + WhatsApp extraction pipeline.

These tests run the full extraction path against synthetic fixture files that
mimic the real data format, verifying that the pipeline components work together
correctly from raw source data to cleaned, deduplicated JSONL output.

Slow tests are marked with ``@pytest.mark.slow`` and require the real ChromaDB
and model artefacts.  The lightweight integration tests (no DB required) run by
default.
"""

import json
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

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

SENDER_ALIASES = {
    "peteroupedro": "Peter",
    "joao_murgeiro": "Murgeiro",
    "Driehoek": "Frederico",
}


def _ts_ms(offset: int = 0) -> int:
    return int((time.time() - offset) * 1000)


def _make_insta_file(tmp_path: Path, messages: List[Dict]) -> Path:
    data = {"participants": [], "messages": messages}
    p = tmp_path / "message_1.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


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
    """peteroupedro in Instagram data resolves to Peter via config override."""
    path = _make_insta_file(tmp_path, [
        {
            "sender_name": "peteroupedro",
            "timestamp_ms": _ts_ms(),
            "content": "Bom dia a todos",
        }
    ])
    msgs = extractor.extract_instagram(path)
    assert len(msgs) == 1
    assert msgs[0]["sender"] == "Peter"


def test_double_encoded_name_resolves_via_token_match(
    extractor: MessageExtractor, tmp_path: Path
) -> None:
    """Double-encoded sender like 'JoÃ£o Gil' resolves to Gil via token match."""
    path = _make_insta_file(tmp_path, [
        {
            "sender_name": "JoÃ£o Gil",
            "timestamp_ms": _ts_ms(),
            "content": "Olá pessoal",
        }
    ])
    msgs = extractor.extract_instagram(path)
    assert len(msgs) == 1
    assert msgs[0]["sender"] == "Gil"


def test_anonymous_sender_dropped_end_to_end(
    extractor: MessageExtractor, tmp_path: Path
) -> None:
    """Messages from 'Instagram user' are dropped throughout the full path."""
    path = _make_insta_file(tmp_path, [
        {
            "sender_name": "Instagram user",
            "timestamp_ms": _ts_ms(),
            "content": "Some message",
        }
    ])
    msgs = extractor.extract_instagram(path)
    assert msgs == []


def test_reel_and_system_messages_filtered(
    extractor: MessageExtractor, tmp_path: Path
) -> None:
    """Reels, reactions, and system messages are all filtered out."""
    path = _make_insta_file(tmp_path, [
        {
            "sender_name": "Gustavo",
            "timestamp_ms": _ts_ms(30),
            "content": "Real message",
        },
        {
            "sender_name": "Gil",
            "timestamp_ms": _ts_ms(20),
            "content": "Check this",
            "share": {"link": "https://www.instagram.com/reel/xyz"},
        },
        {
            "sender_name": "Gil",
            "timestamp_ms": _ts_ms(10),
            "content": "Gustavo liked a message",
        },
    ])
    msgs = extractor.extract_instagram(path)
    assert len(msgs) == 1
    assert msgs[0]["sender"] == "Gustavo"


def test_mixed_sources_whatsapp_and_instagram(
    extractor: MessageExtractor, tmp_path: Path
) -> None:
    """WhatsApp and Instagram extraction both produce correct source labels."""
    insta_path = _make_insta_file(tmp_path, [
        {
            "sender_name": "peteroupedro",
            "timestamp_ms": _ts_ms(),
            "content": "Via Instagram",
        }
    ])
    wpp_path = _make_whatsapp_file(tmp_path, [
        "3/26/20, 15:28 - Gustavo: Via WhatsApp",
    ])
    insta_msgs = extractor.extract_instagram(insta_path)
    wpp_msgs = extractor.extract_whatsapp(wpp_path)

    insta_sources = {m["source"] for m in insta_msgs}
    wpp_sources = {m["source"] for m in wpp_msgs if "Via WhatsApp" in m.get("text", "")}

    assert insta_sources == {"instagram"}
    assert wpp_sources == {"whatsapp"}


def test_non_member_name_preserved(extractor: MessageExtractor, tmp_path: Path) -> None:
    """A non-member sender passes through with their decoded name intact."""
    path = _make_insta_file(tmp_path, [
        {
            "sender_name": "Maria Costa",
            "timestamp_ms": _ts_ms(),
            "content": "Bem-vindos",
        }
    ])
    msgs = extractor.extract_instagram(path)
    assert len(msgs) == 1
    assert msgs[0]["sender"] == "Maria Costa"


def test_all_required_fields_present(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Every extracted Instagram message has timestamp, sender, text, source."""
    path = _make_insta_file(tmp_path, [
        {"sender_name": "Gustavo", "timestamp_ms": _ts_ms(), "content": "Hello"},
        {"sender_name": "Gil", "timestamp_ms": _ts_ms(5), "content": "World"},
    ])
    msgs = extractor.extract_instagram(path)
    required_keys = {"timestamp", "sender", "text", "source"}
    for msg in msgs:
        assert required_keys.issubset(msg.keys()), f"Missing keys in: {msg}"


# ---------------------------------------------------------------------------
# Slow tests (marked, skipped unless --runslow / -m slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_instagram_files_extract_nonzero_messages(extractor: MessageExtractor) -> None:
    """The real data/insta/ files each yield a nonzero number of messages."""
    insta_dir = Path("data/insta")
    if not insta_dir.exists():
        pytest.skip("data/insta/ not present in this environment")
    for json_file in sorted(insta_dir.glob("message_*.json")):
        msgs = extractor.extract_instagram(json_file)
        assert len(msgs) > 0, f"No messages extracted from {json_file.name}"
