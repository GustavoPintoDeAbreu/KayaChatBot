"""Unit tests for extract_instagram() in src/data/extract_all_messages.py."""

import json
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from src.data.extract_all_messages import MessageExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_insta_file(tmp_path: Path, messages: List[Dict]) -> Path:
    """Write a minimal Instagram export JSON and return its Path."""
    data = {
        "participants": [{"name": "Test Group"}],
        "messages": messages,
    }
    p = tmp_path / "message_1.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _ts(offset_secs: int = 0) -> int:
    """Return a timestamp_ms suitable for test messages."""
    return int((time.time() - offset_secs) * 1000)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def extractor() -> MessageExtractor:
    ex = MessageExtractor()
    mock_resolver = MagicMock()
    mock_resolver.resolve.side_effect = lambda name: {
        "Gustavo": "Gustavo",
        "Gil": "Gil",
        "peteroupedro": "Peter",
    }.get(name, name)
    ex._resolver = mock_resolver
    return ex


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_message_extracted(extractor: MessageExtractor, tmp_path: Path) -> None:
    """A plain text message is extracted with correct fields."""
    path = _make_insta_file(tmp_path, [
        {"sender_name": "Gustavo", "timestamp_ms": _ts(), "content": "Boa tarde pessoal"},
    ])
    msgs = extractor.extract_instagram(path)
    assert len(msgs) == 1
    assert msgs[0]["sender"] == "Gustavo"
    assert msgs[0]["source"] == "instagram"
    assert "Boa tarde" in msgs[0]["text"]


def test_reel_share_skipped(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Messages with share.link (reel/post shares) are filtered out."""
    path = _make_insta_file(tmp_path, [
        {
            "sender_name": "Gustavo",
            "timestamp_ms": _ts(),
            "content": "Check this out",
            "share": {"link": "https://www.instagram.com/reel/abc123"},
        },
    ])
    msgs = extractor.extract_instagram(path)
    assert msgs == []


def test_no_content_field_skipped(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Messages without a 'content' field (voice notes, etc.) are skipped."""
    path = _make_insta_file(tmp_path, [
        {"sender_name": "Gustavo", "timestamp_ms": _ts()},  # no content
    ])
    msgs = extractor.extract_instagram(path)
    assert msgs == []


def test_system_notification_skipped(extractor: MessageExtractor, tmp_path: Path) -> None:
    """System messages (liked a message, unsent, etc.) are filtered out."""
    system_msgs = [
        {"sender_name": "Gustavo", "timestamp_ms": _ts(i * 10), "content": text}
        for i, text in enumerate([
            "Gustavo liked a message",
            "Gustavo unsent a message",
            "Gustavo started a call",
            "Gustavo reacted 😂 to your message",
        ])
    ]
    path = _make_insta_file(tmp_path, system_msgs)
    msgs = extractor.extract_instagram(path)
    assert msgs == []


def test_pure_emoji_message_skipped(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Messages containing only emoji (no alphabetic chars) are dropped."""
    path = _make_insta_file(tmp_path, [
        {"sender_name": "Gil", "timestamp_ms": _ts(), "content": "😂😂😂"},
    ])
    msgs = extractor.extract_instagram(path)
    assert msgs == []


def test_anonymous_sender_skipped(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Messages from 'Instagram user' (deleted accounts) are dropped."""
    # Override mock to return None for anonymous senders
    extractor._resolver.resolve.side_effect = lambda name: (
        None if "instagram user" in name.lower() else name
    )
    path = _make_insta_file(tmp_path, [
        {"sender_name": "Instagram user", "timestamp_ms": _ts(), "content": "Hello"},
    ])
    msgs = extractor.extract_instagram(path)
    assert msgs == []


def test_double_encoded_utf8_sender(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Double-encoded UTF-8 senders are passed correctly to the resolver."""
    # "GonÃ§alo" decodes to "Gonçalo"
    extractor._resolver.resolve.side_effect = lambda name: name  # pass-through
    path = _make_insta_file(tmp_path, [
        {
            "sender_name": "GonÃ§alo Mateus",
            "timestamp_ms": _ts(),
            "content": "Boa noite",
        },
    ])
    msgs = extractor.extract_instagram(path)
    assert len(msgs) == 1
    # Resolver was called; sender is whatever the resolver returned
    extractor._resolver.resolve.assert_called()


def test_multiple_valid_messages(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Multiple valid messages all appear in the output."""
    path = _make_insta_file(tmp_path, [
        {"sender_name": "Gustavo", "timestamp_ms": _ts(30), "content": "Bom dia"},
        {"sender_name": "Gil", "timestamp_ms": _ts(20), "content": "Oi pessoal"},
        {"sender_name": "Gustavo", "timestamp_ms": _ts(10), "content": "Tudo bem"},
    ])
    msgs = extractor.extract_instagram(path)
    assert len(msgs) == 3


def test_mixed_valid_and_reel_messages(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Only non-reel messages pass through when reels are mixed in."""
    path = _make_insta_file(tmp_path, [
        {"sender_name": "Gustavo", "timestamp_ms": _ts(20), "content": "Hey"},
        {
            "sender_name": "Gil",
            "timestamp_ms": _ts(10),
            "content": "Watch this",
            "share": {"link": "https://www.instagram.com/reel/xyz"},
        },
    ])
    msgs = extractor.extract_instagram(path)
    assert len(msgs) == 1
    assert msgs[0]["sender"] == "Gustavo"


def test_source_field_is_instagram(extractor: MessageExtractor, tmp_path: Path) -> None:
    """All extracted messages carry source='instagram'."""
    path = _make_insta_file(tmp_path, [
        {"sender_name": "Gustavo", "timestamp_ms": _ts(), "content": "Test message"},
    ])
    msgs = extractor.extract_instagram(path)
    assert all(m["source"] == "instagram" for m in msgs)
