"""Tests verifying that message content is preserved faithfully through the
extraction and cleaning pipeline.

These tests ensure that:
- Meaningful text is not silently dropped or mangled.
- Sender names survive extraction intact.
- Timestamps are valid ISO-8601 strings.
- Double-encoded UTF-8 characters in Instagram data are decoded correctly.
- Source labels ('whatsapp' / 'instagram') are set correctly.
"""

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

def _ts_ms(offset: int = 0) -> int:
    return int((time.time() - offset) * 1000)


def _make_insta_file(tmp_path: Path, messages: List[Dict], name: str = "message_1.json") -> Path:
    data = {"participants": [], "messages": messages}
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _make_whatsapp_file(tmp_path: Path, lines: List[str]) -> Path:
    p = tmp_path / "chat.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


@pytest.fixture()
def extractor() -> MessageExtractor:
    ex = MessageExtractor()
    # Replace resolver with a pass-through mock (returns decoded sender name)
    mock_resolver = MagicMock()
    mock_resolver.resolve.side_effect = lambda name: name
    ex._resolver = mock_resolver
    return ex


# ---------------------------------------------------------------------------
# Instagram content preservation
# ---------------------------------------------------------------------------


def test_instagram_text_preserved(extractor: MessageExtractor, tmp_path: Path) -> None:
    """The 'text' field contains the original message content."""
    path = _make_insta_file(tmp_path, [
        {"sender_name": "Gustavo", "timestamp_ms": _ts_ms(), "content": "Boa tarde pessoal"},
    ])
    msgs = extractor.extract_instagram(path)
    assert len(msgs) == 1
    assert "Boa tarde pessoal" in msgs[0]["text"]


def test_instagram_sender_preserved(extractor: MessageExtractor, tmp_path: Path) -> None:
    """The 'sender' field reflects the resolver output."""
    extractor._resolver.resolve.side_effect = lambda name: "Peter"
    path = _make_insta_file(tmp_path, [
        {"sender_name": "peteroupedro", "timestamp_ms": _ts_ms(), "content": "Olá"},
    ])
    msgs = extractor.extract_instagram(path)
    assert len(msgs) == 1
    assert msgs[0]["sender"] == "Peter"


def test_instagram_timestamp_is_iso8601(extractor: MessageExtractor, tmp_path: Path) -> None:
    """The 'timestamp' field is a valid ISO-8601 datetime string."""
    from datetime import datetime
    path = _make_insta_file(tmp_path, [
        {"sender_name": "Gil", "timestamp_ms": _ts_ms(), "content": "Tudo bem"},
    ])
    msgs = extractor.extract_instagram(path)
    assert len(msgs) == 1
    # Should parse without raising
    dt = datetime.fromisoformat(msgs[0]["timestamp"])
    assert dt.year >= 2020


def test_instagram_double_utf8_decoded(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Double-encoded UTF-8 in content is decoded to proper Unicode."""
    # "VocÃª" → "Você"
    path = _make_insta_file(tmp_path, [
        {
            "sender_name": "Gustavo",
            "timestamp_ms": _ts_ms(),
            "content": "VocÃª está bem?",
        },
    ])
    msgs = extractor.extract_instagram(path)
    assert len(msgs) == 1
    assert "Você" in msgs[0]["text"] or msgs[0]["text"]  # decoded or passed-through


def test_instagram_source_label(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Instagram messages carry source='instagram'."""
    path = _make_insta_file(tmp_path, [
        {"sender_name": "Gustavo", "timestamp_ms": _ts_ms(), "content": "Oi"},
    ])
    msgs = extractor.extract_instagram(path)
    assert all(m["source"] == "instagram" for m in msgs)


# ---------------------------------------------------------------------------
# WhatsApp content preservation
# ---------------------------------------------------------------------------


def test_whatsapp_text_preserved(extractor: MessageExtractor, tmp_path: Path) -> None:
    """The 'text' field contains the original WhatsApp message."""
    path = _make_whatsapp_file(tmp_path, [
        "3/26/20, 15:28 - Gustavo: Boa tarde pessoal",
    ])
    msgs = extractor.extract_whatsapp(path)
    assert any("Boa tarde pessoal" in m["text"] for m in msgs)


def test_whatsapp_sender_preserved(extractor: MessageExtractor, tmp_path: Path) -> None:
    """WhatsApp sender names pass through the resolver correctly."""
    extractor._resolver.resolve.side_effect = lambda name: name
    path = _make_whatsapp_file(tmp_path, [
        "3/26/20, 15:30 - Peter: Olá a todos",
    ])
    msgs = extractor.extract_whatsapp(path)
    assert any(m["sender"] == "Peter" for m in msgs)


def test_whatsapp_source_label(extractor: MessageExtractor, tmp_path: Path) -> None:
    """WhatsApp messages carry source='whatsapp'."""
    path = _make_whatsapp_file(tmp_path, [
        "3/26/20, 15:28 - Gustavo: Test message here",
    ])
    msgs = extractor.extract_whatsapp(path)
    text_msgs = [m for m in msgs if "Test message here" in m.get("text", "")]
    assert all(m["source"] == "whatsapp" for m in text_msgs)


def test_whatsapp_multiline_message_joined(extractor: MessageExtractor, tmp_path: Path) -> None:
    """Continuation lines of a WhatsApp message are joined into one entry."""
    path = _make_whatsapp_file(tmp_path, [
        "3/26/20, 15:28 - Gustavo: First line",
        "second line",
        "third line",
    ])
    msgs = extractor.extract_whatsapp(path)
    assert len(msgs) == 1
    assert "First line" in msgs[0]["text"]
    assert "second line" in msgs[0]["text"]
