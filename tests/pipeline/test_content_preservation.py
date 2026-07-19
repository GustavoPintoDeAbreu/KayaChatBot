"""Tests verifying that message content is preserved faithfully through the
extraction and cleaning pipeline.

These tests ensure that:
- Meaningful text is not silently dropped or mangled.
- Sender names survive extraction intact.
- Source labels ('whatsapp') are set correctly.
"""

from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from src.data.extract_all_messages import MessageExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
