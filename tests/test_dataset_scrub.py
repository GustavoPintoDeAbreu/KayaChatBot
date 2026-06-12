"""Unit tests for the training-data scrubbing in SyntheticDatasetMerger.

Covers the Phase 3 surgery: emoji stripping from assistant targets, dropping
conversations that mention blocked terms, and baking the config-sourced persona
(not the legacy hardcoded prompt) into Kaya examples. Uses model_id=None so the
manual chat-template fallback runs without loading a tokenizer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.readers import SyntheticDatasetMerger

NEW_PROMPT = "És o bot do grupo. Dás a tua avaliação. Não terminas com emojis."


def _merger():
    return SyntheticDatasetMerger(
        model_id=None,
        kaya_system_prompt=NEW_PROMPT,
        blocked_terms=["Dolby Atmos", "8D audio"],
    )


def _kaya_conv(user, assistant):
    return {
        "source": "synthetic_kaya",
        "conversations": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


class TestStripEmojis:
    def test_trailing_emoji_removed(self):
        m = _merger()
        assert m.strip_emojis("Está tudo bem 😊") == "Está tudo bem"

    def test_space_before_punct_fixed(self):
        m = _merger()
        assert m.strip_emojis("Boa ideia 🎉!") == "Boa ideia!"

    def test_emdash_and_quotes_preserved(self):
        m = _merger()
        text = "O Gil — o “ladies man” — corre."
        assert m.strip_emojis(text) == text


class TestFormatConversation:
    def test_emoji_stripped_from_assistant_turn(self):
        m = _merger()
        out = m.format_conversation(_kaya_conv("Tudo bem?", "Sim, tudo 😊"))
        assert "😊" not in out
        assert m.stripped_emoji == 1

    def test_blocked_conversation_dropped(self):
        m = _merger()
        out = m.format_conversation(_kaya_conv("De que gosta o Gil?", "Gil curte Dolby Atmos."))
        assert out is None
        assert m.dropped_blocked == 1

    def test_new_persona_injected(self):
        m = _merger()
        out = m.format_conversation(_kaya_conv("Olá", "Olá, em que posso ajudar?"))
        assert NEW_PROMPT in out

    def test_clean_conversation_kept(self):
        m = _merger()
        out = m.format_conversation(_kaya_conv("Quem é o mais alto?", "O Gustavo, acho eu."))
        assert out is not None
        assert "Gustavo" in out
        assert m.dropped_blocked == 0
