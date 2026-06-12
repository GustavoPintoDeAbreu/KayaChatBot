"""Unit tests for src/data/synthetic_filters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_filters import (
    clean_and_accept,
    is_echo,
    is_refusal,
    is_too_short,
    strip_emojis,
    strip_thinking,
)


class TestStripThinking:
    def test_removes_think_block(self):
        text = "<think>let me reason about this</think>\nO Gustavo é o mais alto."
        assert strip_thinking(text) == "O Gustavo é o mais alto."

    def test_multiline_think_block(self):
        text = "<think>\nline1\nline2\n</think>\n\nResposta final aqui."
        assert strip_thinking(text) == "Resposta final aqui."

    def test_unclosed_think_dropped(self):
        assert strip_thinking("<think>still reasoning and cut off") == ""

    def test_no_think_unchanged(self):
        assert strip_thinking("Resposta normal.") == "Resposta normal."

    def test_clean_and_accept_strips_thinking(self):
        raw = "<think>reasoning</think> O Gustavo é claramente o mais convencido do grupo."
        out = clean_and_accept(raw, "")
        assert out == "O Gustavo é claramente o mais convencido do grupo."
        assert "<think>" not in out


class TestStripEmojis:
    def test_trailing_emoji(self):
        assert strip_emojis("Está tudo bem 😊") == "Está tudo bem"

    def test_punct_kept(self):
        assert strip_emojis("Boa 🎉!") == "Boa!"

    def test_emdash_quotes_preserved(self):
        text = "O Gil — o “rei” — corre."
        assert strip_emojis(text) == text


class TestIsRefusal:
    def test_pt_refusals(self):
        assert is_refusal("Como assistente, não tenho opiniões.")
        assert is_refusal("Não é possível determinar isso.")

    def test_en_refusals(self):
        assert is_refusal("As an AI, I can't determine that.")

    def test_real_answer_not_refusal(self):
        assert not is_refusal("O Gustavo é o mais convencido, sem dúvida.")


class TestShortAndEcho:
    def test_too_short(self):
        assert is_too_short("Sim.", min_words=6)
        assert not is_too_short("O Gustavo é claramente o mais convencido do grupo.", min_words=6)

    def test_echo_detected(self):
        ctx = "=== Conversas ===\nGil: o gugu é o mais alto acho eu\n==="
        assert is_echo('o gugu é o mais alto acho eu', ctx)

    def test_synthesis_not_echo(self):
        ctx = "Gil: o gugu é o mais alto acho eu"
        ans = "Pelo que o grupo diz, parece ser o Gustavo, embora não haja certezas absolutas."
        assert not is_echo(ans, ctx)


class TestCleanAndAccept:
    def test_accepts_good_answer(self):
        ctx = "Gil: techno"
        ans = "O Gil é claramente o mais ligado à música eletrónica do grupo 😎"
        out = clean_and_accept(ans, ctx)
        assert out == "O Gil é claramente o mais ligado à música eletrónica do grupo"

    def test_rejects_refusal(self):
        assert clean_and_accept("Como assistente, não tenho opiniões sobre isso.", "") == ""

    def test_rejects_short(self):
        assert clean_and_accept("Não sei.", "") == ""

    def test_rejects_echo(self):
        ctx = "Peter: desde o liceu"
        assert clean_and_accept("desde o liceu", ctx) == ""
