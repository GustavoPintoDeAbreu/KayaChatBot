"""Unit tests for src/chat/response_utils.clean_response."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.response_utils import (
    clean_response,
    coerce_text,
    truncate_history_line,
    wants_long_answer,
)


class TestCleanResponse:
    def test_multiline_answer_preserved(self):
        """The core regression: multi-line answers must NOT be truncated."""
        text = "Gil enjoys techno music.\nHe owns a dog named Cuca.\nHe recently started running."
        result = clean_response(text, user_name="Gustavo")
        assert result == text
        assert result.count("\n") == 2  # all three lines kept

    def test_single_line_unchanged(self):
        assert clean_response("Peter owns a dog named Kobe.", "Gustavo") == "Peter owns a dog named Kobe."

    def test_hallucinated_user_turn_is_cut(self):
        text = "Gil enjoys techno music.\nGustavo: and what about Peter?"
        assert clean_response(text, user_name="Gustavo") == "Gil enjoys techno music."

    def test_generic_user_label_cut(self):
        text = "The group meets in Lisbon.\nUser: tell me more"
        assert clean_response(text, user_name="Gustavo") == "The group meets in Lisbon."

    def test_portuguese_user_label_cut(self):
        text = "O grupo reúne-se em Lisboa.\nUtilizador: e o Gil?"
        assert clean_response(text, user_name="Gustavo") == "O grupo reúne-se em Lisboa."

    def test_leading_bot_label_stripped(self):
        text = "Kaya Bot: Peter enjoys fast food."
        assert clean_response(text, user_name="Gustavo", bot_name="Kaya Bot") == "Peter enjoys fast food."

    def test_leading_user_label_stripped(self):
        # An echoed leading "<user>:" label is removed, but the answer text on
        # that same line is kept (only the label prefix is stripped).
        text = "Gustavo: Gil is a group member."
        assert clean_response(text, user_name="Gustavo") == "Gil is a group member."

    def test_reply_as_stage_direction_stripped(self):
        # The model sometimes echoes "[reply as <name>]" from the prompt.
        text = "Kaya Bot: [reply as Gustavo] Mais uma mensagem para debug"
        result = clean_response(text, user_name="Gustavo", bot_name="Kaya Bot")
        assert result == "Mais uma mensagem para debug"

    def test_responde_como_stage_direction_stripped(self):
        text = "[responde como Gustavo] Olá pessoal"
        assert clean_response(text, user_name="Gustavo") == "Olá pessoal"


class TestWantsLongAnswer:
    def test_short_chitchat_is_short(self):
        assert wants_long_answer("olá, estás aí?") is False
        assert wants_long_answer("quem é o mais burro?") is False

    def test_elaboration_cue_triggers_long(self):
        assert wants_long_answer("explica-me a história do grupo") is True
        assert wants_long_answer("descreve o Gustavo") is True
        assert wants_long_answer("podes listar os membros?") is True
        assert wants_long_answer("why is Gil like that?") is True

    def test_long_question_triggers_long(self):
        long_q = " ".join(["palavra"] * 30)
        assert wants_long_answer(long_q) is True

    def test_empty_is_short(self):
        assert wants_long_answer("") is False


class TestTruncateHistoryLine:
    def test_short_line_unchanged(self):
        line = "Gustavo: olá tudo bem?"
        assert truncate_history_line(line) == line

    def test_long_bot_line_truncated_keeps_label(self):
        body = " ".join(["palavra"] * 100)
        line = f"Kaya Bot: {body}"
        result = truncate_history_line(line, max_words=40)
        assert result.startswith("Kaya Bot: ")
        assert result.endswith("…")
        # 40 body words kept (label not counted)
        assert len(result.split(": ", 1)[1].split()) == 41  # 40 words + the "…" token

    def test_line_without_label(self):
        body = " ".join(["x"] * 50)
        result = truncate_history_line(body, max_words=10)
        assert result.endswith("…")
        assert ": " not in result

    def test_empty_line(self):
        assert truncate_history_line("") == ""

    def test_empty_and_whitespace(self):
        assert clean_response("", "Gustavo") == ""
        assert clean_response("   \n  ", "Gustavo") == ""

    def test_default_user_name_user(self):
        """When non-interactive, user_name defaults to 'User'."""
        text = "Gil plays padel.\nUser: ok"
        assert clean_response(text, user_name="User") == "Gil plays padel."

    def test_no_label_multiline_fully_preserved(self):
        text = "Line one.\nLine two.\nLine three."
        assert clean_response(text, user_name="Gustavo") == text

    def test_old_truncation_would_have_lost_content(self):
        """Guard against regressing to the first-line-only behaviour."""
        text = "First sentence.\nSecond sentence."
        result = clean_response(text, user_name="Gustavo")
        assert "Second sentence." in result


class TestCoerceText:
    """Guards the suggestion-chip formatting bug: a content-block list must never
    render as ``[{'text': …, 'type': 'text'}]``."""

    def test_plain_string_unchanged(self):
        assert coerce_text("Quem é o gugu?") == "Quem é o gugu?"

    def test_content_block_list_flattened(self):
        content = [{"text": "Como podemos melhorar isso?", "type": "text"}]
        assert coerce_text(content) == "Como podemos melhorar isso?"

    def test_single_content_block_dict(self):
        assert coerce_text({"type": "text", "text": "Olá"}) == "Olá"

    def test_multi_part_list_joined(self):
        content = [{"type": "text", "text": "parte um"}, {"type": "text", "text": "parte dois"}]
        assert coerce_text(content) == "parte um parte dois"

    def test_none_and_empty(self):
        assert coerce_text(None) == ""
        assert coerce_text([]) == ""
