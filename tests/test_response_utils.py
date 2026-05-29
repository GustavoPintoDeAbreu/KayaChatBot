"""Unit tests for src/chat/response_utils.clean_response."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.response_utils import clean_response


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
