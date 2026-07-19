"""Unit tests for the local knowledge-extraction backend.

Covers the LocalTeacherProvider adapter, backend selection in
generate_knowledge_base.load_backend, and the end-to-end
call_llm_for_profiles path with a stubbed teacher (no GPU, no model load).
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.local_teacher import LocalTeacherProvider
from src.data import generate_knowledge_base as gkb


class StubTeacher:
    """Stands in for TeacherModel — records calls, returns a canned answer."""

    def __init__(self, answer: str):
        self.answer = answer
        self.calls = []

    def generate(self, system_prompt: str, user_message: str) -> str:
        self.calls.append((system_prompt, user_message))
        return self.answer


# ---------------------------------------------------------------------------
# LocalTeacherProvider
# ---------------------------------------------------------------------------


def test_provider_delegates_to_teacher():
    teacher = StubTeacher("plain answer")
    provider = LocalTeacherProvider(teacher)
    result = provider.generate_text("sys", "user")
    assert result == "plain answer"
    assert teacher.calls == [("sys", "user")]


def test_provider_strips_thinking_blocks():
    teacher = StubTeacher("<think>internal reasoning here</think>\nthe real answer")
    provider = LocalTeacherProvider(teacher)
    assert provider.generate_text("sys", "user") == "the real answer"


# ---------------------------------------------------------------------------
# load_backend selection
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "generation": {"provider": "xai"},
    "knowledge_generation": {"backend": "local", "max_new_tokens": 500, "temperature": 0.3},
    "synthetic_generation": {"teacher_model_id": "stub/teacher-model", "top_p": 0.8, "top_k": 20},
}


def test_load_backend_local_uses_teacher():
    with patch.object(sys.modules["src.data.local_teacher"], "TeacherModel") as MockTeacher:
        MockTeacher.return_value = StubTeacher("x")
        provider, label = gkb.load_backend(BASE_CONFIG)
    assert isinstance(provider, LocalTeacherProvider)
    assert label == "local:stub/teacher-model"
    MockTeacher.assert_called_once()
    model_id, sampling = MockTeacher.call_args.args
    assert model_id == "stub/teacher-model"
    assert sampling["max_new_tokens"] == 500
    assert sampling["temperature"] == 0.3


def test_load_backend_teacher_model_override():
    with patch.object(sys.modules["src.data.local_teacher"], "TeacherModel") as MockTeacher:
        MockTeacher.return_value = StubTeacher("x")
        _, label = gkb.load_backend(BASE_CONFIG, teacher_model_override="other/model")
    assert label == "local:other/model"


def test_load_backend_knowledge_teacher_id_wins_over_synthetic():
    config = {
        **BASE_CONFIG,
        "knowledge_generation": {**BASE_CONFIG["knowledge_generation"], "teacher_model_id": "kg/model"},
    }
    with patch.object(sys.modules["src.data.local_teacher"], "TeacherModel") as MockTeacher:
        MockTeacher.return_value = StubTeacher("x")
        _, label = gkb.load_backend(config)
    assert label == "local:kg/model"


def test_load_backend_no_teacher_model_raises():
    config = {
        "generation": {"provider": "xai"},
        "knowledge_generation": {"backend": "local"},
        "synthetic_generation": {},
    }
    with pytest.raises(ValueError, match="No teacher model configured"):
        gkb.load_backend(config)


def test_load_backend_unknown_backend_raises():
    config = {**BASE_CONFIG, "knowledge_generation": {"backend": "banana"}}
    with pytest.raises(ValueError, match="Unknown knowledge_generation.backend"):
        gkb.load_backend(config)


def test_load_backend_cloud_uses_get_provider():
    sentinel = object()
    with patch("src.llm_providers.get_provider", return_value=sentinel) as mock_gp:
        provider, label = gkb.load_backend(BASE_CONFIG, backend_override="cloud")
    assert provider is sentinel
    assert label == "cloud:xai"
    mock_gp.assert_called_once()


# ---------------------------------------------------------------------------
# End-to-end: call_llm_for_profiles with a stubbed local provider
# ---------------------------------------------------------------------------


def test_call_llm_for_profiles_parses_fenced_json_with_thinking():
    answer = (
        "<think>let me structure the profiles</think>\n"
        "```json\n"
        '{"members": {"Alice": {"interests": ["hiking"]}}, '
        '"recent_summaries": {"Alice": "Talked about hiking."}}\n'
        "```"
    )
    provider = LocalTeacherProvider(StubTeacher(answer))
    result = gkb.call_llm_for_profiles(provider, "prompt", max_retries=1)
    assert result is not None
    assert result["members"]["Alice"]["interests"] == ["hiking"]
    assert result["recent_summaries"]["Alice"] == "Talked about hiking."


def test_call_llm_for_profiles_returns_none_on_garbage():
    provider = LocalTeacherProvider(StubTeacher("not json at all"))
    with patch("time.sleep"):
        result = gkb.call_llm_for_profiles(provider, "prompt", max_retries=2)
    assert result is None
