"""Unit tests for LLMProvider._retry_with_backoff (rate-limit resilience)."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_providers.base import LLMProvider


class FakeProvider(LLMProvider):
    """Concrete provider exposing only the retry helper."""

    def generate_conversations(self, prompt):
        return []

    def generate_text(self, system_prompt, user_prompt):
        return ""

    def chat_completion(self, messages):
        return ""


@pytest.fixture
def provider():
    return FakeProvider({"retry": {"max_attempts": 3, "delay_seconds": 1}})


def test_returns_result_on_first_success(provider):
    calls = []

    def func():
        calls.append(1)
        return "ok"

    assert provider._retry_with_backoff(func) == "ok"
    assert len(calls) == 1


def test_retries_on_rate_limit_then_succeeds(provider):
    calls = []

    def func():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 Too Many Requests: rate limit exceeded")
        return "ok"

    with patch("time.sleep") as sleep:
        assert provider._retry_with_backoff(func) == "ok"
    assert len(calls) == 3
    assert sleep.call_count == 2


def test_raises_after_max_attempts_on_persistent_rate_limit(provider):
    calls = []

    def func():
        calls.append(1)
        raise RuntimeError("quota exceeded")

    with patch("time.sleep"):
        with pytest.raises(RuntimeError, match="quota"):
            provider._retry_with_backoff(func)
    assert len(calls) == 3


def test_non_rate_limit_error_fails_immediately(provider):
    calls = []

    def func():
        calls.append(1)
        raise ValueError("malformed JSON in response")

    with patch("time.sleep") as sleep:
        with pytest.raises(ValueError, match="malformed"):
            provider._retry_with_backoff(func)
    assert len(calls) == 1
    assert sleep.call_count == 0


def test_backoff_delay_grows_exponentially(provider):
    calls = []

    def func():
        calls.append(1)
        raise RuntimeError("rate limited")

    delays = []
    with patch("time.sleep", side_effect=delays.append):
        with pytest.raises(RuntimeError):
            provider._retry_with_backoff(func)
    assert len(delays) == 2
    assert delays[1] > delays[0]
    assert delays[0] >= 1
    assert delays[1] >= 2
