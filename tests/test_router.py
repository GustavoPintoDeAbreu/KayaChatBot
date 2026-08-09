"""Unit tests for src/chat/router — intent classification and its fallbacks.

No GPU/model: the backend is stubbed. What matters here is that a malformed or
failing router NEVER drops a reply — it falls back to `factual`, which is exactly
the pre-routing behaviour.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import router


class StubBackend:
    """Returns a canned classification, or raises."""

    def __init__(self, output="BANTER", raises=None):
        self.output = output
        self.raises = raises
        self.calls = []

    def generate(self, messages, *, max_new_tokens, sampling):
        self.calls.append({"messages": messages, "max_new_tokens": max_new_tokens,
                           "sampling": sampling})
        if self.raises:
            raise self.raises
        return self.output


def _config(**router_over):
    base = {"enabled": True, "max_new_tokens": 8, "temperature": 0.0,
            "fallback_mode": "factual"}
    base.update(router_over)
    return {
        "chat": {
            "router": base,
            "modes": {
                "banter": {"retrieval": False, "max_new_tokens": 48},
                "mixed": {"retrieval": True, "top_k": 3, "max_new_tokens": 128},
                "factual": {"retrieval": True, "max_new_tokens": 256},
            },
        }
    }


class TestLabelParsing:
    @pytest.mark.parametrize("output,expected", [
        ("BANTER", router.BANTER),
        ("FACTUAL", router.FACTUAL),
        ("MIXED", router.MIXED),
        ("  banter  ", router.BANTER),
        ("The answer is FACTUAL.", router.FACTUAL),
    ])
    def test_parses_mode_labels(self, output, expected):
        route = router.classify(StubBackend(output), _config(), "olá")
        assert route.mode == expected
        assert route.fallback is False

    @pytest.mark.parametrize("output,command", [
        ("CMD_AUDIO", router.CMD_AUDIO),
        ("CMD_TEXT", router.CMD_TEXT),
        ("CMD_CLEAR", router.CMD_CLEAR),
    ])
    def test_parses_commands(self, output, command):
        route = router.classify(StubBackend(output), _config(), "responde em áudio")
        assert route.command == command

    def test_cmd_text_not_shadowed_by_bare_text_match(self):
        """Longest-label-first matching: CMD_TEXT must not be read as something else."""
        route = router.classify(StubBackend("CMD_TEXT"), _config(), "volta a texto")
        assert route.command == router.CMD_TEXT
        assert route.command != router.CMD_AUDIO


class TestFallbacks:
    def test_unparseable_output_falls_back_to_factual(self):
        route = router.classify(StubBackend("I think this is a nice message"), _config(), "olá")
        assert route.mode == router.FACTUAL
        assert route.fallback is True

    def test_backend_exception_falls_back_and_does_not_raise(self):
        route = router.classify(StubBackend(raises=RuntimeError("llama down")), _config(), "olá")
        assert route.mode == router.FACTUAL
        assert route.fallback is True

    def test_disabled_router_returns_fallback_without_calling_model(self):
        backend = StubBackend("BANTER")
        route = router.classify(backend, _config(enabled=False), "olá")
        assert route.mode == router.FACTUAL
        assert backend.calls == []

    def test_empty_message_is_banter_without_calling_model(self):
        backend = StubBackend("FACTUAL")
        route = router.classify(backend, _config(), "   ")
        assert route.mode == router.BANTER
        assert backend.calls == []


class TestRoutingBehaviour:
    def test_banter_disables_retrieval(self):
        assert router.Route(mode=router.BANTER).retrieval_enabled is False
        assert router.Route(mode=router.FACTUAL).retrieval_enabled is True
        assert router.Route(mode=router.MIXED).retrieval_enabled is True

    def test_classification_is_deterministic_and_cheap(self):
        """Routing must be a tiny, greedy call — it runs on every single message."""
        backend = StubBackend("BANTER")
        router.classify(backend, _config(), "Ahahhha")
        call = backend.calls[0]
        assert call["max_new_tokens"] <= 16
        assert call["sampling"]["temperature"] == 0.0

    def test_recent_lines_are_included_as_context(self):
        backend = StubBackend("MIXED")
        router.classify(backend, _config(), "e o Rafa?", recent_lines=["Gustavo: falámos do jantar"])
        user_msg = backend.calls[0]["messages"][-1]["content"]
        assert "jantar" in user_msg

    def test_mode_config_lookup(self):
        cfg = _config()
        assert router.mode_config(cfg, router.BANTER)["retrieval"] is False
        assert router.mode_config(cfg, router.MIXED)["top_k"] == 3
        assert router.mode_config(cfg, "nonexistent") == {}
