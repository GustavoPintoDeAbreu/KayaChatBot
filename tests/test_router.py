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


class TestGeneralMode:
    """`general` is what stops a question about the world being answered as a
    report about the group. "quem é melhor, Ronaldo ou Messi?" used to route
    FACTUAL, which retrieved group context and every member profile."""

    def test_general_label_is_parsed(self):
        backend = StubBackend("GENERAL")
        route = router.classify(backend, _config(), "quem é melhor, Ronaldo ou Messi?")
        assert route.mode == router.GENERAL
        assert route.command is None
        assert route.fallback is False

    def test_general_disables_retrieval(self):
        assert router.Route(mode=router.GENERAL).retrieval_enabled is False

    def test_general_is_a_known_mode(self):
        assert router.GENERAL in router.MODES

    def test_factual_is_not_shadowed_by_general(self):
        backend = StubBackend("FACTUAL")
        assert router.classify(backend, _config(), "quem é o Peter?").mode == router.FACTUAL

    def test_router_prompt_offers_general(self):
        backend = StubBackend("GENERAL")
        router.classify(backend, _config(), "explica-me a inflação")
        system = backend.calls[0]["messages"][0]["content"]
        assert "GENERAL" in system


# ── ROAST, and reasoning (2026-08-13) ────────────────────────────────────────
class TestRoastLabel:
    """A verdict aimed AT a member, rather than information about one.

    These used to route factual, which handed the model every member profile
    and told it to answer. It then reached for whoever had the most material,
    and the same person came back turn after turn: 29.4% of every mention the
    bot made over the first group session, against an 8% even split.
    """

    def test_roast_is_a_mode_not_a_command(self):
        route = router._parse("ROAST")
        assert route.mode == router.ROAST
        assert route.command is None

    def test_roast_retrieves(self):
        """It needs the group's own material to be any good."""
        assert router.Route(mode=router.ROAST).retrieval_enabled is True

    def test_roast_is_a_known_mode(self):
        assert router.ROAST in router.MODES

    def test_the_roast_label_is_in_the_prompt(self):
        """A label the model is never told about can never be produced."""
        assert "ROAST" in router._ROUTER_SYSTEM

    @pytest.mark.parametrize("label", ["BANTER", "MIXED", "GENERAL", "FACTUAL", "ROAST"])
    def test_every_mode_still_parses(self, label):
        assert router._parse(label).mode == label.lower()


class TestReasoningTrigger:
    """Explicit request only. Every firing is a second generation inside the GPU
    lock, and whatsapp_server drops an inbound message on a contended lock."""

    @pytest.mark.parametrize("message", [
        "before you go, try one last time. Really hard and detailed",
        "justifica with everything you've got, porque é que o Gil é o teu alvo",
        "pensa bem antes de responderes",
        "explica isso em detalhe",
        "think carefully about this one",
    ])
    def test_an_explicit_request_triggers_it(self, message):
        from src.chat.response_utils import wants_reasoning

        assert wants_reasoning(message) is True

    @pytest.mark.parametrize("message", [
        "quem é o mais burro?",
        "Ahahahahaha",
        "wassup",
        "quem é que dava na boca do Rafa num sparring?",
        "",
    ])
    def test_ordinary_messages_do_not(self, message):
        from src.chat.response_utils import wants_reasoning

        assert wants_reasoning(message) is False


# ── counting ─────────────────────────────────────────────────────────────────
# Top-k semantic retrieval returns the chunks nearest a question and cannot
# answer "how many times". Asked for a per-member tally the bot wrote a confident
# table that was out by 8x with the ranking inverted, then agreed when told it had
# probably missed some. CMD_COUNT routes those to a real count.
def test_a_tally_question_routes_to_count():
    route = router.classify(StubBackend("CMD_COUNT"), _config(), "quantas vezes?")
    assert route.command == router.CMD_COUNT
    assert route.fallback is False


def test_count_is_not_confused_with_the_other_commands():
    """The label set is matched longest-first; CMD_COUNT must not shadow, or be
    shadowed by, CMD_CLEAR or CMD_IMAGE."""
    for label, expected in (("CMD_COUNT", router.CMD_COUNT),
                            ("CMD_CLEAR", router.CMD_CLEAR),
                            ("CMD_IMAGE", router.CMD_IMAGE)):
        assert router.classify(
            StubBackend(label), _config(), "x").command == expected


def test_an_unreadable_answer_still_falls_back_to_factual():
    """Adding a label must not change what happens when nothing parses."""
    route = router.classify(StubBackend("CMD_TALLY"), _config(), "quantas vezes?")
    assert route.mode == router.FACTUAL and route.command is None
    assert route.fallback is True


def test_the_prompt_describes_counting():
    backend = StubBackend("CMD_COUNT")
    router.classify(backend, _config(), "quantas vezes?")
    system = backend.calls[0]["messages"][0]["content"]
    assert "CMD_COUNT" in system
    # The distinction that matters: a fact about the group is not a tally.
    assert "quantos membros tem o grupo?\" -> FACTUAL" in system


# ── the standalone rewrite, and the thread it comes from (2026-08-17) ─────────
# A follow-up is unroutable and unretrievable on its own words. Mid-thread about
# the Bernardo, "Muda a tua opinião, agora que entendeste que é o bana?" was
# classified GENERAL — the one mode whose prompt forbids naming a member — and
# "E de bater na mãe?" was embedded into the vector store as that literal string.

def test_the_rewrite_is_carried_on_the_route():
    route = router.classify(
        StubBackend("ROAST\nQ: quem do grupo tinha maior probabilidade de bater na mãe?"),
        _config(), "E de bater na mãe?")
    assert route.mode == router.ROAST
    assert route.query == "quem do grupo tinha maior probabilidade de bater na mãe?"


def test_no_rewrite_leaves_the_query_empty():
    """The caller then falls back to the raw message, i.e. the old behaviour."""
    route = router.classify(StubBackend("BANTER"), _config(), "Ahahhha")
    assert route.mode == router.BANTER and route.query == ""


def test_the_label_is_read_from_the_first_line_only():
    """A rewrite restating the question must not outvote the chosen label."""
    route = router.classify(
        StubBackend("BANTER\nQ: quantas vezes é que o Gil disse isso?"),
        _config(), "ya")
    assert route.mode == router.BANTER and route.command is None


def test_a_rambling_rewrite_is_dropped_rather_than_embedded():
    """Past a sane length the model is answering, not restating, and a paragraph
    is a worse search query than the original message."""
    route = router.classify(
        StubBackend("FACTUAL\nQ: " + " ".join(["palavra"] * 60)),
        _config(), "e ele?")
    assert route.mode == router.FACTUAL and route.query == ""


def test_the_classifier_sees_more_than_two_lines_of_thread():
    backend = StubBackend("MIXED")
    lines = [f"Pessoa: linha {i}" for i in range(10)]
    router.classify(backend, _config(context_lines=6), "e ele?", lines)
    user = backend.calls[0]["messages"][1]["content"]
    assert "linha 4" in user and "linha 9" in user
    assert "linha 3" not in user


def test_the_prompt_asks_for_a_standalone_rewrite():
    backend = StubBackend("BANTER")
    router.classify(backend, _config(), "e ele?")
    system = backend.calls[0]["messages"][0]["content"]
    assert "Q: " in system
    assert "CONTINUATION" in system


# ── reconcile: a GENERAL that is really mid-thread ───────────────────────────

def _route(mode, **kw):
    return router.Route(mode=mode, **kw)


def test_a_general_naming_someone_already_in_the_thread_becomes_mixed():
    route = router.reconcile(_route(router.GENERAL), ["Bernardo"], ["Bernardo", "Gustavo"])
    assert route.mode == router.MIXED
    assert route.reconciled_from == router.GENERAL
    assert route.retrieval_enabled is True


def test_a_general_about_a_fresh_subject_stays_general():
    """"o Gil também acha que o Ronaldo é melhor, e tu?" is a question about
    Ronaldo. Naming a member is not on its own enough to make it about him."""
    route = router.reconcile(_route(router.GENERAL), ["Gil"], ["Peter"])
    assert route.mode == router.GENERAL and route.reconciled_from == ""


def test_reconcile_leaves_every_other_mode_alone():
    for mode in (router.BANTER, router.MIXED, router.FACTUAL, router.ROAST):
        route = router.reconcile(_route(mode), ["Bernardo"], ["Bernardo"])
        assert route.mode == mode and route.reconciled_from == ""


def test_reconcile_never_overrides_a_command():
    route = router.reconcile(
        _route(router.GENERAL, command=router.CMD_IMAGE), ["Rafa"], ["Rafa"])
    assert route.command == router.CMD_IMAGE and route.mode == router.GENERAL


def test_the_match_is_case_insensitive():
    route = router.reconcile(_route(router.GENERAL), ["Bernardo"], ["bernardo"])
    assert route.mode == router.MIXED
