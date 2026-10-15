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


class TestDebateMode:
    """DEBATE, added 2026-09-05.

    The turn that motivated it: Frederico asked the bot to referee the
    Pedro/Bernardo argument and be analytical, and the live log records it as
    `roast` with 9,962 chars of member profiles and no web search.
    """

    def test_debate_label_is_parsed(self):
        route = router.classify(StubBackend("DEBATE"), _config(), "debate me")
        assert route.mode == router.DEBATE
        assert route.fallback is False

    def test_debate_retrieves(self):
        """An argument about the group's own thread needs the thread."""
        route = router.classify(StubBackend("DEBATE"), _config(), "quem tem razão?")
        assert route.retrieval_enabled is True

    def test_debate_is_in_modes(self):
        assert router.DEBATE in router.MODES

    def test_debate_carries_a_rewritten_query(self):
        route = router.classify(
            StubBackend("DEBATE\nQ: quem tem razão sobre o custo da comida?"),
            _config(), "e então?")
        assert route.query == "quem tem razão sobre o custo da comida?"

    def test_the_prompt_separates_a_verdict_on_a_claim_from_one_on_a_person(self):
        """The rubric has to state the ROAST/DEBATE line, or the model guesses."""
        assert "DEBATE and ROAST" in router._ROUTER_SYSTEM
        assert "DEBATE and GENERAL" in router._ROUTER_SYSTEM

    def test_the_prompt_forbids_volunteering_into_an_argument(self):
        """Gustavo: "deixa de discutir por AI @Bernardo / Wack ass tactic"."""
        assert "never puts itself into an argument" in router._ROUTER_SYSTEM

    def test_debate_does_not_shadow_another_label(self):
        for output, expected in (("BANTER", router.BANTER),
                                 ("ROAST", router.ROAST),
                                 ("GENERAL", router.GENERAL),
                                 ("DEBATE", router.DEBATE)):
            route = router.classify(StubBackend(output), _config(), "x")
            assert route.mode == expected, output


# ── who "me" is (2026-09-28) ─────────────────────────────────────────────────
# The router was shown only the words, so the same live "Do the same for me"
# rewrote to a roast of Rafa or of Peter depending on nothing in the message.
def test_the_router_is_told_who_is_writing():
    backend = StubBackend("ROAST\nQ: faz um roast ao Rafa")

    router.classify(backend, _config(), "Do the same for me", ["Peter: roast me"],
                    speaker="Rafa")

    assert "written by Rafa" in backend.calls[0]["messages"][-1]["content"]


def test_no_speaker_keeps_the_old_prompt():
    backend = StubBackend("BANTER")

    router.classify(backend, _config(), "lol")

    assert "Message to classify:\nlol" in backend.calls[0]["messages"][-1]["content"]


def test_the_rubric_routes_a_question_wrapped_in_an_insult_by_the_question():
    assert '"tu és tão burro que nem sabes o que é o jogo do titz, explica lá" -> FACTUAL' \
        in router._ROUTER_SYSTEM
    assert '"Kaya Bot" is the bot itself' in router._ROUTER_SYSTEM


def test_a_friendly_ranking_is_not_a_roast():
    """Only an unflattering verdict is a roast; "quem é o mais engraçado?" is an opinion."""
    assert '"quem é o mais engraçado?" -> FACTUAL' in router._ROUTER_SYSTEM
    assert '"quem é o mais engraçado?" -> ROAST' not in router._ROUTER_SYSTEM
    assert '"good morning bro"' in router._ROUTER_SYSTEM


def test_the_reviewer_flags_a_put_down_nobody_asked_for():
    from src.chat.reply_review import REVIEW_SYSTEM

    assert "boca que ninguém pediu" in REVIEW_SYSTEM


# ── CMD_POLL: turning what the chat discussed into a WhatsApp poll ───────────
# A poll is built from the chat's own log, so the label only matters for
# requests to MAKE one: talking about a poll, voting, or asking which option is
# best stays in the ordinary modes.

def test_a_poll_request_routes_to_poll():
    route = router.classify(StubBackend("CMD_POLL"), _config(),
                            "faz uma sondagem com os sítios que falámos")
    assert route.command == router.CMD_POLL
    assert route.fallback is False


def test_poll_is_not_confused_with_the_other_commands():
    for label, expected in (("CMD_POLL", router.CMD_POLL),
                            ("CMD_COUNT", router.CMD_COUNT),
                            ("CMD_IMAGE", router.CMD_IMAGE),
                            ("CMD_CLEAR", router.CMD_CLEAR)):
        assert router.classify(StubBackend(label), _config(), "x").command == expected


def test_a_poll_request_keeps_a_standalone_query():
    """The Q: line names what the poll is about; the engine reads the log for it."""
    route = router.classify(
        StubBackend("CMD_POLL\nQ: sondagem com os restaurantes propostos para o almoço de Natal"),
        _config(), "faz uma poll com isso")
    assert route.command == router.CMD_POLL
    assert route.query == "sondagem com os restaurantes propostos para o almoço de Natal"


def test_the_prompt_describes_polling():
    backend = StubBackend("CMD_POLL")
    router.classify(backend, _config(), "faz uma sondagem")
    system = backend.calls[0]["messages"][0]["content"]
    assert "CMD_POLL" in system
    # The line that matters: asking for an opinion is not a request to make one.
    assert '"qual é o melhor restaurante para o almoço?" -> GENERAL' in system
    assert "The bot never suggests or creates a poll that was not asked for." in system
    # And a poll is the one pure command that still gets a Q: line.
    assert "pure commands other than CMD_POLL" in system


# ── poll status (2026-10-15) ─────────────────────────────────────────────────
# "quem está a ganhar?" after the bot put up a poll is a question about the
# recorded votes, not a generated answer: the router sends it to CMD_POLL_STATUS
# and the adapter builds the reply from the registry.
def test_poll_status_routes():
    route = router.classify(StubBackend("CMD_POLL_STATUS"), _config(),
                            "quem está a ganhar a votação?")
    assert route.command == router.CMD_POLL_STATUS
    assert route.fallback is False


def test_poll_status_not_confused_with_other_commands():
    """Longest-label-first: CMD_POLL_STATUS must not shadow, or be shadowed by,
    the other command labels."""
    for label, expected in (("CMD_POLL_STATUS", router.CMD_POLL_STATUS),
                            ("CMD_COUNT", router.CMD_COUNT),
                            ("CMD_IMAGE", router.CMD_IMAGE)):
        assert router.classify(StubBackend(label), _config(), "x").command == expected


def test_the_prompt_describes_poll_status():
    backend = StubBackend("CMD_POLL_STATUS")
    router.classify(backend, _config(), "quem falta votar?")
    system = backend.calls[0]["messages"][0]["content"]
    assert "CMD_POLL_STATUS" in system
    # The line that keeps a match question away from the bot's poll.
    assert '"quem está a ganhar o jogo?" -> GENERAL' in system
