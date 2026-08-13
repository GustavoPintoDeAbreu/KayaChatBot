"""The bot must not answer two different messages with the same sentence.

From the first full group session: "So attack" got "Escolhe um alvo e diz quem
eu tenho de partir primeiro." and, one turn later, so did "Godamn".
``repetition_penalty`` and ``no_repeat_ngram_size`` were both on — they act only
within a single generation and cannot see the previous turn at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.engine import KayaEngine

REPEAT = "Escolhe um alvo e diz quem eu tenho de partir primeiro."
FRESH = "Diz lá quem queres ver destruído hoje."


class ScriptedBackend:
    """Answers with a queued list, so a retry can be made to differ (or not)."""

    def __init__(self, answers, label="BANTER"):
        self.answers = list(answers)
        self.label = label
        self.answer_calls = []

    def generate(self, messages, *, max_new_tokens=None, sampling=None):
        if max_new_tokens is not None and max_new_tokens <= 16:
            return self.label            # the router's own tiny call
        self.answer_calls.append({"messages": messages, "sampling": sampling})
        return self.answers.pop(0) if self.answers else self.answers_exhausted()

    def answers_exhausted(self):
        raise AssertionError("generated more times than the test scripted")


class StubRetriever:
    def extract_query_persons(self, query):
        return []

    def named_members(self, text):
        return []

    def retrieve_all(self, *a, **kw):
        return ""


def make_engine(backend):
    config = {
        "chat": {
            "router": {"enabled": True, "max_new_tokens": 8, "fallback_mode": "factual"},
            "modes": {"banter": {"retrieval": False, "max_new_tokens": 48}},
            "concurrency": {"acquire_timeout": 5},
        },
        "rag": {"enabled": False},
        "inference": {"max_new_tokens": 768, "max_new_tokens_default": 256,
                      "temperature": 0.8, "no_repeat_last_replies": 4},
        "web_search": {"enabled": False},
    }
    return KayaEngine(model=None, tokenizer=None, retriever=StubRetriever(),
                      config=config, backend=backend)


HISTORY = ["Pedro: So attack", f"Kaya Bot: {REPEAT}"]


def test_a_repeated_reply_is_regenerated():
    backend = ScriptedBackend([REPEAT, FRESH])

    reply = make_engine(backend).respond("Godamn", "Gustavo", HISTORY, "sys")

    assert reply.text == FRESH
    assert len(backend.answer_calls) == 2


def test_the_retry_is_hotter_than_the_first_try():
    backend = ScriptedBackend([REPEAT, FRESH])

    make_engine(backend).respond("Godamn", "Gustavo", HISTORY, "sys")

    first, retry = backend.answer_calls
    assert retry["sampling"]["temperature"] > first["sampling"]["temperature"]


def test_only_one_retry_is_ever_spent():
    """This runs inside the GPU lock, and whatsapp_server DROPS a message on a
    contended lock rather than queueing it — a second retry is paid for by
    somebody else's reply going missing."""
    backend = ScriptedBackend([REPEAT, REPEAT])

    reply = make_engine(backend).respond("Godamn", "Gustavo", HISTORY, "sys")

    assert len(backend.answer_calls) == 2
    assert reply.text == REPEAT   # kept: a second duplicate is no improvement


def test_a_fresh_reply_costs_no_extra_generation():
    backend = ScriptedBackend([FRESH])

    reply = make_engine(backend).respond("Godamn", "Gustavo", HISTORY, "sys")

    assert reply.text == FRESH
    assert len(backend.answer_calls) == 1


def test_the_model_is_shown_what_it_already_said():
    backend = ScriptedBackend([FRESH])

    make_engine(backend).respond("Godamn", "Gustavo", HISTORY, "sys")

    user_turn = backend.answer_calls[0]["messages"][-1]["content"]
    assert "não repitas" in user_turn.lower()
    assert "Escolhe um alvo" in user_turn


def test_an_empty_history_adds_no_hint():
    backend = ScriptedBackend([FRESH])

    make_engine(backend).respond("olá", "Gustavo", [], "sys")

    assert "não repitas" not in backend.answer_calls[0]["messages"][-1]["content"].lower()
