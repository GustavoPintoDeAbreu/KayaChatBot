"""GoldenTestRunner passes multi-thread history to the response fn (W2)."""

from src.testing.conversation_tester import GoldenTestRunner


def _runner(response_fn, cases):
    r = GoldenTestRunner(provider=None, response_fn=response_fn, golden_tests_file="/nonexistent")
    r.test_cases = cases
    return r


def test_history_passed_when_fn_accepts_it():
    seen = {}

    def response_fn(question, history=None):
        seen["question"] = question
        seen["history"] = history
        return "ok"

    runner = _runner(response_fn, [])
    runner._get_model_response("E ele?", ["A: falámos do Kobe"])
    assert seen["question"] == "E ele?"
    assert seen["history"] == ["A: falámos do Kobe"]


def test_single_arg_response_fn_still_works():
    calls = []

    def response_fn(question):  # legacy single-arg signature
        calls.append(question)
        return "ok"

    runner = _runner(response_fn, [])
    out = runner._get_model_response("hello", ["A: prior turn"])
    assert out == "ok"
    assert calls == ["hello"]
