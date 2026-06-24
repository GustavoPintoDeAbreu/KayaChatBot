"""Unit tests for src/chat/web_search.py — trigger logic + safe degradation.

No network: the trigger is exercised with a stub retriever, and the formatter is
tested directly. ``maybe_web_search`` must return "" (never raise) when disabled,
keyless, or off-trigger.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import web_search


class StubRetriever:
    def __init__(self, persons=None, score=0.1):
        self._persons = persons or []
        self._score = score

    def extract_query_persons(self, query):
        return self._persons

    def best_similarity(self, query, query_embedding=None):
        return self._score


CFG = {"web_search": {"enabled": True, "trigger_similarity": 0.40, "max_results": 3}}


def test_should_search_true_for_offtopic(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.12)  # no member, low RAG score
    assert web_search.should_search("quem ganhou a liga dos campeões?", r, CFG) is True


def test_should_search_false_when_member_named(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=["gustavo"], score=0.05)  # group question
    assert web_search.should_search("o que faz o gustavo?", r, CFG) is False


def test_should_search_false_when_relevant(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.72)  # RAG has a strong match
    assert web_search.should_search("o que o grupo costuma fazer?", r, CFG) is False


def test_should_search_false_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    r = StubRetriever(persons=[], score=0.1)
    assert web_search.should_search("capital da mongólia?", r, CFG) is False


def test_should_search_false_when_disabled(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.1)
    assert web_search.should_search("x", r, {"web_search": {"enabled": False}}) is False


def test_maybe_web_search_returns_empty_offtrigger(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.9)  # relevant → no search
    assert web_search.maybe_web_search("q", r, CFG) == ""


def test_format_results():
    out = web_search._format_results(
        "q", [{"title": "t", "url": "http://x", "content": "Lisboa é a capital."}]
    )
    assert "Lisboa é a capital." in out
    assert "http://x" in out
    assert out.startswith("=== Resultados de pesquisa web")


def test_format_results_empty():
    assert web_search._format_results("q", []) == ""
