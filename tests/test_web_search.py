"""Unit tests for src/chat/web_search.py — trigger logic, citations, safe degradation.

No network: the trigger is exercised with a stub retriever, and the Grok client is
monkeypatched. ``maybe_web_search`` must return an unused ``WebSearchResult`` (never
raise) when disabled, keyless, or off-trigger, and a finished ``answer`` when a search
runs.
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


CFG = {"web_search": {"enabled": True, "trigger_similarity": 0.40, "current_events_max": 0.60},
       "generation": {"xai": {"model": "grok-test"}}}


# ── explicit request always fires (even with a strong RAG match) ────────────────
def test_explicit_request_triggers_regardless_of_score(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    r = StubRetriever(persons=[], score=0.95)
    assert web_search.should_search("podes procurar na net quando sai o gta6?", r, CFG) is True


def test_explicit_english(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    r = StubRetriever(persons=[], score=0.99)
    assert web_search.should_search("can you search the web for the score?", r, CFG) is True


# ── current-events intent fires under the backstop, not when strongly group-anchored ─
def test_current_events_triggers_under_backstop(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    r = StubRetriever(persons=[], score=0.45)  # < current_events_max 0.60
    assert web_search.should_search("quem ganhou o último jogo de Portugal?", r, CFG) is True


def test_current_events_skipped_when_strongly_group(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    r = StubRetriever(persons=[], score=0.80)  # strongly anchored to group memory
    assert web_search.should_search("qual foi o resultado do nosso último jogo de padel?", r, CFG) is False


# ── low-RAG fallback ───────────────────────────────────────────────────────────
def test_lowrag_fallback_triggers(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    r = StubRetriever(persons=[], score=0.12)
    assert web_search.should_search("capital da mongólia?", r, CFG) is True


# ── group questions never search ───────────────────────────────────────────────
def test_member_named_never_searches(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    r = StubRetriever(persons=["gustavo"], score=0.05)
    assert web_search.should_search("o que faz o gustavo?", r, CFG) is False


def test_group_superlative_does_not_search(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    r = StubRetriever(persons=[], score=0.75)
    assert web_search.should_search("quem é o mais teimoso do grupo?", r, CFG) is False


# ── gating: disabled / keyless ─────────────────────────────────────────────────
def test_no_key_disables(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    r = StubRetriever(persons=[], score=0.1)
    assert web_search.should_search("procura na net as notícias de hoje", r, CFG) is False


def test_disabled_config(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    r = StubRetriever(persons=[], score=0.1)
    assert web_search.should_search("x", r, {"web_search": {"enabled": False}}) is False


# ── tightened current-events regex + group-history guard ───────────────────────
def test_bare_aconteceu_no_longer_current_events():
    assert web_search.is_current_events("qual foi a melhor cena que aconteceu") is False
    assert web_search.is_current_events("o que aconteceu no mundo hoje") is True


def test_group_history_guard_blocks_search(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    r = StubRetriever(persons=[], score=0.05)  # low RAG would otherwise fallback-search
    assert web_search.should_search("conta-me a melhor cena que aconteceu no grupo", r, CFG) is False
    assert web_search.should_search("o que se passou nas conversas do kaya?", r, CFG) is False


def test_explicit_request_overrides_group_history(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    r = StubRetriever(persons=[], score=0.9)
    assert web_search.should_search("pesquisa na net as notícias do grupo automóvel", r, CFG) is True


# ── maybe_web_search: off-trigger returns unused; on-trigger returns Grok's answer ─
def test_maybe_web_search_unused_offtrigger(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    r = StubRetriever(persons=[], score=0.95)  # relevant, no explicit/current cue
    res = web_search.maybe_web_search("o que o grupo costuma fazer?", r, CFG)
    assert res.used is False and res.answer == "" and res.sources == []


def test_maybe_web_search_returns_grok_answer(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    web_search._client = None  # reset cached client

    class FakeGrok:
        def __init__(self, *a, **k):
            pass

        def search(self, query):
            return ("Portugal venceu o Uzbequistão por 5-0.",
                    ["https://www.sofascore.com/x", "https://www.espn.com/y"])

    monkeypatch.setattr(web_search, "GrokSearchClient", FakeGrok)
    r = StubRetriever(persons=[], score=0.05)
    res = web_search.maybe_web_search("quem ganhou o último jogo de Portugal?", r, CFG)
    assert res.used is True
    assert "5-0" in res.answer
    assert res.citation_line().startswith(web_search.CITATION_PREFIX)
    assert "sofascore.com" in res.citation_line()
    web_search._client = None


def test_maybe_web_search_never_raises(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    web_search._client = None

    class BoomGrok:
        def __init__(self, *a, **k):
            pass

        def search(self, query):
            raise RuntimeError("xai down")

    monkeypatch.setattr(web_search, "GrokSearchClient", BoomGrok)
    r = StubRetriever(persons=[], score=0.05)
    res = web_search.maybe_web_search("quem ganhou o jogo de hoje?", r, CFG)
    assert res.used is False
    web_search._client = None


# ── citations ──────────────────────────────────────────────────────────────────
def test_citation_line_dedups_domains():
    line = web_search.citation_line([
        "https://www.placardefutebol.com/a",
        "https://www.placardefutebol.com/b",  # same domain → dedup
        "https://record.pt/c",
    ])
    assert line.startswith(web_search.CITATION_PREFIX)
    assert "placardefutebol.com" in line and "record.pt" in line
    assert line.count(",") == 1  # only 2 distinct domains kept


def test_citation_line_empty():
    assert web_search.citation_line([]) == ""
    assert web_search.citation_line([""]) == ""


def test_websearchresult_default_unused():
    res = web_search.WebSearchResult()
    assert res.used is False and res.citation_line() == "" and res.answer == ""


# ── search_for: the planner decides, but privacy still does not ──────────────
class RecordingClient:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return "uma resposta", ["https://oecd.org/x"]


def test_search_for_skips_the_heuristics(monkeypatch):
    """A debate's lookup is already a considered decision; do not re-guess it.

    "real wage growth in Europe since 1970" is not an explicit search request and
    is not a current-events cue, so `should_search` would refuse it.
    """
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    client = RecordingClient()
    monkeypatch.setattr(web_search, "_get_client", lambda cfg: client)

    query = "real wage growth in Europe since 1970"
    assert web_search.should_search(query, StubRetriever(score=0.9), CFG) is False

    result = web_search.search_for(query, StubRetriever(score=0.9), CFG)
    assert result.used is True
    assert client.queries == [query]


def test_search_for_never_sends_a_member_name(monkeypatch):
    """The plan is written by a model that just read the group's own thread."""
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    client = RecordingClient()
    monkeypatch.setattr(web_search, "_get_client", lambda cfg: client)

    result = web_search.search_for(
        "quanto ganha o Pedro", StubRetriever(persons=["pedro"]), CFG)

    assert result.used is False
    assert client.queries == [], "a member's name must never leave the box"


def test_search_for_fails_closed_when_the_guard_raises(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    client = RecordingClient()
    monkeypatch.setattr(web_search, "_get_client", lambda cfg: client)

    class Boom:
        def extract_query_persons(self, query):
            raise RuntimeError("members file gone")

    assert web_search.search_for("inflação 1970", Boom(), CFG).used is False
    assert client.queries == []


def test_search_for_is_silent_without_a_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert web_search.search_for("inflação", StubRetriever(), CFG).used is False


def test_search_for_respects_the_enabled_flag(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    off = {"web_search": {"enabled": False}}
    assert web_search.search_for("inflação", StubRetriever(), off).used is False


def test_search_for_never_raises(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")

    class Exploding:
        def search(self, query):
            raise RuntimeError("xai down")

    monkeypatch.setattr(web_search, "_get_client", lambda cfg: Exploding())
    assert web_search.search_for("inflação", StubRetriever(), CFG).used is False
