"""Unit tests for src/chat/web_search.py — trigger logic, citations, safe degradation.

No network: the trigger is exercised with a stub retriever, and the formatter +
citation line are tested directly. ``maybe_web_search`` must return an unused
``WebSearchResult`` (never raise) when disabled, keyless, or off-trigger.
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


CFG = {"web_search": {"enabled": True, "trigger_similarity": 0.40, "current_events_max": 0.60, "max_results": 3}}


# ── explicit request always fires (even with a strong RAG match) ────────────────
def test_explicit_request_triggers_regardless_of_score(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.95)
    assert web_search.should_search("podes procurar na net quando sai o gta6?", r, CFG) is True


def test_explicit_english(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.99)
    assert web_search.should_search("can you search the web for the score?", r, CFG) is True


# ── current-events intent fires under the backstop, not when strongly group-anchored ─
def test_current_events_triggers_under_backstop(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.45)  # < current_events_max 0.60
    assert web_search.should_search("quem ganhou o último jogo de Portugal?", r, CFG) is True


def test_current_events_skipped_when_strongly_group(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.80)  # strongly anchored to group memory
    assert web_search.should_search("qual foi o resultado do nosso último jogo de padel?", r, CFG) is False


# ── low-RAG fallback ───────────────────────────────────────────────────────────
def test_lowrag_fallback_triggers(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.12)
    assert web_search.should_search("capital da mongólia?", r, CFG) is True


# ── group questions never search ───────────────────────────────────────────────
def test_member_named_never_searches(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=["gustavo"], score=0.05)
    assert web_search.should_search("o que faz o gustavo?", r, CFG) is False


def test_group_superlative_does_not_search(monkeypatch):
    # no member named, no current-events cue, strong group similarity → no search
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.75)
    assert web_search.should_search("quem é o mais teimoso do grupo?", r, CFG) is False


# ── gating: disabled / keyless ─────────────────────────────────────────────────
def test_no_key_disables(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    r = StubRetriever(persons=[], score=0.1)
    assert web_search.should_search("procura na net as notícias de hoje", r, CFG) is False


def test_disabled_config(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.1)
    assert web_search.should_search("x", r, {"web_search": {"enabled": False}}) is False


# ── maybe_web_search returns an unused result off-trigger, never raises ─────────
def test_maybe_web_search_unused_offtrigger(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.95)  # relevant, no explicit/current cue
    res = web_search.maybe_web_search("o que o grupo costuma fazer?", r, CFG)
    assert res.used is False and res.context == "" and res.sources == []


# ── formatting + citations ─────────────────────────────────────────────────────
def test_format_results():
    out = web_search._format_results(
        [{"title": "t", "url": "https://www.placardefutebol.com/x", "content": "Portugal venceu o Uzbequistão."}]
    )
    assert "Portugal venceu o Uzbequistão." in out
    assert out.startswith("=== Resultados de pesquisa web")


def test_format_results_empty():
    assert web_search._format_results([]) == ""


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
    assert res.used is False and res.citation_line() == ""


# ── tightened current-events regex + group-history guard ───────────────────────
def test_bare_aconteceu_no_longer_current_events():
    assert web_search.is_current_events("qual foi a melhor cena que aconteceu") is False
    assert web_search.is_current_events("o que aconteceu no mundo hoje") is True


def test_group_history_guard_blocks_search(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.05)  # low RAG would otherwise fallback-search
    assert web_search.should_search("conta-me a melhor cena que aconteceu no grupo", r, CFG) is False
    assert web_search.should_search("o que se passou nas conversas do kaya?", r, CFG) is False


def test_explicit_request_overrides_group_history(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    r = StubRetriever(persons=[], score=0.9)
    # explicit "pesquisa" wins even though "do grupo" is present
    assert web_search.should_search("pesquisa na net as notícias do grupo automóvel", r, CFG) is True


# ── exclude_domains is forwarded to the Tavily request body ─────────────────────
def test_exclude_domains_in_request_body(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"answer": "Resumo.", "results": []}

    def fake_post(url, json=None, timeout=None):
        captured["body"] = json
        return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    client = web_search.TavilyClient(api_key="k", exclude_domains=["youtube.com", "genius.com"])
    client.search("preço do iphone")
    assert captured["body"].get("exclude_domains") == ["youtube.com", "genius.com"]


def test_no_exclude_domains_key_when_empty(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"answer": "x", "results": []}

    def fake_post(url, json=None, timeout=None):
        captured["body"] = json
        return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    client = web_search.TavilyClient(api_key="k")  # no exclude_domains
    client.search("q")
    assert "exclude_domains" not in captured["body"]


def test_truncate_clean_no_midword_cut():
    long_text = "Portugal " * 100
    out = web_search._truncate_clean(long_text, 50)
    assert out.endswith("…")
    assert "Portuga…" not in out  # didn't cut mid-word
