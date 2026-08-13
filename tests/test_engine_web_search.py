"""The web-search path: one voice answers the whole message, sources stay apart.

Returning Grok's answer verbatim bypassed the persona, the uncensored preamble
and clean_response. The logged result: "manda-o para o caralho? Já agora quem é
melhor, Cristiano ou Messi?" came back as a comparison plus "A primeira parte da
pergunta não se enquadra em resposta factual baseada na web", with the sources
line glued on — and that whole string was then read aloud.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import engine as engine_module
from src.chat import router
from src.chat.engine import KayaEngine, Reply


class StubBackend:
    """Classifies, then answers. Records every prompt it is given."""

    def __init__(self, label="GENERAL", answer="O Ronaldo, e manda-o tu."):
        self.label = label
        self.answer = answer
        self.prompts = []

    def generate(self, messages, *, max_new_tokens=None, sampling=None):
        self.prompts.append(messages)
        # The router call is the tiny one; everything else is the real answer.
        if max_new_tokens is not None and max_new_tokens <= 16:
            return self.label
        return self.answer


class StubRetriever:
    def extract_query_persons(self, query):
        return []

    def named_members(self, text):
        return []

    def retrieve_all(self, *a, **kw):
        return ""

    def best_similarity(self, *a, **kw):
        return 0.0


def _config(**web_over):
    web = {"enabled": True, "synthesize_locally": True}
    web.update(web_over)
    return {
        "chat": {
            "router": {"enabled": True, "max_new_tokens": 8, "temperature": 0.0,
                       "fallback_mode": "factual"},
            "modes": {
                "banter": {"retrieval": False, "max_new_tokens": 48},
                "mixed": {"retrieval": True, "top_k": 3, "max_new_tokens": 128},
                "general": {"retrieval": False, "max_new_tokens": 256,
                            "system_prompt": "És o bot."},
                "factual": {"retrieval": True, "max_new_tokens": 256,
                            "system_prompt": None},
            },
            "concurrency": {"acquire_timeout": 5},
        },
        "rag": {"enabled": False},
        "inference": {"max_new_tokens": 768, "max_new_tokens_default": 256},
        "web_search": web,
    }


def _engine(config, backend):
    return KayaEngine(None, None, StubRetriever(), config, backend=backend)


@pytest.fixture
def fake_search(monkeypatch):
    """Stand in for the xAI call with a finished answer plus sources."""
    def install(answer="O Benfica ganhou 6-1.", sources=("https://espn.com.br/x",)):
        from src.chat import web_search

        def fake(query, retriever, config, query_embedding=None):
            return web_search.WebSearchResult(
                used=True, answer=answer, sources=list(sources)
            )

        monkeypatch.setattr(web_search, "maybe_web_search", fake)
    return install


class TestCitationIsSeparate:
    def test_citation_is_not_inside_the_reply_text(self, fake_search):
        fake_search()
        backend = StubBackend()
        reply = _engine(_config(), backend).respond("quem ganhou?", "Gustavo", None, "sys")

        assert "Fontes" not in reply.text
        assert reply.citation.startswith("🌐 Fontes:")
        assert "espn.com.br" in reply.citation

    def test_written_form_reassembles_them(self, fake_search):
        fake_search()
        reply = _engine(_config(), StubBackend()).respond("quem ganhou?", "Gustavo", None, "sys")
        assert reply.text_with_citation == f"{reply.text}\n\n{reply.citation}"

    def test_no_search_means_no_citation(self):
        reply = _engine(_config(), StubBackend()).respond("olá", "Gustavo", None, "sys")
        assert reply.citation == ""


class TestLocalSynthesis:
    def test_web_answer_is_injected_as_context_not_returned(self, fake_search):
        fake_search(answer="O Benfica ganhou 6-1 ao Hearts.")
        backend = StubBackend(answer="Ganharam 6-1, foi uma desgraça para o Hearts.")

        reply = _engine(_config(), backend).respond("quem ganhou?", "Gustavo", None, "sys")

        # The local model wrote the reply …
        assert reply.text == "Ganharam 6-1, foi uma desgraça para o Hearts."
        # … and it was given the search result to work from.
        answering_turn = backend.prompts[-1][-1]["content"]
        assert "Resultados de pesquisa web" in answering_turn
        assert "6-1 ao Hearts" in answering_turn

    def test_verbatim_mode_still_available(self, fake_search):
        fake_search(answer="O Benfica ganhou 6-1.")
        config = _config(synthesize_locally=False)

        reply = _engine(config, StubBackend()).respond("quem ganhou?", "Gustavo", None, "sys")

        assert reply.text == "O Benfica ganhou 6-1."
        assert reply.citation.startswith("🌐 Fontes:")


class TestSearchGating:
    def test_general_mode_may_search(self, fake_search):
        fake_search()
        backend = StubBackend(label="GENERAL")
        reply = _engine(_config(), backend).respond("who won?", "Gustavo", None, "sys")
        assert reply.route.mode == router.GENERAL
        assert reply.citation != ""

    def test_banter_never_searches(self, fake_search):
        fake_search()
        backend = StubBackend(label="BANTER")
        reply = _engine(_config(), backend).respond("ahahah", "Gustavo", None, "sys")
        assert reply.route.mode == router.BANTER
        assert reply.citation == ""


def test_generate_reply_keeps_the_citation_for_written_callers(fake_search):
    """Benchmarks and probes take a plain string; they are written surfaces."""
    fake_search()
    text = _engine(_config(), StubBackend()).generate_reply(
        "quem ganhou?", "Gustavo", None, "sys"
    )
    assert "🌐 Fontes:" in text
