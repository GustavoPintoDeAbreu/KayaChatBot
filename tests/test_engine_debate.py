"""Arguing when asked, with sources it actually has.

On 2026-09-05 Frederico asked the bot to referee a long argument between Pedro
and Bernardo — "dá a tua opinião para saber quem tem razão, sê analítico e se
necessário faz a tua própria pesquisa de factos" — and the interaction log
records the turn as `route_mode: roast`, `retrieved_chars: 9962`,
`web_search_used: false`. It mocked both men and researched nothing.

The argument they were having was itself about fabricated sources: Pedro had
gone through Bernardo's AI-generated references and found they did not say what
his numbers claimed ("Porque me mandaste ai slop a pensar que era factos"). So a
bot that argues in this group has to be held to the bar they are enforcing on
each other: cite what you have, or say you have nothing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import router
from src.chat.engine import KayaEngine

MEMBERS = ["Gil", "Rafa", "Pedro", "Gustavo", "Bernardo"]


class ScriptedBackend:
    """Router label first, then a plan, then the answer.

    Generation calls are told apart by their token budget the way the real ones
    are: the router asks for a handful, the plan for a few hundred.
    """

    def __init__(self, label="DEBATE", plan="NEED: none", answer="um argumento"):
        self.label = label
        self.plan = plan
        self.answer = answer
        self.answer_calls = []
        self.plan_calls = []

    def generate(self, messages, *, max_new_tokens=None, sampling=None):
        if max_new_tokens is not None and max_new_tokens <= 16:
            return self.label
        if max_new_tokens is not None and max_new_tokens == 300:
            self.plan_calls.append(messages)
            return self.plan
        self.answer_calls.append({"messages": messages,
                                  "max_new_tokens": max_new_tokens})
        return self.answer


class StubRetriever:
    def __init__(self, docs=None):
        self.docs = docs or []
        self.document_queries = []

    def extract_query_persons(self, query):
        return [m.lower() for m in MEMBERS if m.lower() in (query or "").lower()]

    def named_members(self, text):
        return [m for m in MEMBERS if m.lower() in (text or "").lower()]

    def retrieve_all(self, *a, **kw):
        return "contexto do grupo"

    def retrieve_documents(self, query, top_k=None, scope=None, **kw):
        self.document_queries.append(query)
        return list(self.docs)


DOC = {
    "doc_id": "abc", "filename": "labour.pdf", "sender": "Bernardo",
    "page_start": 51, "page_end": 51, "page_count": 300,
    "text": "Real wages grew about 55% between 1970 and 2025.",
    "similarity_score": 0.8,
}


def make_engine(backend, retriever=None, **over):
    config = {
        "chat": {
            "router": {"enabled": True, "max_new_tokens": 8, "fallback_mode": "factual"},
            "modes": {
                "debate": {"retrieval": True, "max_new_tokens": 400,
                           "system_prompt": "És o bot. Argumenta.",
                           "mode_hint": "Argumenta a sério."},
                "roast": {"retrieval": True, "max_new_tokens": 120},
                "factual": {"retrieval": True, "max_new_tokens": 256},
            },
            "debate": {"enabled": True, "max_lookups": 3,
                       "plan_max_new_tokens": 300, "temperature": 0.3},
            "concurrency": {"acquire_timeout": 5},
            **over,
        },
        "rag": {"enabled": True},
        "inference": {"max_new_tokens": 768, "max_new_tokens_default": 256,
                      "temperature": 0.8, "no_repeat_last_replies": 4},
        "web_search": {"enabled": False},
        "documents": {"enabled": True},
    }
    return KayaEngine(model=None, tokenizer=None,
                      retriever=retriever or StubRetriever(),
                      config=config, backend=backend)


def user_turn(backend, index=0):
    return backend.answer_calls[index]["messages"][-1]["content"]


# ── the evidence gate ────────────────────────────────────────────────────────
def test_a_plan_saying_none_searches_nothing():
    """Pure-logic arguments must not pay for a web lookup."""
    backend = ScriptedBackend(plan="- a habitação é um direito\nNEED: none")
    reply = make_engine(backend).respond(
        "defende que a habitação pública é a solução", "Rafa", [], "sys")

    assert reply.route.mode == router.DEBATE
    assert reply.telemetry["debate_web_lookups"] == 0


def test_the_plan_decides_what_to_look_up():
    backend = ScriptedBackend(
        plan="- comparar salários\nNEED: real wage growth Europe since 1970\n"
             "NEED: food expenditure share Europe 1970 2025")
    engine = make_engine(backend)
    calls = []
    engine._gather_evidence = lambda queries, scope, web: (
        calls.append((list(queries), list(web))) or ([], [], []))

    engine.respond("quem tem razão sobre o custo da comida?", "Frederico", [], "sys")

    _, web_queries = calls[0]
    assert web_queries == ["real wage growth Europe since 1970",
                           "food expenditure share Europe 1970 2025"]


def test_lookups_are_capped():
    backend = ScriptedBackend(
        plan="\n".join(f"NEED: pergunta {i}" for i in range(9)))
    engine = make_engine(backend)
    seen = []
    engine._gather_evidence = lambda queries, scope, web: (
        seen.append(list(web)) or ([], [], []))

    engine.respond("quem tem razão?", "Frederico", [], "sys")

    assert len(seen[0]) == 3, "max_lookups must bound the work a debate can do"


def test_documents_are_searched_even_when_no_lookup_is_requested():
    """An argument about a shared paper must find the paper."""
    retriever = StubRetriever(docs=[DOC])
    backend = ScriptedBackend(plan="NEED: none")

    make_engine(backend, retriever).respond(
        "quem tem razão sobre os salários reais?", "Frederico", [], "sys")

    assert retriever.document_queries, "documents were never searched"


def test_a_failed_plan_still_answers():
    class Boom(ScriptedBackend):
        def generate(self, messages, *, max_new_tokens=None, sampling=None):
            if max_new_tokens == 300:
                raise RuntimeError("planner down")
            return super().generate(messages, max_new_tokens=max_new_tokens,
                                    sampling=sampling)

    reply = make_engine(Boom()).respond("debate me", "Rafa", [], "sys")
    assert reply.text, "a failed plan must not cost the reply"


# ── advocate vs arbitrate ────────────────────────────────────────────────────
def test_an_assigned_side_is_told_to_commit():
    backend = ScriptedBackend()
    make_engine(backend).respond(
        "I'll defend communism u capitalism!", "Rafa", [], "sys")

    turn = user_turn(backend)
    assert "Defende essa posição" in turn
    assert "arbitrar" not in turn


def test_an_unassigned_debate_arbitrates():
    backend = ScriptedBackend()
    make_engine(backend).respond(
        "vê lá quem tem razão entre o Pedro e o Bana", "Frederico", [], "sys")

    turn = user_turn(backend)
    assert "Estás a arbitrar" in turn
    assert "não estás a escolher uma equipa" in turn


# ── cite or concede ──────────────────────────────────────────────────────────
def test_sources_are_offered_with_their_pages():
    backend = ScriptedBackend(plan="NEED: none")
    make_engine(backend, StubRetriever(docs=[DOC])).respond(
        "quem tem razão sobre os salários?", "Frederico", [], "sys")

    turn = user_turn(backend)
    assert "[D1] labour.pdf, p. 51" in turn
    assert "Fontes disponíveis" in turn


def test_an_invented_citation_is_stripped_from_the_reply():
    backend = ScriptedBackend(
        plan="NEED: none",
        answer="Os salários subiram 55% [D1], e a comida 100% [D7].")

    reply = make_engine(backend, StubRetriever(docs=[DOC])).respond(
        "quem tem razão?", "Frederico", [], "sys")

    assert "[D1]" in reply.text, "a real citation must survive"
    assert "[D7]" not in reply.text, "an invented citation must not"
    assert reply.telemetry["citations_stripped"] == ["[D7]"]


def test_an_invented_page_is_redacted():
    backend = ScriptedBackend(
        plan="NEED: none",
        answer="Como diz a página 288, os salários caíram.")

    reply = make_engine(backend, StubRetriever(docs=[DOC])).respond(
        "quem tem razão?", "Frederico", [], "sys")

    assert "288" not in reply.text
    assert "parte do documento" in reply.text


def test_with_no_sources_the_model_is_told_it_has_none():
    backend = ScriptedBackend(plan="NEED: none")
    make_engine(backend).respond("debate me sobre ética", "Rafa", [], "sys")

    assert "Não tens nenhuma fonte" in user_turn(backend)


def test_the_citation_line_lists_only_what_was_cited():
    backend = ScriptedBackend(plan="NEED: none",
                              answer="Subiram 55% [D1].")
    reply = make_engine(backend, StubRetriever(docs=[DOC])).respond(
        "quem tem razão?", "Frederico", [], "sys")

    assert "labour.pdf p. 51" in reply.citation


def test_an_uncited_source_is_not_advertised():
    """Listing everything retrieved is how a bot looks like it read 300 pages."""
    backend = ScriptedBackend(plan="NEED: none", answer="Acho que sim, sem fontes.")
    reply = make_engine(backend, StubRetriever(docs=[DOC])).respond(
        "quem tem razão?", "Frederico", [], "sys")

    assert reply.citation == ""


# ── the mode's shape ─────────────────────────────────────────────────────────
def test_a_debate_does_not_get_the_member_profiles():
    """The 2026-09-05 misroute handed a fact-check 9,962 chars of profiles."""
    backend = ScriptedBackend()
    engine = make_engine(backend)
    engine.system_prompt_factory = lambda **kw: "PERFIS DE TODOS OS MEMBROS"

    engine.respond("quem tem razão?", "Frederico", [], "sys")

    system = backend.answer_calls[0]["messages"][0]["content"]
    assert "PERFIS DE TODOS OS MEMBROS" not in system


def test_a_debate_is_logged_as_having_reasoned():
    backend = ScriptedBackend(plan="NEED: none")
    reply = make_engine(backend).respond("debate me", "Rafa", [], "sys")

    assert reply.telemetry["reasoning_used"] is True
    assert reply.telemetry["route_mode"] == "debate"


def test_disabling_debate_skips_the_plan_entirely():
    backend = ScriptedBackend()
    make_engine(backend, debate={"enabled": False}).respond(
        "debate me", "Rafa", [], "sys")

    assert backend.plan_calls == []
