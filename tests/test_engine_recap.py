"""A recap turn reads the log for the topic and nothing else.

No ordinary top-k retrieval on top of the transcript (it would bring back the
chatter the topic filter removed), no rolling summary (model-written, PR #95), no
web search (the topic is group data), and room for three parts.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import recap, router
from src.chat.engine import KayaEngine
from src.data.message_log import MessageLog

BASE = 1797000000
GROUP = "120363000000000001@g.us"


class Backend:
    def __init__(self, label="CMD_RECAP\nQ: almoço de natal", answer="*Decidido*\nnada decidido ainda"):
        self.label = label
        self.answer = answer
        self.answer_calls = []

    def generate(self, messages, *, max_new_tokens=None, sampling=None):
        if max_new_tokens is not None and max_new_tokens <= 16:
            return self.label
        self.answer_calls.append({"messages": messages, "max_new_tokens": max_new_tokens})
        return self.answer


class Retriever:
    def __init__(self):
        self.retrieve_all_calls = []
        self.retrieve_calls = []

    def named_members(self, text):
        return []

    def extract_query_persons(self, query):
        return []

    def retrieve_all(self, *args, **kwargs):
        self.retrieve_all_calls.append((args, kwargs))
        return "contexto do grupo"

    def retrieve(self, query, top_k=None, scope=None, **kwargs):
        self.retrieve_calls.append({"query": query, "scope": scope})
        return []


@pytest.fixture(autouse=True)
def _no_resolver_no_web(monkeypatch):
    from src.chat import tally, web_search

    monkeypatch.setattr(tally, "load_resolver", lambda config: None)

    def no_web(*args, **kwargs):
        raise AssertionError("web search on a recap")

    monkeypatch.setattr(web_search, "maybe_web_search", no_web)


def make_engine(tmp_path, backend, retriever, **recap_over):
    log = MessageLog(str(tmp_path / "live"))
    log.append(chat_id=GROUP, message_id="m1", sender="Rafa", text="almoço de natal no Ramiro?",
               timestamp=BASE, scope="shared")
    log.append(chat_id=GROUP, message_id="m2", sender="Gil", text="e o Zé da Mouraria?",
               timestamp=BASE + 60, scope="shared")
    config = {
        "chat": {
            "router": {"enabled": True, "max_new_tokens": 8, "fallback_mode": "factual"},
            "modes": {"factual": {"retrieval": True, "max_new_tokens": 256, "brevity_hint": "BREVE"}},
            "concurrency": {"acquire_timeout": 5},
            "recap": {"common_floor": 5, **recap_over},
        },
        "rag": {"enabled": True},
        "inference": {"max_new_tokens": 768, "max_new_tokens_default": 256, "temperature": 0.8},
        "web_search": {"enabled": False},
        "whatsapp": {"message_log_dir": str(tmp_path / "live")},
        "power": {"timezone": "Europe/Lisbon"},
    }
    return KayaEngine(model=None, tokenizer=None, retriever=retriever, config=config, backend=backend)


def user_turn(backend):
    return backend.answer_calls[0]["messages"][-1]["content"]


def test_a_recap_routes_reads_the_log_and_answers(tmp_path):
    backend = Backend()
    reply = make_engine(tmp_path, backend, Retriever()).respond(
        "o que é que decidimos?", "Gustavo", [], "sys")
    assert reply.route.command == router.CMD_RECAP
    assert "almoço de natal no Ramiro?" in user_turn(backend)
    assert reply.text == "*Decidido*\nnada decidido ainda"


def test_a_recap_does_no_ordinary_retrieval_and_no_web_search(tmp_path):
    retriever = Retriever()
    make_engine(tmp_path, Backend(), retriever).respond("o que é que decidimos?", "Gustavo", [], "sys")
    assert retriever.retrieve_all_calls == []
    assert [call["query"] for call in retriever.retrieve_calls] == ["almoço de natal"]


def test_a_recap_gets_no_summary_and_no_brevity_hint(tmp_path):
    backend = Backend()
    make_engine(tmp_path, backend, Retriever()).respond(
        "o que é que decidimos?", "Gustavo", [], "sys", summary="RESUMO-SECRETO")
    assert "RESUMO-SECRETO" not in user_turn(backend) and "BREVE" not in user_turn(backend)


def test_a_recap_has_room_to_answer(tmp_path):
    backend = Backend()
    make_engine(tmp_path, backend, Retriever()).respond("o que é que decidimos?", "Gustavo", [], "sys")
    assert backend.answer_calls[0]["max_new_tokens"] == 450


def test_the_scope_reaches_the_recap(tmp_path):
    retriever = Retriever()
    make_engine(tmp_path, Backend(), retriever).respond(
        "o que é que decidimos?", "Gustavo", [], "sys", scope="dm:abc")
    assert retriever.retrieve_calls[0]["scope"] == "dm:abc"


def test_the_speakers_own_line_survives_cleaning(tmp_path):
    backend = Backend(answer="*Quem disse o quê*\nGil: Ramiro\nRafa: Zé")
    reply = make_engine(tmp_path, backend, Retriever()).respond(
        "o que é que decidimos?", "Gil", [], "sys")
    assert "- Gil: Ramiro" in reply.text and "Rafa: Zé" in reply.text


def test_recap_telemetry_is_recorded(tmp_path):
    reply = make_engine(tmp_path, Backend(), Retriever()).respond(
        "o que é que decidimos?", "Gustavo", [], "sys")
    assert reply.telemetry["recap_topic"] == "almoço de natal"
    assert "recap_rows" in reply.telemetry and "recap_keywords" in reply.telemetry


def test_a_disabled_recap_answers_as_factual(tmp_path, monkeypatch):
    from src.chat import web_search

    monkeypatch.setattr(web_search, "maybe_web_search",
                        lambda *args, **kwargs: type("Web", (), {"used": False, "answer": ""})())
    backend = Backend()
    retriever = Retriever()
    reply = make_engine(tmp_path, backend, retriever, enabled=False).respond(
        "o que é que decidimos?", "Gustavo", [], "sys")
    assert reply.route.command is None
    assert retriever.retrieve_all_calls
    assert "*Decidido*" not in user_turn(backend)


def test_a_recap_with_no_topic_asks_for_one(tmp_path):
    backend = Backend(label="CMD_RECAP")
    make_engine(tmp_path, backend, Retriever()).respond("faz um resumo", "Gustavo", [], "sys")
    assert recap.NO_TOPIC_INSTRUCTION in user_turn(backend)
