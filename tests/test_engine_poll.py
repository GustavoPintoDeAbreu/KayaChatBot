"""The CMD_POLL path in engine.respond: the log read and the draft, nothing else.

A poll is built from the chat's own log, not from retrieval — so these tests
check that a poll turn never touches the retriever, that the draft lands on the
Reply, and that the failure shapes (no chat, disabled, unreadable log) come back
as statuses the adapter can turn into a line.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import polls
from src.chat.engine import KayaEngine

CHAT = "12036300000000@g.us"

FINAL = (
    "PERGUNTA: Onde é que vamos a jantar?\n"
    "OPÇÃO: O Botequim do Zé\n"
    "OPÇÃO: A Tasca da Rita\n"
    "FORA: O Sushi Bar | rejeitado por todos\n"
)


class PollStubBackend:
    """The router call is the tiny one; the next generation is the poll draft."""

    def __init__(self, draft=FINAL):
        self.draft = draft
        self.calls = []

    def generate(self, messages, *, max_new_tokens=None, sampling=None):
        self.calls.append({"max_new_tokens": max_new_tokens,
                           "sampling": sampling,
                           "user": messages[-1]["content"]})
        if max_new_tokens is not None and max_new_tokens <= 16:
            return "CMD_POLL\nQ: sondagem com os restaurantes propostos"
        return self.draft


class StubRetriever:
    def __init__(self):
        self.calls = []

    def extract_query_persons(self, query):
        self.calls.append(("extract_query_persons", query))
        return []

    def named_members(self, text):
        self.calls.append(("named_members", text))
        return []

    def retrieve_all(self, *a, **kw):
        self.calls.append(("retrieve_all",))
        return ""

    def best_similarity(self, *a, **kw):
        return 0.0


def _config(tmp_path, **poll_over):
    poll = {"enabled": True, "lookback_days": 90,
            "store": str(tmp_path / "polls.jsonl")}
    poll.update(poll_over)
    return {
        "chat": {
            "router": {"enabled": True, "max_new_tokens": 8, "temperature": 0.0,
                       "fallback_mode": "factual"},
            "modes": {
                "factual": {"retrieval": True, "max_new_tokens": 256,
                            "system_prompt": None},
            },
            "poll": poll,
            "concurrency": {"acquire_timeout": 5},
        },
        "whatsapp": {"message_log_dir": str(tmp_path / "log")},
        "rag": {"enabled": False},
        "inference": {"max_new_tokens": 768, "max_new_tokens_default": 256},
    }


def _engine(config, backend):
    return KayaEngine(None, None, StubRetriever(), config, backend=backend)


def write_log(tmp_path, records):
    path = tmp_path / "log" / "shared.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _record(i, text, sender="Alice", chat_id=CHAT, scope="shared"):
    return {"id": f"uid-{i}", "chat_id": chat_id, "sender": sender, "text": text,
            "timestamp": 1791900000 + i, "scope": scope}


def _log(tmp_path):
    write_log(tmp_path, [
        _record(0, "vamos ao Botequim do Zé?", sender="Pedro"),
        _record(1, "ou a Tasca da Rita", sender="Alice"),
        _record(2, "o sushi bar já foi, ninguém quer", sender="Alice"),
    ])


class TestPollRoute:
    def test_a_poll_turn_returns_the_draft_not_a_reply(self, tmp_path):
        _log(tmp_path)
        backend = PollStubBackend()
        retriever = StubRetriever()
        engine = KayaEngine(None, None, retriever, _config(tmp_path), backend=backend)

        reply = engine.respond("faz uma sondagem com os sítios", "Pedro", None,
                               "sys", scope="shared", chat_id=CHAT)

        assert reply.route.command == "poll"
        assert reply.text == ""
        assert reply.poll.status == polls.STATUS_OK
        assert reply.poll.options == ["O Botequim do Zé", "A Tasca da Rita"]
        assert reply.poll.messages_read == 3
        assert reply.poll.dropped == [("O Sushi Bar", "rejeitado por todos")]

    def test_a_poll_turn_never_retrieves(self, tmp_path):
        _log(tmp_path)
        retriever = StubRetriever()
        engine = KayaEngine(None, None, retriever, _config(tmp_path),
                            backend=PollStubBackend())

        engine.respond("faz uma sondagem", "Pedro", None, "sys",
                       scope="shared", chat_id=CHAT)

        # named_members runs for telemetry on every turn; the poll must never
        # ask the retriever for context.
        assert not any(call[0] == "retrieve_all" for call in retriever.calls)

    def test_the_draft_prompt_carries_the_query_and_the_conversation(self, tmp_path):
        _log(tmp_path)
        backend = PollStubBackend()
        engine = _engine(_config(tmp_path), backend)

        engine.respond("faz uma sondagem", "Pedro", None, "sys",
                       scope="shared", chat_id=CHAT)

        # The router call, then the poll generation.
        assert len(backend.calls) == 2
        user = backend.calls[1]["user"]
        assert "Tema da sondagem:" in user
        assert "Botequim do Zé" in user
        # The poll runs at its own temperature, not the reply default.
        assert backend.calls[1]["sampling"]["temperature"] == 0.2

    def test_telemetry_carries_the_poll_fields(self, tmp_path):
        _log(tmp_path)
        engine = _engine(_config(tmp_path), PollStubBackend())

        reply = engine.respond("faz uma sondagem", "Pedro", None, "sys",
                               scope="shared", chat_id=CHAT)

        assert reply.telemetry["route_command"] == "poll"
        assert reply.telemetry["poll_status"] == "ok"
        assert reply.telemetry["poll_options"] == 2
        assert reply.telemetry["poll_dropped"] == ["O Sushi Bar"]
        assert reply.telemetry["poll_invented"] == []


class TestPollFailureShapes:
    def test_no_chat_id_is_no_chat(self, tmp_path):
        engine = _engine(_config(tmp_path), PollStubBackend())

        reply = engine.respond("faz uma sondagem", "Pedro", None, "sys",
                               scope="shared", chat_id=None)

        assert reply.poll.status == polls.STATUS_NO_CHAT
        assert reply.text == polls.NO_CHAT_LINE
        assert reply.telemetry["poll_status"] == "no_chat"

    def test_disabled_is_a_status_with_no_generation(self, tmp_path):
        _log(tmp_path)
        backend = PollStubBackend()
        engine = _engine(_config(tmp_path, enabled=False), backend)

        reply = engine.respond("faz uma sondagem", "Pedro", None, "sys",
                               scope="shared", chat_id=CHAT)

        assert reply.poll.status == polls.STATUS_DISABLED
        assert reply.text == ""
        # Only the router ran; the disabled feature never reaches the model.
        assert len(backend.calls) == 1

    def test_an_empty_log_is_no_messages(self, tmp_path):
        write_log(tmp_path, [])
        engine = _engine(_config(tmp_path), PollStubBackend())

        reply = engine.respond("faz uma sondagem", "Pedro", None, "sys",
                               scope="shared", chat_id=CHAT)

        assert reply.poll.status == polls.STATUS_NO_MESSAGES

    def test_a_generation_failure_is_an_error_draft(self, tmp_path):
        _log(tmp_path)

        class Boom(PollStubBackend):
            def generate(self, messages, *, max_new_tokens=None, sampling=None):
                if max_new_tokens is not None and max_new_tokens <= 16:
                    return "CMD_POLL\nQ: sondagem"
                raise RuntimeError("llama down")

        engine = _engine(_config(tmp_path), Boom())

        reply = engine.respond("faz uma sondagem", "Pedro", None, "sys",
                               scope="shared", chat_id=CHAT)

        assert reply.poll.status == polls.STATUS_ERROR
        assert reply.poll.options == []
