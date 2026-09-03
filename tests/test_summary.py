"""The rolling per-chat summary.

Two properties are worth more than the prose it produces: it must never delay a
reply, and it must never lose an update it decided to skip. The bridge DROPS an
inbound message when the GPU lock is contended, so a summary that waited on the
lock would cost somebody their turn.
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.gpu_lock import GpuBusyError
from src.chat.summary import ChatSummaryStore, SummaryWriter


class FakeBackend:
    """Records what it was asked to summarise; optionally refuses."""

    def __init__(self, reply="Resumo curto.", raises=None):
        self.reply = reply
        self.raises = raises
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        if self.raises:
            raise self.raises
        return self.reply


def _config(tmp_path, **summary_over):
    summary = {"enabled": True, "every_lines": 5, "max_words": 150,
               "max_new_tokens": 220, "lock_timeout_seconds": 1}
    summary.update(summary_over)
    return {
        "chat": {"summary": summary,
                 "concurrency": {"max_concurrent": 1, "acquire_timeout": 1}},
        "whatsapp": {"summaries_dir": str(tmp_path / "summaries")},
    }


def _writer(tmp_path, backend=None, **over):
    config = _config(tmp_path, **over)
    store = ChatSummaryStore(config["whatsapp"]["summaries_dir"])
    return SummaryWriter(config, backend or FakeBackend(), store=store)


def _settle(writer, chat_id="c1", timeout=5.0):
    """Wait for the worker to finish, without asserting on timing."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if writer.store.summary_for(chat_id):
            return True
        time.sleep(0.02)
    return False


def _settle_again(writer, chat_id, timeout=5.0):
    """Wait for a queued update to drain when a summary already exists."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with writer._guard:
            pending = chat_id in writer._pending
        if not pending:
            return True
        time.sleep(0.02)
    return False


def _lines(n, start=0):
    return [f"Alguém: linha {i}" for i in range(start, start + n)]


class TestTrigger:
    def test_below_threshold_does_nothing(self, tmp_path):
        writer = _writer(tmp_path)
        assert writer.maybe_update("c1", _lines(4)) is False

    def test_threshold_queues_an_update(self, tmp_path):
        writer = _writer(tmp_path)
        assert writer.maybe_update("c1", _lines(5)) is True
        assert _settle(writer)

    def test_a_chat_already_queued_is_not_queued_twice(self, tmp_path):
        writer = _writer(tmp_path)
        assert writer.maybe_update("c1", _lines(10)) is True
        assert writer.maybe_update("c1", _lines(10)) is False

    def test_does_not_retrigger_until_more_lines_arrive(self, tmp_path):
        writer = _writer(tmp_path)
        writer.maybe_update("c1", _lines(5))
        assert _settle(writer)
        assert writer.maybe_update("c1", _lines(5)) is False
        assert writer.maybe_update("c1", _lines(10)) is True

    def test_disabled_never_triggers(self, tmp_path):
        writer = _writer(tmp_path, enabled=False)
        assert writer.maybe_update("c1", _lines(50)) is False

    def test_empty_history_is_safe(self, tmp_path):
        writer = _writer(tmp_path)
        assert writer.maybe_update("c1", []) is False
        assert writer.maybe_update("", _lines(50)) is False


class TestASaturatedWindowStillTriggers:
    """The bug that killed the live summary from 2026-08-13 to 2026-09-04.

    The session window is capped, so len(history) stops growing while the
    conversation does not. The old trigger compared lines_seen against
    len(history) and ratcheted lines_seen up to meet it, so once the two got
    within every_lines of each other the condition was unsatisfiable FOREVER —
    the group sat at lines_seen=90 against a 100-line window for three weeks and
    ~1,500 messages, and nothing fired.
    """

    def test_a_capped_window_keeps_firing_as_the_chat_moves_on(self, tmp_path):
        writer = _writer(tmp_path)
        window = 100
        history = _lines(window)
        assert writer.maybe_update("c1", history) is True
        assert _settle(writer)

        # 300 more messages arrive. The window never grows past 100; it slides.
        fired = 0
        for batch in range(1, 31):
            history = (history + _lines(10, start=window + (batch - 1) * 10))[-window:]
            assert len(history) == window
            if writer.maybe_update("c1", history):
                fired += 1
                assert _settle_again(writer, "c1")
        assert fired >= 8, f"only {fired} updates across 300 messages"

    def test_a_state_file_left_stuck_by_the_old_counter_heals(self, tmp_path):
        """The three live files had lines_seen and no marker. They must not stay
        stuck once the code is fixed — no manual repair should be needed."""
        writer = _writer(tmp_path)
        writer.store.save("c1", "resumo antigo", _lines(90))
        path = writer.store._path("c1")
        import json
        state = json.loads(path.read_text(encoding="utf-8"))
        del state["marker"]           # exactly what was on disk in prod
        state["lines_seen"] = 90
        path.write_text(json.dumps(state), encoding="utf-8")

        assert writer.maybe_update("c1", _lines(100, start=500)) is True

    def test_a_quiet_chat_still_does_not_retrigger(self, tmp_path):
        """The heal must not become "always fire": an unchanged window is not
        new material."""
        writer = _writer(tmp_path)
        history = _lines(40)
        assert writer.maybe_update("c1", history) is True
        assert _settle(writer)
        assert writer.maybe_update("c1", history) is False
        assert writer.maybe_update("c1", history + _lines(3, start=40)) is False


class TestGeneration:
    def test_the_summary_is_stored_with_its_position(self, tmp_path):
        writer = _writer(tmp_path, FakeBackend("O Manel comprou um Golf."))
        writer.maybe_update("c1", _lines(8))
        assert _settle(writer)
        assert writer.store.summary_for("c1") == "O Manel comprou um Golf."
        assert writer.store.load("c1")["lines_seen"] == 8

    def test_only_the_new_lines_are_sent(self, tmp_path):
        """Cost must not grow with the age of a conversation."""
        backend = FakeBackend()
        writer = _writer(tmp_path, backend)
        writer.maybe_update("c1", _lines(5))
        assert _settle(writer)

        writer.maybe_update("c1", _lines(12))
        deadline = time.time() + 5
        while time.time() < deadline and len(backend.calls) < 2:
            time.sleep(0.02)

        second = backend.calls[1][-1]["content"]
        assert "linha 11" in second       # the new lines are there
        assert "linha 0" not in second    # the old ones are not re-sent
        assert "Resumo até agora" in second  # the previous summary is revised

    def test_an_empty_generation_leaves_the_previous_summary(self, tmp_path):
        writer = _writer(tmp_path, FakeBackend("Primeiro resumo."))
        writer.maybe_update("c1", _lines(5))
        assert _settle(writer)

        writer.backend = FakeBackend("   ")
        writer.maybe_update("c1", _lines(20))
        time.sleep(0.4)
        assert writer.store.summary_for("c1") == "Primeiro resumo."

    def test_a_backend_failure_is_swallowed(self, tmp_path):
        writer = _writer(tmp_path, FakeBackend(raises=RuntimeError("model down")))
        writer.maybe_update("c1", _lines(5))
        time.sleep(0.4)
        assert writer.store.summary_for("c1") == ""


class TestNeverBlocksAReply:
    def test_a_busy_gpu_skips_without_losing_the_update(self, tmp_path):
        """The load-bearing property: skip, don't wait, and retry later."""
        backend = FakeBackend()
        writer = _writer(tmp_path, backend)

        def busy(*args, **kwargs):
            raise GpuBusyError("held by a reply")

        # summary.py imports gpu_section inside the worker, so patching the
        # module attribute is what the running code will actually resolve.
        import src.chat.gpu_lock as gpu_lock

        real_section = gpu_lock.gpu_section
        gpu_lock.gpu_section = busy
        try:
            writer.maybe_update("c1", _lines(5))
            time.sleep(0.4)
            # Nothing written, nothing generated …
            assert writer.store.summary_for("c1") == ""
            assert backend.calls == []
            # … and the position was NOT advanced, so the next turn retries.
            assert writer.store.load("c1")["lines_seen"] == 0
        finally:
            gpu_lock.gpu_section = real_section

        assert writer.maybe_update("c1", _lines(5)) is True
        assert _settle(writer)


class TestStore:
    def test_a_missing_file_reads_as_empty(self, tmp_path):
        store = ChatSummaryStore(str(tmp_path / "s"))
        assert store.summary_for("nobody") == ""
        assert store.load("nobody")["lines_seen"] == 0

    def test_a_corrupt_file_does_not_raise(self, tmp_path):
        store = ChatSummaryStore(str(tmp_path / "s"))
        store.save("c1", "fine", ["a", "b", "c"])
        store._path("c1").write_text("{not json", encoding="utf-8")
        assert store.summary_for("c1") == ""

    def test_chats_are_isolated_from_each_other(self, tmp_path):
        store = ChatSummaryStore(str(tmp_path / "s"))
        store.save("dm:alice", "segredo da Alice", ["a"] * 5)
        store.save("group:kaya", "planos do grupo", ["b"] * 5)
        assert store.summary_for("dm:alice") == "segredo da Alice"
        assert store.summary_for("group:kaya") == "planos do grupo"
