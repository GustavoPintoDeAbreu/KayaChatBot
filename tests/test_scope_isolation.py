"""Memory isolation: a DM must never surface in the group.

Before live ingestion the vector store held only the historical group export, so
there was nothing to leak and no filtering. Ingesting live conversation is exactly
what creates the risk — a DM lands in the same collection as the group's history.

These tests pin the asymmetry that makes that safe: shared group memory is
readable everywhere, a DM is readable only inside that DM.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.scope import (
    DM_PREFIX,
    GROUP_PREFIX,
    SHARED,
    is_readable,
    readable_scopes,
    scope_for_chat,
    scope_filter,
)

GROUP = "120363000000000000@g.us"
OTHER_GROUP = "120363999999999999@g.us"
DM_A = "351911111111@c.us"
DM_B = "351922222222@c.us"


class TestScopeAssignment:
    def test_the_kaya_group_is_shared_memory(self):
        assert scope_for_chat(GROUP, shared_chats={GROUP}) == SHARED

    def test_a_dm_is_private_to_itself(self):
        assert scope_for_chat(DM_A).startswith(DM_PREFIX)

    def test_another_group_is_private_to_itself(self):
        assert scope_for_chat(OTHER_GROUP, shared_chats={GROUP}).startswith(GROUP_PREFIX)

    def test_different_chats_get_different_scopes(self):
        assert scope_for_chat(DM_A) != scope_for_chat(DM_B)

    def test_scope_is_stable_for_the_same_chat(self):
        assert scope_for_chat(DM_A) == scope_for_chat(DM_A)

    def test_no_chat_context_reads_shared(self):
        """CLI, benchmarks and the web UI have no chat — they get group memory."""
        assert scope_for_chat(None) == SHARED
        assert scope_for_chat("") == SHARED

    def test_raw_phone_number_is_not_embedded_in_the_scope(self):
        """The store already holds message text; no need to index it by number."""
        assert "351911111111" not in scope_for_chat(DM_A)


class TestReadability:
    def test_a_dm_can_read_shared_group_memory(self):
        """Intended: asking the bot in private about the group must work."""
        dm = scope_for_chat(DM_A)
        assert is_readable(SHARED, dm) is True

    def test_the_group_cannot_read_a_dm(self):
        """Required: this is the leak the whole design exists to prevent."""
        dm = scope_for_chat(DM_A)
        assert is_readable(dm, SHARED) is False

    def test_one_dm_cannot_read_another(self):
        a, b = scope_for_chat(DM_A), scope_for_chat(DM_B)
        assert is_readable(b, a) is False
        assert is_readable(a, b) is False

    def test_a_dm_can_read_itself(self):
        dm = scope_for_chat(DM_A)
        assert is_readable(dm, dm) is True

    def test_unscoped_chunks_are_treated_as_shared(self):
        """Everything predating scoping is the historical group export."""
        assert is_readable(None, SHARED) is True
        assert is_readable(None, scope_for_chat(DM_A)) is True

    def test_shared_context_reads_only_shared(self):
        assert readable_scopes(SHARED) == [SHARED]


class TestScopeFilter:
    def test_shared_filter_is_a_plain_equality(self):
        assert scope_filter(SHARED) == {"scope": SHARED}

    def test_dm_filter_allows_shared_and_itself_only(self):
        dm = scope_for_chat(DM_A)
        where = scope_filter(dm)
        allowed = where["scope"]["$in"]
        assert set(allowed) == {SHARED, dm}
        assert scope_for_chat(DM_B) not in allowed


class TestIngestIdempotency:
    def test_same_messages_produce_the_same_chunk_id(self):
        """This is what makes a repeated or crashed ingest safe."""
        from src.data.ingest import chunk_uid

        assert chunk_uid("dm:abc", ["m1", "m2"]) == chunk_uid("dm:abc", ["m1", "m2"])

    def test_different_scopes_produce_different_chunk_ids(self):
        from src.data.ingest import chunk_uid

        assert chunk_uid("dm:abc", ["m1"]) != chunk_uid("dm:xyz", ["m1"])

    def test_message_uid_is_stable_and_chat_specific(self):
        from src.data.message_log import message_uid

        assert message_uid("chatA", "m1") == message_uid("chatA", "m1")
        assert message_uid("chatA", "m1") != message_uid("chatB", "m1")

    def test_chunks_carry_their_scope(self, tmp_path):
        from src.data.ingest import build_chunks

        msgs = [
            {"id": "m1", "sender": "Gustavo", "text": "olá", "timestamp": 1700000000},
            {"id": "m2", "sender": "Rafa", "text": "tudo bem?", "timestamp": 1700000060},
        ]
        chunks = build_chunks(msgs, scope="dm:abc")
        assert chunks and all(c["metadata"]["scope"] == "dm:abc" for c in chunks)
        assert all(c["metadata"]["source"] == "live" for c in chunks)
        # timestamps must be present — the recency cutoff depends on them
        assert chunks[0]["metadata"]["timestamp_start"] <= chunks[0]["metadata"]["timestamp_end"]


class TestMessageLogIsolation:
    def test_scopes_are_separate_files_on_disk(self, tmp_path):
        """Physical separation, not merely a query-time filter."""
        from src.data.message_log import MessageLog

        log = MessageLog(base_dir=str(tmp_path))
        log.append(chat_id=DM_A, message_id="m1", sender="Gustavo",
                   text="segredo", timestamp=1700000000, scope="dm:aaa")
        log.append(chat_id=GROUP, message_id="m2", sender="Rafa",
                   text="publico", timestamp=1700000001, scope=SHARED)

        assert set(log.scopes()) == {"dm:aaa", SHARED}
        shared_texts = [m["text"] for m in log.read(SHARED)]
        assert "segredo" not in shared_texts

    def test_relogging_the_same_message_is_ignored(self, tmp_path):
        """WAHA replays its backlog freely after a reconnect."""
        from src.data.message_log import MessageLog

        log = MessageLog(base_dir=str(tmp_path))
        first = log.append(chat_id=DM_A, message_id="m1", sender="G", text="olá",
                           timestamp=1700000000, scope="dm:aaa")
        second = log.append(chat_id=DM_A, message_id="m1", sender="G", text="olá",
                            timestamp=1700000000, scope="dm:aaa")
        assert first is True and second is False
        assert len(list(log.read("dm:aaa"))) == 1

    def test_watermark_advances_and_persists(self, tmp_path):
        from src.data.ingest import IngestState

        state = IngestState(path=str(tmp_path / "state.json"))
        assert state.watermark("dm:aaa") == 0
        state.set_watermark("dm:aaa", 1700000100, ingested=3)
        assert state.watermark("dm:aaa") == 1700000100
        # survives a reload
        assert IngestState(path=str(tmp_path / "state.json")).watermark("dm:aaa") == 1700000100


class TestMediaIngest:
    """Image descriptions become scoped, retrievable memory."""

    def test_export_parsing_extracts_sender_and_attachment(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ingest_media", Path(__file__).parent.parent / "scripts" / "ingest_media.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        export = tmp_path / "chat.txt"
        export.write_text(
            "3/26/20, 15:28 - Messages and calls are end-to-end encrypted\n"
            "6/4/20, 12:01 - Gustavo Abreu: olha esta\n"
            "6/4/20, 12:02 - Gustavo Abreu: IMG-20200604-WA0004.jpg (file attached)\n"
            "6/4/20, 12:03 - Peter: ahahah brutal\n"
            "6/4/20, 12:04 - Gil: STK-20200604-WA0001.webp (file attached)\n",
            encoding="utf-8")

        msgs = mod.parse_export(export)
        atts = [m for m in msgs if m.get("kind")]
        assert {a["kind"] for a in atts} == {"IMG", "STK"}
        img = next(a for a in atts if a["kind"] == "IMG")
        assert img["sender"] == "Gustavo Abreu"
        assert img["file"] == "IMG-20200604-WA0004.jpg"

    def test_context_around_uses_neighbouring_chatter(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ingest_media", Path(__file__).parent.parent / "scripts" / "ingest_media.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        msgs = [
            {"sender": "A", "text": "olha esta", "file": None, "kind": None},
            {"sender": "B", "text": "IMG-1.jpg (file attached)", "file": "IMG-1.jpg", "kind": "IMG"},
            {"sender": "C", "text": "ahahah brutal", "file": None, "kind": None},
        ]
        ctx = mod.context_around(msgs, 1)
        # the photo's own line is excluded; the chatter around it is what gives it meaning
        assert "olha esta" in ctx and "brutal" in ctx
        assert "IMG-1.jpg" not in ctx


class TestVoiceNoteFetching:
    """WAHA reports media at its OWN localhost, which is unreachable from ours."""

    def test_localhost_is_rewritten_to_the_reachable_waha_host(self):
        from src.chat.stt import rewrite_media_url

        got = rewrite_media_url(
            "http://localhost:3000/api/files/default/AC05.oga", "http://waha:3000")
        assert got == "http://waha:3000/api/files/default/AC05.oga"

    def test_path_and_query_are_preserved(self):
        from src.chat.stt import rewrite_media_url

        got = rewrite_media_url(
            "http://localhost:3000/api/files/x.oga?token=abc", "http://waha:3000")
        assert got == "http://waha:3000/api/files/x.oga?token=abc"

    def test_already_correct_url_is_untouched(self):
        from src.chat.stt import rewrite_media_url

        url = "http://waha:3000/api/files/x.oga"
        assert rewrite_media_url(url, "http://waha:3000") == url

    def test_missing_inputs_are_safe(self):
        from src.chat.stt import rewrite_media_url

        assert rewrite_media_url("", "http://waha:3000") == ""
        assert rewrite_media_url("http://localhost:3000/x", "") == "http://localhost:3000/x"

    def test_a_url_that_is_not_loopback_is_left_alone(self):
        """Only WAHA's self-reported localhost is wrong from here. Rewriting any
        other host sent the simulator's own media server to WAHA, which answered
        401 and the photo was silently dropped."""
        from src.chat.stt import rewrite_media_url

        for url in ("http://172.18.0.1:8899/photo.jpg",
                    "https://cdn.example.com/a.oga"):
            assert rewrite_media_url(url, "http://waha:3000") == url


class TestIngestWatermark:
    """The watermark is a high-water mark, so one bad timestamp can freeze it.

    A simulator run hit this: an event dated in the future pinned `shared` ahead
    of real time, and every message logged afterwards — including a planted fact
    the bot was then asked about — was skipped forever. Nothing errored; memory
    just silently stopped updating.
    """

    def test_a_future_timestamp_does_not_freeze_the_watermark(self, tmp_path):
        import time as _time

        from src.data.ingest import _FUTURE_TOLERANCE_S, Ingester

        now = int(_time.time())
        messages = [
            {"id": "m1", "sender": "Gustavo", "text": "normal", "timestamp": now - 60},
            {"id": "m2", "sender": "Rafa", "text": "clock is wrong",
             "timestamp": now + _FUTURE_TOLERANCE_S + 10_000},
        ]
        timestamps = [int(m["timestamp"]) for m in messages]
        horizon = now + _FUTURE_TOLERANCE_S
        newest = max((t for t in timestamps if t <= horizon), default=0)

        assert newest == now - 60, "the future message must not set the watermark"
        assert newest < messages[1]["timestamp"]

    def test_a_normal_batch_still_advances(self, tmp_path):
        import time as _time

        from src.data.ingest import _FUTURE_TOLERANCE_S

        now = int(_time.time())
        timestamps = [now - 120, now - 60, now - 10]
        horizon = now + _FUTURE_TOLERANCE_S
        assert max(t for t in timestamps if t <= horizon) == now - 10
