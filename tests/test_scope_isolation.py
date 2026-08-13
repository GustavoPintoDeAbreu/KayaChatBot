"""Memory isolation: a DM must never surface in the group.

Before live ingestion the vector store held only the historical group export, so
there was nothing to leak and no filtering. Ingesting live conversation is exactly
what creates the risk — a DM lands in the same collection as the group's history.

These tests pin the asymmetry that makes that safe: shared group memory is
readable everywhere, a DM is readable only inside that DM.
"""
import json
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
        chunks, _consumed = build_chunks(msgs, scope="dm:abc")
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


class _FakeCollection:
    """Captures upserts so a test can look at what was actually written."""

    def __init__(self):
        self.ids = []
        self.metadatas = []

    def upsert(self, ids, documents, metadatas, embeddings):
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)


class _FakeEncoder:
    """Deterministic vectors — this suite is about chunking, not embeddings."""

    def encode(self, texts, **kwargs):
        import numpy as np

        return np.zeros((len(texts), 8), dtype="float32")


def _fake_ingester(tmp_path, settle_minutes=0):
    """A real Ingester with the GPU and the vector store stubbed out."""
    from src.data.ingest import Ingester

    config = {
        "rag": {"db_path": str(tmp_path / "db")},
        "chat": {"concurrency": {"max_concurrent": 1, "acquire_timeout": 5}},
        "whatsapp": {
            "message_log_dir": str(tmp_path / "log"),
            "ingest_state_file": str(tmp_path / "state.json"),
            "ingest": {"settle_minutes": settle_minutes},
        },
    }
    collection = _FakeCollection()
    return Ingester(config, encoder=_FakeEncoder(), collection=collection), collection


def _write_log(ingester, scope, rows):
    """Append ``(id, text, timestamp)`` rows to the scope's message log."""
    for message_id, text, ts in rows:
        ingester.log.append(chat_id="chat", message_id=message_id, sender="Alguém",
                            text=text, timestamp=ts, scope=scope)


class TestIngestWatermark:
    """The watermark is a high-water mark, so one bad timestamp can freeze it.

    A simulator run hit this: an event dated in the future pinned `shared` ahead
    of real time, and every message logged afterwards — including a planted fact
    the bot was then asked about — was skipped forever. Nothing errored; memory
    just silently stopped updating.
    """

    def test_a_future_timestamp_does_not_freeze_the_watermark(self, tmp_path):
        import time as _time

        from src.data.ingest import _FUTURE_TOLERANCE_S

        now = int(_time.time())
        ingester, _ = _fake_ingester(tmp_path)
        _write_log(ingester, "shared", [
            ("m1", "normal", now - 60),
            ("m2", "clock is wrong", now + _FUTURE_TOLERANCE_S + 10_000),
        ])

        ingester.ingest_scope("shared")

        assert ingester.state.watermark("shared") == now - 60
        # The real proof: a message logged AFTERWARDS is still reachable.
        _write_log(ingester, "shared", [("m3", "planted fact", now - 30)])
        result = ingester.ingest_scope("shared")
        assert result["messages"] >= 1

    def test_a_normal_batch_still_advances(self, tmp_path):
        import time as _time

        now = int(_time.time())
        ingester, _ = _fake_ingester(tmp_path)
        _write_log(ingester, "shared", [
            ("m1", "um", now - 7200), ("m2", "dois", now - 7100),
            ("m3", "tres", now - 7000),
        ])
        ingester.ingest_scope("shared")
        assert ingester.state.watermark("shared") == now - 7000

    def test_a_warm_tail_is_held_back_and_merged_next_run(self, tmp_path):
        """A 15-minute cadence must not shatter the index into 1-3 message chunks."""
        import time as _time

        now = int(_time.time())
        ingester, collection = _fake_ingester(tmp_path, settle_minutes=10)
        # 16 settled messages plus a 4-message tail that is still warm.
        settled = [(f"s{i}", f"velho {i}", now - 7200 + i) for i in range(16)]
        warm = [(f"w{i}", f"novo {i}", now - 60 + i) for i in range(4)]
        _write_log(ingester, "shared", settled + warm)

        ingester.ingest_scope("shared")

        sizes = [m["message_count"] for m in collection.metadatas]
        assert sizes == [16], f"the warm tail should not have been written: {sizes}"
        assert ingester.state.watermark("shared") < now - 60, \
            "the watermark must stop short of messages left unconsumed"

        # Once the tail settles it is picked up — nothing was lost.
        collection.metadatas.clear()
        ingester.settle_seconds = 0
        ingester.ingest_scope("shared")
        assert [m["message_count"] for m in collection.metadatas] == [4]


# ── image consent is not memory scope (2026-08-13) ───────────────────────────
def _image_config(**imagegen):
    return {"chat": {"imagegen": {"enabled": True, **imagegen}}}


def test_a_group_may_edit_even_when_its_memory_is_private():
    """The filed bug. The Kaya group was missing from whatsapp_shared_chats.json,
    so scope_for_chat returned "group:…" and an edit asked for IN the group was
    refused with "Só faço imagens no grupo, não por aqui."."""
    from src.chat import imagegen

    config = _image_config(allow_groups=True, allowed_scopes=["shared"])
    private_group = scope_for_chat("351939498856-1585236524@g.us", [])

    assert imagegen.allowed_here(config, private_group,
                                 chat_id="351939498856-1585236524@g.us",
                                 is_group=True) is True


def test_a_dm_is_still_refused():
    """Consent is the point of the gate: an arbitrary DM may not put a real
    member's face into an invented scene."""
    from src.chat import imagegen

    config = _image_config(allow_groups=True, allowed_scopes=["shared"])

    assert imagegen.allowed_here(config, scope_for_chat("351911111111@c.us", []),
                                 chat_id="351911111111@c.us", is_group=False) is False


def test_allowed_chats_wins_over_everything_else():
    from src.chat import imagegen

    config = _image_config(allowed_chats=["120363@g.us"], allow_groups=True)

    assert imagegen.allowed_here(config, "shared", chat_id="120363@g.us", is_group=True)
    assert not imagegen.allowed_here(config, "shared", chat_id="999@g.us", is_group=True)


def test_the_original_scope_rule_still_applies_when_nothing_else_is_set():
    from src.chat import imagegen

    config = _image_config(allowed_scopes=["shared"])

    assert imagegen.allowed_here(config, "shared", chat_id="x@g.us", is_group=True)
    assert not imagegen.allowed_here(config, "dm:abc", chat_id="x@c.us", is_group=False)


def test_a_disabled_feature_refuses_everyone():
    from src.chat import imagegen

    config = {"chat": {"imagegen": {"enabled": False, "allow_groups": True}}}

    assert imagegen.allowed_here(config, "shared", chat_id="x@g.us", is_group=True) is False


# ── naming a group that was never registered as shared memory ────────────────
def _write_chat_log(directory, name, chat_ids):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        "\n".join(json.dumps({"chat_id": c, "text": "olá"}) for c in chat_ids),
        encoding="utf-8")


def test_an_unregistered_group_is_named(tmp_path):
    """The state prod was actually in: the Kaya group was missing from
    whatsapp_shared_chats.json, so its history went to a private scope and image
    editing was refused in the room the bot exists for."""
    from src.chat.scope import unregistered_groups

    _write_chat_log(tmp_path / "live_messages", "shared.jsonl",
               ["351939498856-1585236524@g.us", "351911111111@c.us"])

    assert unregistered_groups(tmp_path / "live_messages", []) == \
        ["351939498856-1585236524@g.us"]


def test_a_registered_group_is_silent(tmp_path):
    from src.chat.scope import unregistered_groups

    _write_chat_log(tmp_path / "live_messages", "shared.jsonl", ["120363@g.us"])

    assert unregistered_groups(tmp_path / "live_messages", ["120363@g.us"]) == []


def test_dms_are_never_flagged(tmp_path):
    """A DM is private by design, not a misconfiguration."""
    from src.chat.scope import unregistered_groups

    _write_chat_log(tmp_path / "live_messages", "dm_abc.jsonl", ["351911111111@c.us"])

    assert unregistered_groups(tmp_path / "live_messages", []) == []


def test_a_missing_directory_is_not_an_error(tmp_path):
    from src.chat.scope import unregistered_groups

    assert unregistered_groups(tmp_path / "nope", []) == []


def test_a_corrupt_log_does_not_stop_startup(tmp_path):
    from src.chat.scope import unregistered_groups

    log_dir = tmp_path / "live_messages"
    _write_chat_log(log_dir, "good.jsonl", ["120363@g.us"])
    (log_dir / "bad.jsonl").write_text("{not json", encoding="utf-8")

    assert unregistered_groups(log_dir, []) == ["120363@g.us"]


# ── the reply edge in the corpus (bug 7acbc092, 2026-08-13) ──────────────────
# 158 of 504 messages in the live shared log are four words or fewer, and most
# are replies. Without the parent, "Ao contrário de outros" is unreadable — to a
# person and to anything extracting facts from it.
def _reply_msg(chat_id, mid, sender, text, ts, reply_to=None, quoted=""):
    from src.data.message_log import message_uid

    row = {"id": message_uid(chat_id, mid), "chat_id": chat_id, "sender": sender,
           "text": text, "timestamp": ts}
    if reply_to:
        row["reply_to_id"] = reply_to
        row["reply_to_text"] = quoted
    return row


CHAT = "120@g.us"
PARENT_TEXT = "Os judeus é que mandam nisto tudo"


def test_a_reply_carries_its_parent_into_the_chunk():
    from src.data.ingest import build_chunks

    chunks, _ = build_chunks(
        [_reply_msg(CHAT, "m2", "Gil", "A culpa é do", 200,
                    reply_to="m1", quoted=PARENT_TEXT)], "shared")

    assert f'Gil (a responder a "{PARENT_TEXT}"): A culpa é do' in chunks[0]["text"]


def test_a_parent_already_in_the_chunk_is_not_repeated():
    """A back-and-forth must not restate every message twice."""
    from src.data.ingest import build_chunks

    chunks, _ = build_chunks([
        _reply_msg(CHAT, "m1", "Pedro", PARENT_TEXT, 100),
        _reply_msg(CHAT, "m2", "Gil", "A culpa é do", 200,
                   reply_to="m1", quoted=PARENT_TEXT),
    ], "shared")

    assert chunks[0]["text"].count(PARENT_TEXT) == 1
    assert "a responder a" not in chunks[0]["text"]


def test_a_message_that_is_not_a_reply_is_unchanged():
    from src.data.ingest import build_chunks

    chunks, _ = build_chunks([_reply_msg(CHAT, "m3", "Rafa", "boas", 300)], "shared")

    assert chunks[0]["text"] == "Rafa: boas"


def test_the_log_records_the_reply_edge(tmp_path):
    from src.data.message_log import MessageLog

    log = MessageLog(base_dir=str(tmp_path))
    log.append(chat_id=CHAT, message_id="m2", sender="Gil", text="A culpa é do",
               timestamp=200, scope="shared", reply_to_id="m1",
               reply_to_text=PARENT_TEXT)

    row = json.loads((tmp_path / "shared.jsonl").read_text(encoding="utf-8").strip())
    assert row["reply_to_id"] == "m1"
    assert row["reply_to_text"] == PARENT_TEXT


def test_a_plain_message_gains_no_reply_keys(tmp_path):
    """Every message carrying empty reply fields would bloat the log for nothing."""
    from src.data.message_log import MessageLog

    log = MessageLog(base_dir=str(tmp_path))
    log.append(chat_id=CHAT, message_id="m3", sender="Rafa", text="boas",
               timestamp=300, scope="shared")

    row = json.loads((tmp_path / "shared.jsonl").read_text(encoding="utf-8").strip())
    assert "reply_to_id" not in row
