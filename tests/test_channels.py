"""The Community's sub-groups: the bot knows which one it is in, and memory knows
where each thing was said (2026-10-10)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import channels
from src.chat.channels import ChannelNames
from src.chat.engine import _channel_line
from src.data.ingest import Ingester, build_chunks

GENERAL = "351900000000-1500000000@g.us"
TRIPS = "120363000000000002@g.us"


# ── names ────────────────────────────────────────────────────────────────────
def test_a_label_overrides_the_subject(tmp_path):
    names = ChannelNames(tmp_path / "names.json")
    names.remember(TRIPS, "Viagens ✈️ 2027", now=100)
    assert names.label(TRIPS) == "Viagens ✈️ 2027"
    data = json.loads((tmp_path / "names.json").read_text())
    data["labels"] = {TRIPS: "Viagens"}
    (tmp_path / "names.json").write_text(json.dumps(data))
    assert names.label(TRIPS) == "Viagens"
    assert names.label(GENERAL) == ""
    assert names.label(None) == ""


def test_a_name_is_looked_up_again_after_a_day(tmp_path):
    names = ChannelNames(tmp_path / "names.json")
    assert names.stale(TRIPS, 86400, now=1000)
    names.remember(TRIPS, "", now=1000)
    assert not names.stale(TRIPS, 86400, now=1000 + 3600)
    assert names.stale(TRIPS, 86400, now=1000 + 86400)
    assert names.label(TRIPS) == ""


def test_the_prompt_line_names_the_channel_only_for_a_known_group(tmp_path):
    path = tmp_path / "names.json"
    ChannelNames(path).remember(TRIPS, "Viagens", now=1)
    config = {"whatsapp": {"chat_names_file": str(path)}}
    assert "«Viagens»" in _channel_line(config, TRIPS)
    assert _channel_line(config, GENERAL) == ""
    assert _channel_line(config, "351911111111@c.us") == ""
    assert _channel_line(config, None) == ""
    assert channels.prompt_line("") == ""


# ── chunks ───────────────────────────────────────────────────────────────────
def _row(message_id, chat_id, text, ts):
    return {"id": message_id, "chat_id": chat_id, "sender": "Alguém", "text": text,
            "timestamp": ts}


def test_a_chunk_records_its_chat_and_channel():
    chunks, _ = build_chunks([_row("a", TRIPS, "Ibiza?", 100), _row("b", TRIPS, "bora", 101)],
                             "shared", channel_of=lambda chat: "Viagens" if chat == TRIPS else "")
    assert chunks[0]["metadata"]["chat_id"] == TRIPS
    assert chunks[0]["metadata"]["channel"] == "Viagens"


class _Collection:
    def __init__(self):
        self.documents, self.metadatas = [], []

    def upsert(self, ids, documents, metadatas, embeddings):
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)


class _Encoder:
    def encode(self, texts, **kwargs):
        import numpy as np

        return np.zeros((len(texts), 8), dtype="float32")


def _ingester(tmp_path, settle_minutes=0):
    names = tmp_path / "names.json"
    ChannelNames(names).remember(TRIPS, "Viagens", now=1)
    config = {
        "rag": {"db_path": str(tmp_path / "db")},
        "chat": {"concurrency": {"max_concurrent": 1, "acquire_timeout": 5}},
        "whatsapp": {
            "message_log_dir": str(tmp_path / "log"),
            "ingest_state_file": str(tmp_path / "state.json"),
            "ingest": {"settle_minutes": settle_minutes},
            "chat_names_file": str(names),
        },
    }
    collection = _Collection()
    return Ingester(config, encoder=_Encoder(), collection=collection), collection


def _log(ingester, rows):
    for message_id, chat_id, text, ts in rows:
        ingester.log.append(chat_id=chat_id, message_id=message_id, sender="Alguém",
                            text=text, timestamp=ts, scope="shared")


def test_interleaved_sub_groups_are_chunked_apart(tmp_path):
    now = int(time.time())
    ingester, collection = _ingester(tmp_path)
    _log(ingester, [("g1", GENERAL, "bom dia", now - 7200), ("t1", TRIPS, "Ibiza ou Split?", now - 7190),
                    ("g2", GENERAL, "jogo logo?", now - 7180), ("t2", TRIPS, "Split", now - 7170)])
    ingester.ingest_scope("shared")
    by_channel = {meta["channel"]: doc for meta, doc in zip(collection.metadatas, collection.documents)}
    assert set(by_channel) == {"", "Viagens"}
    assert "Ibiza" in by_channel["Viagens"] and "bom dia" not in by_channel["Viagens"]
    assert "Ibiza" not in by_channel[""]


def test_a_quiet_chats_warm_tail_does_not_hold_back_or_re_chunk_a_busy_one(tmp_path):
    now = int(time.time())
    ingester, collection = _ingester(tmp_path, settle_minutes=10)
    general = [(f"g{i}", GENERAL, f"velho {i}", now - 7200 + i) for i in range(16)]
    _log(ingester, [("t1", TRIPS, "Ibiza?", now - 7300), *general,
                    ("t2", TRIPS, "bora decidir", now - 60)])
    ingester.ingest_scope("shared")
    assert sorted(m["message_count"] for m in collection.metadatas) == [16]

    collection.metadatas.clear()
    collection.documents.clear()
    ingester.settle_seconds = 0
    ingester.ingest_scope("shared")
    assert [m["message_count"] for m in collection.metadatas] == [2]
    assert collection.metadatas[0]["channel"] == "Viagens"


def test_a_tail_that_is_all_warm_is_not_skipped(tmp_path):
    """Nothing flushed used to mean the watermark jumped to the newest message."""
    now = int(time.time())
    ingester, collection = _ingester(tmp_path, settle_minutes=10)
    _log(ingester, [("t1", TRIPS, "Ibiza?", now - 120), ("t2", TRIPS, "Split", now - 60)])
    ingester.ingest_scope("shared")
    assert collection.metadatas == []
    ingester.settle_seconds = 0
    ingester.ingest_scope("shared")
    assert [m["message_count"] for m in collection.metadatas] == [2]


def test_an_existing_scope_watermark_is_where_every_chat_starts(tmp_path):
    now = int(time.time())
    ingester, collection = _ingester(tmp_path)
    ingester.state.set_watermark("shared", now - 5000, 0)
    _log(ingester, [("old", GENERAL, "já ingerido", now - 6000), ("new", TRIPS, "novo", now - 4000)])
    ingester.ingest_scope("shared")
    assert len(collection.documents) == 1 and "novo" in collection.documents[0]


# ── retrieval ────────────────────────────────────────────────────────────────
def _retriever(collection, rag=None):
    from src.chat.retriever import ConversationRetriever

    retriever = ConversationRetriever.__new__(ConversationRetriever)
    retriever.collection = collection
    retriever.rag_config = rag or {"same_channel_top_k": 2, "min_similarity": 0.3}
    return retriever


class _QueryCollection:
    def __init__(self, rows):
        self.rows = rows
        self.where = None

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results, where, include):
        self.where = where
        chat = where["$and"][1]["chat_id"] if "$and" in where else where["chat_id"]
        hits = [row for row in self.rows if row[1].get("chat_id") == chat][:n_results]
        return {"documents": [[row[0] for row in hits]], "metadatas": [[row[1] for row in hits]],
                "distances": [[row[2] for row in hits]]}


def test_the_channel_being_asked_goes_first_within_scope():
    rows = [("Split ganhou", {"chat_id": TRIPS, "scope": "shared", "channel": "Viagens"}, 0.4),
            ("muito longe", {"chat_id": TRIPS, "scope": "shared", "channel": "Viagens"}, 0.9)]
    collection = _QueryCollection(rows)
    others = [{"text": "jogo do Benfica", "metadata": {}, "rank": 1}]
    merged = _retriever(collection)._prepend_same_channel(others, TRIPS, [0.0], "shared", None)
    assert [chunk["text"] for chunk in merged] == ["Split ganhou", "jogo do Benfica"]
    assert collection.where["$and"][0] == {"scope": "shared"}


def test_no_boost_outside_a_group_or_when_off():
    collection = _QueryCollection([])
    chunks = [{"text": "x", "metadata": {}}]
    assert _retriever(collection)._prepend_same_channel(chunks, "351911111111@c.us", [0.0],
                                                         "dm:abc", None) is chunks
    assert _retriever(collection, {"same_channel_top_k": 0})._prepend_same_channel(
        chunks, TRIPS, [0.0], "shared", None) is chunks
    assert collection.where is None


def test_the_context_says_which_channel_a_conversation_was_in():
    text = _retriever(None).format_context([
        {"text": "Split ganhou", "metadata": {"channel": "Viagens"}},
        {"text": "bom dia", "metadata": {}},
    ])
    assert "--- Conversa 1 [canal: Viagens] ---" in text
    assert "--- Conversa 2 ---" in text


# ── the adapter learns the names ─────────────────────────────────────────────
def test_the_adapter_learns_a_shared_groups_name_once_a_day(tmp_path):
    from src.chat.waha_client import MockWahaClient
    from src.chat.whatsapp_adapter import WhatsAppAdapter

    class Client(MockWahaClient):
        lookups = 0

        def group_info(self, chat_id):
            Client.lookups += 1
            return {"subject": "Viagens", "linkedParent": "parent@g.us"}

    names = tmp_path / "names.json"
    adapter = WhatsAppAdapter(lambda *a, **k: "ok", Client(), {"whatsapp": {
        "shared_chats": [TRIPS], "chat_names_file": str(names),
        "message_log_dir": str(tmp_path / "log")}})
    adapter._learn_channel(TRIPS)
    adapter._learn_channel(TRIPS)
    adapter._learn_channel("120363000000000009@g.us")
    assert Client.lookups == 1
    assert ChannelNames(names).label(TRIPS) == "Viagens"


# ── the one-time rebuild of interleaved chunks ───────────────────────────────
class _StoreCollection(_Collection):
    def __init__(self, stored):
        super().__init__()
        self.stored = stored
        self.deleted = []

    def get(self, where, include):
        return {"ids": list(self.stored), "metadatas": list(self.stored.values())}

    def delete(self, ids):
        self.deleted.extend(ids)


def test_rechunk_since_rebuilds_the_interleaved_chunks_once(tmp_path):
    ingester, _ = _ingester(tmp_path)
    store = _StoreCollection({
        "old": {"timestamp_start": "2026-09-20T10:00:00", "timestamp_end": "2026-09-20T11:00:00"},
        "mixed": {"timestamp_start": "2026-09-30T23:00:00", "timestamp_end": "2026-10-01T01:00:00"},
        "new": {"timestamp_start": "2026-10-05T10:00:00", "timestamp_end": "2026-10-05T11:00:00"},
    })
    ingester._collection = store
    ingester.state.set_chat_watermarks("shared", {GENERAL: 1_800_000_000}, 0)

    assert ingester.rechunk_since("shared", "2026-10-01")["rechunked"] == 2
    assert sorted(store.deleted) == ["mixed", "new"]
    from datetime import datetime, timezone
    rewind = int(datetime(2026, 9, 30, 23, tzinfo=timezone.utc).timestamp()) - 1
    assert ingester.state.watermark("shared") == rewind
    assert ingester.state.chat_watermark("shared", GENERAL) == rewind

    assert ingester.rechunk_since("shared", "2026-10-01")["rechunked"] == 0
    assert len(store.deleted) == 2
