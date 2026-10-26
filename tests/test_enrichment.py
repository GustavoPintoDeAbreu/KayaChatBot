"""Media nobody asked about waits while GPU1 is lent, and is filled in later.

Only a reply takes the card back from an idea build (2026-10-26). These pin the
queue, the fill (log amendment + session line), what a reply hears first, what the
adapter defers and what it does not, the ingest hold, and the other paths that
must not load Kaya while the card is lent.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import enrichment, inference_backend
from src.chat.enrichment import EnrichmentQueue, Enricher, PLACEHOLDERS, barriers, strip_pending
from src.chat.memory import KeyedSessionMemory
from src.chat.waha_client import MockWahaClient
from src.chat.whatsapp_adapter import WhatsAppAdapter
from src.data.message_log import MessageLog, message_uid

BOT_JID = "351900000000@c.us"
GROUP = "12036300000000@g.us"
ALICE = "351911111111@c.us"
CONFIG = {"chat": {"concurrency": {"max_concurrent": 1, "acquire_timeout": 5}}}

_ids = itertools.count(1)


@pytest.fixture(autouse=True)
def _fresh_lent_cache():
    inference_backend._lent_cache.update(at=0.0, value=False)
    yield
    inference_backend._lent_cache.update(at=0.0, value=False)


# ── inference_backend: gpu1_lent and the non-urgent path ──────────────────────

class _Resp:
    def __init__(self, payload=None):
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def _broker_config(tmp_path):
    return {"inference": {"gguf": {"server_url": "http://b:8080/upstream/kaya"},
                          "broker": {"evict_before_use": ["big", "*-g1"],
                                     "activity_file": str(tmp_path / "active"),
                                     "lent_cache_seconds": 0}}}


def _running(monkeypatch, models):
    def fake_get(url, **kwargs):
        if url.endswith("/running"):
            return _Resp({"running": [{"model": m, "state": "ready"} for m in models]})
        return _Resp()
    monkeypatch.setattr(requests, "get", fake_get)


def test_gpu1_is_lent_only_to_a_borrower(tmp_path, monkeypatch):
    monkeypatch.delenv("KAYA_LLAMA_URL", raising=False)
    config = _broker_config(tmp_path)
    _running(monkeypatch, ["kaya", "qwen-impl"])
    assert not inference_backend.gpu1_lent(config)
    _running(monkeypatch, ["qwen-flash-g1"])
    assert inference_backend.gpu1_lent(config)
    assert not inference_backend.gpu1_lent({"inference": {"gguf": {"server_url": "http://llama:8080"}}})


def test_an_unreachable_broker_counts_as_lent_and_the_env_forces_it(tmp_path, monkeypatch):
    monkeypatch.delenv("KAYA_LLAMA_URL", raising=False)

    def down(*args, **kwargs):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "get", down)
    assert inference_backend.gpu1_lent(_broker_config(tmp_path))
    monkeypatch.setenv("KAYA_GPU1_LENT", "0")
    assert not inference_backend.gpu1_lent(_broker_config(tmp_path))


def test_non_urgent_work_never_evicts_or_restarts_the_idle_clock(tmp_path, monkeypatch):
    monkeypatch.delenv("KAYA_LLAMA_URL", raising=False)
    posts = []
    monkeypatch.setattr(requests, "post", lambda url, **k: posts.append(url) or _Resp())
    _running(monkeypatch, ["qwen-flash-g1"])
    config = _broker_config(tmp_path)
    assert inference_backend.ensure_model_loaded(config, urgent=False) is False
    _running(monkeypatch, ["kaya"])
    assert inference_backend.ensure_model_loaded(config, urgent=False) is True
    assert posts == [] and not (tmp_path / "active").exists()
    _running(monkeypatch, ["qwen-flash-g1"])
    assert inference_backend.ensure_model_loaded(config) is True
    assert posts[0] == "http://b:8080/api/models/unload/qwen-flash-g1"
    assert (tmp_path / "active").exists()


# ── message_log: amendments ────────────────────────────────────────────────────

def test_an_amendment_replaces_the_text_in_place_and_wins_over_a_replay(tmp_path):
    log = MessageLog(str(tmp_path / "log"))
    log.append(chat_id=GROUP, message_id="m1", sender="Alice", text=PLACEHOLDERS["image"],
               timestamp=100, scope="shared")
    log.append(chat_id=GROUP, message_id="m2", sender="Bob", text="lol", timestamp=101, scope="shared")
    log.amend(chat_id=GROUP, message_id="m1", sender="Alice", text="[Imagem: um barco à vela no mar ao pôr do sol]",
              timestamp=100, scope="shared")
    log._seen.clear()
    log.append(chat_id=GROUP, message_id="m1", sender="Alice", text=PLACEHOLDERS["image"],
               timestamp=100, scope="shared")
    texts = [row["text"] for row in log.read("shared")]
    assert texts[:2] == ["[Imagem: um barco à vela no mar ao pôr do sol]", "lol"]
    assert PLACEHOLDERS["image"] not in texts
    assert [row["timestamp"] for row in log.read("shared", after_ts=100)] == [101]


def test_an_empty_amendment_hides_the_message(tmp_path):
    log = MessageLog(str(tmp_path / "log"))
    log.append(chat_id=GROUP, message_id="v1", sender="Alice", text=PLACEHOLDERS["audio"],
               timestamp=100, scope="shared")
    log.amend(chat_id=GROUP, message_id="v1", sender="Alice", text="", timestamp=100, scope="shared")
    assert list(log.read("shared")) == []


# ── the queue and the Enricher ─────────────────────────────────────────────────

class _Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def _rig(tmp_path, can_run=True, fillers=None, clock=None):
    log = MessageLog(str(tmp_path / "log"))
    sessions = KeyedSessionMemory(base_dir=str(tmp_path / "s"), max_lines=20)
    state = {"can_run": can_run, "urgent": 0}

    def prepare():
        state["urgent"] += 1
    enricher = Enricher(CONFIG, EnrichmentQueue(tmp_path / "q"),
                        fillers or {"image": lambda part, payload, entry: "[Imagem: um barco à vela no mar ao pôr do sol]",
                                    "sticker": lambda part, payload, entry: "[Sticker: um sapo]",
                                    "audio": lambda part, payload, entry: "vamos às 8"},
                        log, sessions, can_run=lambda: state["can_run"], prepare_urgent=prepare,
                        now=clock or time.time)
    return enricher, log, sessions, state


def _defer(enricher, log, sessions, message_id, kind, caption="", ts=100, sender="Alice"):
    block = PLACEHOLDERS[kind]
    text = f"{caption}\n{block}" if caption else block
    log.append(chat_id=GROUP, message_id=message_id, sender=sender, text=text,
               timestamp=ts, scope="shared")
    line = f"{sender}: {text}"
    sessions.append(GROUP, line)
    enricher.defer(uid=message_uid(GROUP, message_id), message_id=message_id, chat_id=GROUP,
                   scope="shared", timestamp=ts, log={"sender": sender}, text=text,
                   session_line=line, parts=[{"kind": kind, "block": block}],
                   payloads={0: b"bytes"})
    return line


def test_a_fill_amends_the_log_and_the_session_line_and_drops_the_media(tmp_path):
    enricher, log, sessions, _ = _rig(tmp_path)
    line = _defer(enricher, log, sessions, "p1", "image", caption="olha isto")
    media = list((tmp_path / "q" / "media").iterdir())
    assert len(media) == 1
    assert enricher.drain() == 1
    assert [row["text"] for row in log.read("shared")] == ["olha isto\n[Imagem: um barco à vela no mar ao pôr do sol]"]
    assert sessions.recent(GROUP) == ["Alice: olha isto\n[Imagem: um barco à vela no mar ao pôr do sol]"]
    assert enricher.resolve_line(line) == "Alice: olha isto\n[Imagem: um barco à vela no mar ao pôr do sol]"
    assert len(enricher.queue) == 0 and not media[0].exists()


def test_a_failed_fill_drops_the_placeholder_and_a_bare_voice_note_disappears(tmp_path):
    enricher, log, sessions, _ = _rig(tmp_path, fillers={"audio": lambda *a: None,
                                                          "image": lambda *a: None})
    _defer(enricher, log, sessions, "v1", "audio")
    _defer(enricher, log, sessions, "p1", "image", caption="olha", ts=101)
    enricher.drain()
    assert [row["text"] for row in log.read("shared")] == ["olha"]
    assert sessions.recent(GROUP) == ["Alice: olha"]


def test_the_queue_survives_a_restart(tmp_path):
    enricher, log, sessions, _ = _rig(tmp_path, can_run=False)
    _defer(enricher, log, sessions, "p1", "image")
    again, *_ = _rig(tmp_path, can_run=True)
    assert len(again.queue) == 1
    assert again.drain() == 1


def test_draining_stops_while_gpu1_is_lent_and_errors_are_retried_then_given_up(tmp_path):
    calls = []

    def flaky(part, payload, entry):
        calls.append(1)
        raise RuntimeError("broker down")
    enricher, log, sessions, state = _rig(tmp_path, can_run=False, fillers={"image": flaky})
    _defer(enricher, log, sessions, "p1", "image", caption="x")
    assert enricher.drain() == 0 and calls == []
    state["can_run"] = True
    for _ in range(2):
        assert enricher.drain() == 0
    assert enricher.queue.get(message_uid(GROUP, "p1"))["attempts"] == 2
    assert enricher.drain() == 1
    assert [row["text"] for row in log.read("shared")] == ["x"]


def test_a_busy_gpu_leaves_the_entry_queued(tmp_path):
    from src.chat.gpu_lock import get_gpu_lock, reset_gpu_lock

    reset_gpu_lock()
    enricher, log, sessions, _ = _rig(tmp_path)
    _defer(enricher, log, sessions, "p1", "image")
    lock = get_gpu_lock(CONFIG)
    lock.acquire()
    try:
        assert enricher.process(message_uid(GROUP, "p1"), lock_timeout=0.05) is False
    finally:
        lock.release()
        reset_gpu_lock()
    assert len(enricher.queue) == 1


def test_entries_older_than_a_week_are_dropped_unfilled(tmp_path):
    clock = _Clock()
    enricher, log, sessions, _ = _rig(tmp_path, clock=clock)
    _defer(enricher, log, sessions, "p1", "image", caption="velha")
    clock.now += 8 * 86400
    enricher.drain()
    assert [row["text"] for row in log.read("shared")] == ["velha"]


def test_before_a_reply_only_this_chats_voice_notes_and_the_quoted_photo_are_filled(tmp_path):
    enricher, log, sessions, state = _rig(tmp_path, can_run=False)
    _defer(enricher, log, sessions, "v1", "audio", ts=100)
    _defer(enricher, log, sessions, "p1", "image", ts=101)
    _defer(enricher, log, sessions, "p2", "image", ts=102)
    assert enricher.before_reply(GROUP, reply_to_id="p2") == 2
    assert state["urgent"] == 1
    remaining = [entry["message_id"] for entry in enricher.queue.entries()]
    assert remaining == ["p1"]
    assert enricher.before_reply("other@g.us") == 0 and state["urgent"] == 1


def test_barriers_hold_a_chat_at_its_oldest_pending_message_for_a_while(tmp_path, monkeypatch):
    clock = _Clock(time.time())
    enricher, log, sessions, _ = _rig(tmp_path, clock=clock)
    monkeypatch.setenv("KAYA_ENRICHMENT_DIR", str(tmp_path / "q"))
    _defer(enricher, log, sessions, "p1", "image", ts=200)
    _defer(enricher, log, sessions, "p2", "image", ts=150)
    assert barriers(CONFIG, "shared") == {GROUP: 150}
    assert barriers(CONFIG, "dm:x") == {}
    assert barriers(CONFIG, "shared", now=time.time() + 13 * 3600) == {}


def test_strip_pending_keeps_the_caption():
    assert strip_pending(f"olha isto\n{PLACEHOLDERS['image']}") == "olha isto"
    assert strip_pending(PLACEHOLDERS["audio"]) == ""
    assert strip_pending(enrichment.document_placeholder("relatório.pdf")) == ""
    assert strip_pending("[Imagem: um barco à vela no mar ao pôr do sol]") == "[Imagem: um barco à vela no mar ao pôr do sol]"


# ── the adapter: what is deferred and what is not ──────────────────────────────

class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, message, speaker, recent_lines, scope=None, exclude_from=None):
        self.calls.append({"message": message, "recent": list(recent_lines)})
        return "ok"


def _event(text="", media=None, mention=False, reply_to=None):
    payload = {"id": f"en{next(_ids)}", "from": GROUP, "participant": ALICE, "body": text,
               "notifyName": "Alice", "mentionedIds": [BOT_JID] if mention else [],
               "timestamp": int(time.time()), "_data": {"message": {}}}
    if media:
        payload["media"] = media
        payload["hasMedia"] = True
    if reply_to:
        payload["replyTo"] = {"participant": ALICE, "body": "", "id": reply_to}
    return {"event": "message", "payload": payload}


PHOTO = {"url": "http://waha/p.jpg", "mimetype": "image/jpeg"}
VOICE = {"url": "http://waha/v.ogg", "mimetype": "audio/ogg; codecs=opus"}


def _adapter(tmp_path, lent=True):
    config = {**CONFIG, "whatsapp": {"bot_jid": BOT_JID, "send_seen": False, "history_turns": 5,
                                     "contacts": {ALICE: "Alice"}, "shared_chats": [GROUP]}}
    responder = _Recorder()
    called = {"describe": [], "transcribe": []}
    adapter = WhatsAppAdapter(
        responder, MockWahaClient(echo=False), config,
        session_store=KeyedSessionMemory(base_dir=str(tmp_path / "s"), max_lines=20),
        message_log=MessageLog(str(tmp_path / "log")),
        describe_image=lambda url, mime, urgent=True: called["describe"].append(urgent) or "um barco à vela no mar ao pôr do sol",
        transcribe=lambda url, mime, urgent=True: called["transcribe"].append(urgent) or "vamos às 8",
        fetch_media=lambda msg, cap: b"bytes")
    state = {"lent": lent}
    adapter.enricher = Enricher(
        config, EnrichmentQueue(tmp_path / "q"),
        {"image": lambda part, payload, entry: "[Imagem: um barco à vela no mar ao pôr do sol]",
         "audio": lambda part, payload, entry: "vamos às 8"},
        adapter.message_log, adapter.session_store,
        can_run=lambda: not state["lent"], prepare_urgent=lambda: state.update(lent=False))
    return adapter, responder, called, state


def _texts(adapter):
    return [row["text"] for row in adapter.message_log.read("shared")]


def test_an_unaddressed_photo_and_voice_note_wait_while_gpu1_is_lent(tmp_path):
    adapter, _, called, _ = _adapter(tmp_path)
    adapter.handle_event(_event("olha", media=PHOTO))
    adapter.handle_event(_event(media=VOICE))
    assert called == {"describe": [], "transcribe": []}
    assert _texts(adapter) == [f"olha\n{PLACEHOLDERS['image']}", PLACEHOLDERS["audio"]]
    assert len(adapter.enricher.queue) == 2


def test_with_gpu1_free_media_is_read_at_once_without_taking_the_card(tmp_path):
    adapter, _, called, _ = _adapter(tmp_path, lent=False)
    adapter.handle_event(_event(media=PHOTO))
    assert called["describe"] == [False]
    assert _texts(adapter) == ["[Imagem: um barco à vela no mar ao pôr do sol]"] and len(adapter.enricher.queue) == 0


def test_an_addressed_photo_is_read_at_once_as_today(tmp_path):
    adapter, responder, called, _ = _adapter(tmp_path)
    adapter.handle_event(_event("@351900000000 quem é este?", media=PHOTO, mention=True))
    assert called["describe"] == [True]
    assert "um barco à vela" in _texts(adapter)[0] and len(adapter.enricher.queue) == 0


def test_a_mention_hears_the_voice_notes_sent_while_gpu1_was_lent(tmp_path):
    adapter, responder, called, state = _adapter(tmp_path)
    adapter.handle_event(_event(media=VOICE))
    adapter.handle_event(_event(media=VOICE))
    adapter.handle_event(_event("olha", media=PHOTO))
    adapter.handle_event(_event("@351900000000 o que disse ela no áudio?", mention=True))
    recent = responder.calls[-1]["recent"]
    assert recent.count("Alice: vamos às 8") == 2
    assert not any("(ainda por ouvir)" in line for line in recent)
    assert [entry["parts"][0]["kind"] for entry in adapter.enricher.queue.entries()] == ["image"]


def test_a_reply_quoting_a_pending_photo_sees_it_described(tmp_path):
    adapter, responder, _, _ = _adapter(tmp_path)
    photo = _event("olha", media=PHOTO)
    adapter.handle_event(photo)
    adapter.handle_event(_event("@351900000000 o que é isto?", mention=True,
                                reply_to=photo["payload"]["id"]))
    assert any("[Imagem: um barco à vela no mar ao pôr do sol]" in line for line in responder.calls[-1]["recent"])
    assert len(adapter.enricher.queue) == 0


# ── ingest: chunks wait for pending media ──────────────────────────────────────

class _Collection:
    def __init__(self):
        self.documents = []

    def upsert(self, ids, documents, metadatas, embeddings):
        self.documents.extend(documents)


class _Encoder:
    def encode(self, texts, **kwargs):
        import numpy as np

        return np.zeros((len(texts), 8), dtype="float32")


def test_ingest_holds_a_chat_at_its_pending_media_then_chunks_it_filled(tmp_path):
    from src.data.ingest import Ingester

    now = int(time.time()) - 3600
    config = {**CONFIG, "rag": {"db_path": str(tmp_path / "db")},
              "whatsapp": {"message_log_dir": str(tmp_path / "log"),
                           "ingest_state_file": str(tmp_path / "state.json"),
                           "ingest": {"settle_minutes": 0}}}
    collection = _Collection()
    ingester = Ingester(config, encoder=_Encoder(), collection=collection)
    enricher = Enricher(config, EnrichmentQueue(enrichment.queue_dir(config)),
                        {"image": lambda *a: "[Imagem: um barco à vela no mar ao pôr do sol]"}, ingester.log,
                        KeyedSessionMemory(base_dir=str(tmp_path / "s")), can_run=lambda: True,
                        prepare_urgent=lambda: None)
    ingester.log.append(chat_id=GROUP, message_id="a", sender="Bob", text="bom dia",
                        timestamp=now, scope="shared")
    ingester.log.append(chat_id=GROUP, message_id="p", sender="Alice", text=PLACEHOLDERS["image"],
                        timestamp=now + 1, scope="shared")
    ingester.log.append(chat_id=GROUP, message_id="b", sender="Bob", text="que barco!",
                        timestamp=now + 2, scope="shared")
    enricher.defer(uid=message_uid(GROUP, "p"), message_id="p", chat_id=GROUP, scope="shared",
                   timestamp=now + 1, log={"sender": "Alice"}, text=PLACEHOLDERS["image"],
                   session_line="", parts=[{"kind": "image", "block": PLACEHOLDERS["image"]}],
                   payloads={})
    ingester.ingest_scope("shared")
    assert collection.documents == []
    enricher.drain()
    ingester.ingest_scope("shared")
    assert collection.documents == ["Bob: bom dia\nAlice: [Imagem: um barco à vela no mar ao pôr do sol]\nBob: que barco!"]


def test_a_late_fill_becomes_a_chunk_of_its_own(tmp_path):
    from src.data.ingest import Ingester

    config = {**CONFIG, "rag": {"db_path": str(tmp_path / "db")},
              "whatsapp": {"message_log_dir": str(tmp_path / "log"),
                           "ingest_state_file": str(tmp_path / "state.json")}}
    queue = EnrichmentQueue(enrichment.queue_dir(config))
    queue.add({"uid": "u1", "parts": []}, {})
    queue.remove("u1", late_fill={"id": "u1", "chat_id": GROUP, "scope": "shared", "timestamp": 100,
                                  "text": "[Imagem: um barco à vela no mar ao pôr do sol]", "sender": "Alice"})
    collection = _Collection()
    Ingester(config, encoder=_Encoder(), collection=collection).ingest_scope("shared")
    assert collection.documents == ["Alice: [Imagem: um barco à vela no mar ao pôr do sol]"]
    assert queue.late_fills("shared") == []


# ── the other paths that could load Kaya ───────────────────────────────────────

def test_a_bio_refresh_call_is_refused_while_gpu1_is_lent(monkeypatch):
    from src.chat.gpu_lock import GpuBusyError
    from src.data import bio_refresh

    monkeypatch.setenv("KAYA_GPU1_LENT", "1")

    class Backend:
        def generate(self, *args, **kwargs):
            raise AssertionError("must not be called")
    with pytest.raises(GpuBusyError):
        bio_refresh._generate(Backend(), CONFIG, "s", "u", 10)


def test_a_link_synopsis_is_skipped_while_gpu1_is_lent(monkeypatch):
    from src.chat import links

    monkeypatch.setenv("KAYA_GPU1_LENT", "1")
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    content = links.LinkContent(url="https://e.pt/a", title="Título", site="e.pt",
                                text=" ".join(["palavra"] * 40))
    assert links.synopsis(content, CONFIG) == ""


def test_the_summary_never_sees_a_placeholder_line():
    from src.chat.summary import SummaryWriter

    writer = SummaryWriter.__new__(SummaryWriter)
    writer.ledger = None
    lines = [f"Alice: {PLACEHOLDERS['image']}", f"Bob: olha\n{PLACEHOLDERS['image']}", "Bob: lol"]
    assert writer._prompt_lines(GROUP, lines) == [lines[1], "Bob: lol"]
