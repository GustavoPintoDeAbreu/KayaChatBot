"""ensure_model_loaded against a mocked llama-swap broker, and Whisper's idle unload."""
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import stt
from src.chat.inference_backend import broker_target, ensure_model_loaded


class _Resp:
    def __init__(self, payload=None):
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def _config(tmp_path, url):
    return {"inference": {"gguf": {"server_url": url},
                          "broker": {"evict_before_use": ["big", "qwen-impl-g1"],
                                     "activity_file": str(tmp_path / "active")}}}


def test_broker_target_parses_upstream_urls():
    assert broker_target("http://llm-broker:8080/upstream/kaya") == ("http://llm-broker:8080", "kaya")
    assert broker_target("http://llm-broker:8080/upstream/kaya/") == ("http://llm-broker:8080", "kaya")
    assert broker_target("http://llama:8080") is None


def test_plain_llama_server_only_records_activity(tmp_path, monkeypatch):
    monkeypatch.delenv("KAYA_LLAMA_URL", raising=False)
    calls = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: calls.append(a) or _Resp())
    ensure_model_loaded(_config(tmp_path, "http://llama:8080"))
    assert calls == []
    assert (tmp_path / "active").read_text().strip().isdigit()


def test_broker_evicts_borrowers_then_loads(tmp_path, monkeypatch):
    monkeypatch.delenv("KAYA_LLAMA_URL", raising=False)
    gets, posts = [], []

    def fake_get(url, **kwargs):
        gets.append(url)
        if url.endswith("/running"):
            return _Resp({"running": [{"model": "big", "state": "ready"}]})
        return _Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", lambda url, **k: posts.append(url) or _Resp())
    ensure_model_loaded(_config(tmp_path, "http://b:8080/upstream/kaya"))
    assert posts == ["http://b:8080/api/models/unload/big"]
    assert gets[-1] == "http://b:8080/upstream/kaya/health"


def test_broker_failure_never_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("KAYA_LLAMA_URL", raising=False)

    def boom(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "get", boom)
    ensure_model_loaded(_config(tmp_path, "http://b:8080/upstream/kaya"))


def test_whisper_unload_drops_the_model(monkeypatch):
    monkeypatch.setattr(stt, "_model", object())
    stt.unload()
    assert stt._model is None


def test_whisper_idle_timer_respects_zero(monkeypatch):
    monkeypatch.setattr(stt, "_unload_timer", None)
    stt._schedule_unload({"chat": {"audio": {"whisper_idle_unload_minutes": 0}}})
    assert stt._unload_timer is None
    stt._schedule_unload({"chat": {"audio": {"whisper_idle_unload_minutes": 5}}})
    assert stt._unload_timer is not None
    stt._unload_timer.cancel()
