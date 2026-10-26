"""Backend selection after the fine-tuning removal (2026-10-26).

Only HTTP backends remain: ``gguf`` (a llama.cpp server) and ``ollama`` (Ollama
behind the GPU broker, prod's runtime). The in-process ``hf`` backend is gone:
the env or config value now raises, and ``engine._load_model`` loads a tokenizer
only — no weights, no adapter. kaya-test loads ``.env``, which may set the
backend vars, so every test starts from a clean environment.
"""

import sys
import types

import pytest

from src.chat.inference_backend import build_backend, resolve_backend
from src.chat.engine import _load_model


@pytest.fixture(autouse=True)
def clean_backend_env(monkeypatch):
    for name in ("KAYA_INFERENCE_BACKEND", "KAYA_LLAMA_URL", "KAYA_OLLAMA_URL", "KAYA_OLLAMA_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_default_backend_is_gguf():
    assert resolve_backend({}) == "gguf"


def test_env_wins_over_config(monkeypatch):
    monkeypatch.setenv("KAYA_INFERENCE_BACKEND", "ollama")
    assert resolve_backend({"inference": {"backend": "gguf"}}) == "ollama"


def test_build_backend_gguf():
    backend = build_backend(
        {"inference": {"backend": "gguf", "gguf": {"server_url": "http://llama.test:8080"}}},
        None,
        object(),
    )
    assert type(backend).__name__ == "LlamaCppBackend"
    assert backend.server_url == "http://llama.test:8080"


def test_build_backend_ollama():
    backend = build_backend(
        {"inference": {"backend": "ollama", "ollama": {"server_url": "http://ollama.test:11434", "model": "m"}}},
        None,
        object(),
    )
    assert type(backend).__name__ == "OllamaBackend"
    assert backend.server_url == "http://ollama.test:11434"
    assert backend.model == "m"


def test_build_backend_rejects_hf():
    with pytest.raises(ValueError, match="hf"):
        build_backend({"inference": {"backend": "hf"}}, None, object())


def test_hf_backend_class_is_gone():
    from src.chat import inference_backend

    assert not hasattr(inference_backend, "HFBackend")


@pytest.mark.parametrize("backend", ["gguf", "ollama"])
def test_load_model_is_tokenizer_only(monkeypatch, backend):
    loaded = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            loaded.append(path)
            return "tokenizer"

    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
    )
    result = _load_model({"training": {"output_dir": "models/tok"}, "inference": {"backend": backend}})
    assert result == (None, "tokenizer")
    assert loaded == ["models/tok"]


def test_no_in_process_loader_left():
    from pathlib import Path

    for source in ("src/chat/engine.py", "src/chat/inference_backend.py"):
        text = Path(source).read_text().lower()
        assert "unsloth" not in text, f"{source} still mentions unsloth"
        assert "peft" not in text, f"{source} still mentions peft"
