"""The embedder is pinned to one Hugging Face commit, read from the cache, and runs no repo code."""
import re
import sys
import types
from pathlib import Path

import pytest

from src.config_loader import load_config
from src.data import embedder

ROOT = Path(__file__).resolve().parent.parent
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


@pytest.fixture
def calls(monkeypatch):
    made = []

    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs):
            made.append((name, kwargs))
            if kwargs.get("local_files_only") and name == "not/cached":
                raise OSError("couldn't find them in the cached files")

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    return made


def test_loads_the_pinned_commit_from_the_cache_without_remote_code(calls):
    embedder.load_encoder("BAAI/bge-m3", REVISION, "cuda")
    assert calls == [("BAAI/bge-m3", {"revision": REVISION, "device": "cuda",
                                      "local_files_only": True})]


def test_an_uncached_model_downloads_the_same_commit(calls):
    embedder.load_encoder("not/cached", REVISION)
    assert [kwargs.get("local_files_only") for _, kwargs in calls] == [True, None]
    assert all(kwargs["revision"] == REVISION for _, kwargs in calls)
    assert not any("trust_remote_code" in kwargs for _, kwargs in calls)


def test_the_revision_belongs_to_the_configured_model_only(calls):
    config = {"rag": {"embedding_model": "BAAI/bge-m3", "embedding_revision": REVISION,
                      "embedding_device": "cpu"}}
    embedder.load_encoder_from_config(config)
    embedder.load_encoder_from_config(config, model="other/model", device="cuda")
    assert calls[0] == ("BAAI/bge-m3", {"revision": REVISION, "device": "cpu",
                                        "local_files_only": True})
    assert calls[1] == ("other/model", {"revision": None, "device": "cuda",
                                        "local_files_only": True})


def test_config_pins_the_embedder_to_a_commit():
    rag = load_config(str(ROOT / "config.yaml"))["rag"]
    assert rag["embedding_model"] == "BAAI/bge-m3"
    assert re.fullmatch(r"[0-9a-f]{40}", rag["embedding_revision"]), rag["embedding_revision"]


def test_nothing_in_the_app_loads_an_embedder_another_way():
    allowed = {ROOT / "src" / "data" / "embedder.py"}
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if path in allowed or "testing" in path.parts or path.name == "local_teacher.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "SentenceTransformer(" in text or "trust_remote_code=True" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], offenders
