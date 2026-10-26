import pytest


@pytest.fixture(autouse=True)
def _enrichment_queue_in_tmp(tmp_path, monkeypatch):
    """The deferred-media queue (src/chat/enrichment.py) is never the live one."""
    monkeypatch.setenv("KAYA_ENRICHMENT_DIR", str(tmp_path / "enrichment"))
    monkeypatch.delenv("KAYA_GPU1_LENT", raising=False)
