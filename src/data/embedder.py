"""The one way Kaya loads her embedding model (bge-m3 for RAG, documents and ingest).

Pinned and offline-first, since 2026-11-16:

- `rag.embedding_revision` in config.yaml names the exact Hugging Face commit. A commit hash
  cannot be moved, so a compromised or changed model repo cannot reach Kaya through a restart.
- It loads from the local cache only (`local_files_only`): no request to huggingface.co at
  start-up. Only when that commit is not cached yet (a fresh machine) does it download it,
  still pinned to the same commit.
- No `trust_remote_code`: bge-m3 is a plain XLM-RoBERTa model and needs no code from the repo.
  A model that does need it is a deliberate change here, not a flag on a call site.

Why: torch 2.6 (pinned for CUDA 12.4) has CVE-2026-24747, memory corruption in the
`weights_only` unpickler that bge-m3's `pytorch_model.bin` goes through. The file is only
dangerous if someone else can replace it; pinning the commit and reading the cache closes that.
The embeddings are bit-identical to the unpinned load (checked on the prod cache, 2026-11-16).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"


def embedder_settings(config: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """model, revision and device from the config's `rag` section."""
    rag = (config or {}).get("rag", {}) or {}
    return {
        "model": rag.get("embedding_model") or DEFAULT_MODEL,
        "revision": rag.get("embedding_revision") or None,
        "device": rag.get("embedding_device") or None,
    }


def load_encoder(model: str, revision: Optional[str] = None, device: Optional[str] = None):
    """A SentenceTransformer for `model` at `revision`, from the local cache when it is there."""
    from sentence_transformers import SentenceTransformer

    if not revision:
        logger.warning("embedder %s has no pinned revision (rag.embedding_revision): "
                       "it follows the repo's main branch", model)
    try:
        return SentenceTransformer(model, revision=revision, device=device, local_files_only=True)
    except OSError as exc:  # not in the cache yet: fetch that same pinned commit once
        logger.info("embedder %s@%s not cached (%s); downloading it", model, revision or "main", exc)
        return SentenceTransformer(model, revision=revision, device=device)


def load_encoder_from_config(config: Dict[str, Any], model: Optional[str] = None,
                             device: Optional[str] = None):
    """`load_encoder` with the config's settings.

    `model` and `device` override `rag.embedding_model` / `rag.embedding_device`; the pinned
    revision belongs to the configured model, so another model is loaded unpinned.
    """
    settings = embedder_settings(config)
    name = model or settings["model"]
    revision = settings["revision"] if name == settings["model"] else None
    return load_encoder(name, revision, device if device is not None else settings["device"])
