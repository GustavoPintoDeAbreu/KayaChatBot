"""Process-wide GPU serialization for the web UI.

The box has a single GPU, so generation and RAG embedding must run one job at a
time. ``gpu_section`` is a context manager that callers wrap around any GPU work
(``model.generate``, ``SentenceTransformer.encode``); concurrent callers are
serialized by a shared ``BoundedSemaphore``. Gradio's own ``.queue()`` provides
fairness and queue position, but this lock is the hard guarantee — it also covers
the daemon generation thread and the second suggestions call, which the queue
alone does not.

Pure-Python (no torch/GPU import) so it is unit-testable without a GPU.
"""
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional

DEFAULT_MAX_CONCURRENT = 1
DEFAULT_ACQUIRE_TIMEOUT = 120.0


class GpuBusyError(Exception):
    """Raised when the GPU lock cannot be acquired within the timeout."""


_lock_instance: Optional[threading.BoundedSemaphore] = None
_lock_guard = threading.Lock()


def _concurrency_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("chat", {}).get("concurrency", {}) or {}


def get_gpu_lock(config: Dict[str, Any]) -> threading.BoundedSemaphore:
    """Return the process-wide GPU semaphore, creating it on first use.

    Sized by ``chat.concurrency.max_concurrent``. Mirrors the double-checked
    locking singleton used by ``get_retriever`` in retriever.py.
    """
    global _lock_instance
    if _lock_instance is None:
        with _lock_guard:
            if _lock_instance is None:
                max_concurrent = int(
                    _concurrency_config(config).get("max_concurrent", DEFAULT_MAX_CONCURRENT)
                )
                _lock_instance = threading.BoundedSemaphore(max(1, max_concurrent))
    return _lock_instance


@contextmanager
def gpu_section(config: Dict[str, Any], timeout: Optional[float] = None):
    """Serialize a block of GPU work behind the shared semaphore.

    Acquires the lock (blocking up to ``timeout`` seconds, defaulting to
    ``chat.concurrency.acquire_timeout``) and always releases it on exit, even if
    the wrapped body raises mid-stream. Raises ``GpuBusyError`` if the lock cannot
    be acquired in time.
    """
    if timeout is None:
        timeout = float(_concurrency_config(config).get("acquire_timeout", DEFAULT_ACQUIRE_TIMEOUT))

    lock = get_gpu_lock(config)
    acquired = lock.acquire(timeout=timeout)
    if not acquired:
        raise GpuBusyError(
            f"GPU busy: could not acquire the generation lock within {timeout:.0f}s."
        )
    try:
        yield
    finally:
        lock.release()


def reset_gpu_lock() -> None:
    """Drop the cached semaphore so the next ``get_gpu_lock`` rebuilds it.

    Intended for tests that vary ``max_concurrent`` between cases.
    """
    global _lock_instance
    with _lock_guard:
        _lock_instance = None
