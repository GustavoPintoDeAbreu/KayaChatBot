"""Unit tests for src/chat/gpu_lock — GPU request serialization.

Threading-only, no GPU/torch: validates that gpu_section serializes concurrent
callers, bounds concurrency, times out into GpuBusyError, and releases on error.
"""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.gpu_lock import gpu_section, get_gpu_lock, reset_gpu_lock, GpuBusyError


def _config(max_concurrent=1, acquire_timeout=5.0):
    return {
        "chat": {
            "concurrency": {
                "max_concurrent": max_concurrent,
                "acquire_timeout": acquire_timeout,
            }
        }
    }


@pytest.fixture(autouse=True)
def _fresh_lock():
    """Each test starts with a fresh semaphore so max_concurrent can vary."""
    reset_gpu_lock()
    yield
    reset_gpu_lock()


class TestMutualExclusion:
    def test_never_exceeds_max_concurrent(self):
        config = _config(max_concurrent=2)
        state = {"current": 0, "peak": 0}
        guard = threading.Lock()

        def worker():
            with gpu_section(config):
                with guard:
                    state["current"] += 1
                    state["peak"] = max(state["peak"], state["current"])
                time.sleep(0.05)
                with guard:
                    state["current"] -= 1

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state["peak"] <= 2
        assert state["peak"] >= 2  # with 8 threads, both slots should be used

    def test_single_slot_serializes(self):
        config = _config(max_concurrent=1)
        intervals = []
        guard = threading.Lock()

        def worker():
            with gpu_section(config):
                start = time.monotonic()
                time.sleep(0.05)
                end = time.monotonic()
            with guard:
                intervals.append((start, end))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No two execution windows may overlap when only one slot exists.
        intervals.sort()
        for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:]):
            assert next_start >= prev_end - 1e-3


class TestTimeout:
    def test_busy_raises_gpu_busy_error(self):
        config = _config(max_concurrent=1, acquire_timeout=0.1)
        held = threading.Event()
        release = threading.Event()

        def holder():
            with gpu_section(config):
                held.set()
                release.wait(timeout=2.0)

        t = threading.Thread(target=holder)
        t.start()
        assert held.wait(timeout=1.0)

        with pytest.raises(GpuBusyError):
            with gpu_section(config):
                pass

        release.set()
        t.join()

    def test_explicit_timeout_overrides_config(self):
        config = _config(max_concurrent=1, acquire_timeout=999)
        held = threading.Event()
        release = threading.Event()

        def holder():
            with gpu_section(config):
                held.set()
                release.wait(timeout=2.0)

        t = threading.Thread(target=holder)
        t.start()
        assert held.wait(timeout=1.0)

        with pytest.raises(GpuBusyError):
            with gpu_section(config, timeout=0.1):
                pass

        release.set()
        t.join()


class TestRelease:
    def test_releases_on_exception(self):
        config = _config(max_concurrent=1, acquire_timeout=0.5)

        with pytest.raises(ValueError):
            with gpu_section(config):
                raise ValueError("boom")

        # Lock must be free again — this would raise GpuBusyError if it leaked.
        with gpu_section(config):
            pass


class TestSingleton:
    def test_same_instance(self):
        config = _config()
        assert get_gpu_lock(config) is get_gpu_lock(config)

    def test_reset_rebuilds(self):
        config = _config()
        first = get_gpu_lock(config)
        reset_gpu_lock()
        assert get_gpu_lock(config) is not first
