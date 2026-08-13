"""Borrowing GPU0 must never lose somebody's picture.

Image generation is the only thing on GPU0 — serving, Whisper and vision all sit
on the other card — so pausing it frees 24GB while the bot keeps answering. That
is what lets heavy GPU work run without an outage.

Two properties matter. A paused queue HOLDS, so nothing is dropped; and a lease
EXPIRES, so a crashed maintenance job cannot leave image generation dead.
"""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import imagegen


@pytest.fixture
def config(tmp_path):
    return {"chat": {"imagegen": {"enabled": True,
                                  "lease_file": str(tmp_path / "gpu0_lease.json")}}}


# ── the lease ────────────────────────────────────────────────────────────────
def test_no_lease_means_not_paused(config):
    assert imagegen.is_paused(config) is False
    assert imagegen.pause_remaining(config) == 0


def test_acquiring_pauses(config):
    imagegen.acquire_lease(config, "regenerating profiles", 600)

    assert imagegen.is_paused(config) is True
    assert imagegen.read_lease(config)["reason"] == "regenerating profiles"


def test_releasing_resumes(config):
    imagegen.acquire_lease(config, "bake-off", 600)
    imagegen.release_lease(config)

    assert imagegen.is_paused(config) is False


def test_an_expired_lease_resumes_by_itself(config):
    """The safety property. A crashed job must not pause images forever."""
    imagegen.acquire_lease(config, "job that died", -1)

    assert imagegen.is_paused(config) is False


def test_releasing_twice_is_not_an_error(config):
    imagegen.acquire_lease(config, "x", 600)
    assert imagegen.release_lease(config) is True
    assert imagegen.release_lease(config) is True


def test_an_unreadable_lease_counts_as_free(config, tmp_path):
    """Failing open: a corrupt file must not silently stop image generation."""
    Path(config["chat"]["imagegen"]["lease_file"]).write_text("{not json",
                                                             encoding="utf-8")

    assert imagegen.is_paused(config) is False


def test_the_remaining_wait_is_reported_in_whole_minutes(config):
    imagegen.acquire_lease(config, "x", 605)

    assert imagegen.pause_remaining(config) == 11  # 10m05s rounds up


def test_a_lease_shorter_than_a_minute_still_reports_one(config):
    """Saying "0 min" then not delivering is worse than rounding up."""
    imagegen.acquire_lease(config, "x", 20)

    assert imagegen.pause_remaining(config) == 1


# ── the queue ────────────────────────────────────────────────────────────────
def _wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_a_paused_queue_holds_the_job_instead_of_running_it(config):
    ran = []
    queue = imagegen.ImageQueue(config=config)
    imagegen.acquire_lease(config, "maintenance", 600)

    assert queue.submit(lambda: ran.append(1)) == 1
    time.sleep(0.5)

    assert ran == []
    assert queue.depth == 1, "the job keeps its place, so the count stays honest"


def test_release_drains_what_was_held(config):
    ran = []
    queue = imagegen.ImageQueue(config=config)
    imagegen.acquire_lease(config, "maintenance", 600)
    queue.submit(lambda: ran.append(1))
    time.sleep(0.3)
    assert ran == []

    imagegen.release_lease(config)

    assert _wait_until(lambda: ran == [1]), "the held job must run on release"


def test_an_expired_lease_drains_without_anyone_releasing(config):
    ran = []
    queue = imagegen.ImageQueue(config=config)
    imagegen.acquire_lease(config, "job that died", 1)
    queue.submit(lambda: ran.append(1))

    assert _wait_until(lambda: ran == [1], timeout=8.0)


def test_the_queue_still_declines_past_its_cap_while_paused(config):
    """Past the limit the bot says no, which is honest — pausing does not turn
    the bounded queue into an unbounded one."""
    queue = imagegen.ImageQueue(maxsize=2, config=config)
    imagegen.acquire_lease(config, "maintenance", 600)

    assert queue.submit(lambda: None) == 1
    assert queue.submit(lambda: None) == 2
    assert queue.submit(lambda: None) is None


def test_an_unpaused_queue_runs_normally(config):
    ran = []
    queue = imagegen.ImageQueue(config=config)

    queue.submit(lambda: ran.append(1))

    assert _wait_until(lambda: ran == [1])


def test_a_queue_with_no_config_is_never_paused():
    """The process-wide queue is built at import, before any config exists."""
    ran = []
    queue = imagegen.ImageQueue()

    queue.submit(lambda: ran.append(1))

    assert _wait_until(lambda: ran == [1])
