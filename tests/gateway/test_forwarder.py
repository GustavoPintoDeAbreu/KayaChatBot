"""Forwarder: in-order, at-least-once delivery from the journal to the PC."""
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.forwarder import Forwarder
from src.gateway.journal import Journal
from src.gateway.monitor import PcMonitor, PcState


class Clock:
    def __init__(self, start: float = 10_000.0):
        self.value = start

    def __call__(self) -> float:
        return self.value


def _journal(tmp_path) -> Journal:
    return Journal(str(tmp_path / "journal.sqlite3"), str(tmp_path / "media"))


def _append(journal: Journal, key: str, *, received_at: float, sender: str = "a@c.us",
            chat: str = "group@g.us", addressed: bool = False, backlog: bool = False) -> int:
    event = {"event": "message", "payload": {"id": key, "from": chat, "body": key}}
    return journal.append(event, dedup_key=key, event_type="message", chat_id=chat,
                          sender_id=sender, message_id=key, wa_ts=int(received_at),
                          received_at=received_at, addressed=addressed, backlog=backlog)


def _online_monitor() -> PcMonitor:
    monitor = PcMonitor("http://pc:7860", probe=lambda: PcState.ONLINE)
    monitor.observe(PcState.ONLINE)
    return monitor


def _forwarder(journal, handler, clock, monitor=None) -> Forwarder:
    async def no_sleep(_seconds):
        return None

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Forwarder(journal, monitor or _online_monitor(), pc_url="http://pc:7860",
                     relay_token="secret", media_base_url="http://pi:8088",
                     client=client, now=clock, sleep=no_sleep)


def _acking(received):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.headers["X-Relay-Token"] == "secret"
        received.append(body)
        return httpx.Response(200, json={"ack": body["kaya_relay"]["seq"]})
    return handler


async def _drain(forwarder: Forwarder, journal: Journal) -> None:
    for _ in range(20):
        entry = journal.next_pending()
        if entry is None:
            return
        await forwarder.deliver_one(entry)


def test_delivers_in_order_with_the_backlog_count(tmp_path):
    journal, clock, received = _journal(tmp_path), Clock(), []
    for index in range(3):
        _append(journal, f"m{index}", received_at=clock.value)
    asyncio.run(_drain(_forwarder(journal, _acking(received), clock), journal))
    assert [body["kaya_relay"]["seq"] for body in received] == [1, 2, 3]
    assert [body["kaya_relay"]["backlog_remaining"] for body in received] == [2, 1, 0]
    assert journal.pending_count() == 0


def test_a_failure_retries_the_same_seq(tmp_path):
    journal, clock, calls = _journal(tmp_path), Clock(), []

    def flaky(request):
        calls.append(json.loads(request.content)["kaya_relay"]["seq"])
        if len(calls) == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"ack": calls[-1]})

    _append(journal, "m0", received_at=clock.value)
    forwarder = _forwarder(journal, flaky, clock)
    first = asyncio.run(forwarder.deliver_one(journal.next_pending()))
    assert not first.delivered and journal.get(1).attempts == 1
    assert asyncio.run(forwarder.deliver_one(journal.next_pending())).delivered
    assert calls == [1, 1]


def test_a_wrong_ack_is_not_a_delivery(tmp_path):
    journal, clock = _journal(tmp_path), Clock()
    _append(journal, "m0", received_at=clock.value)
    forwarder = _forwarder(journal, lambda request: httpx.Response(200, json={"ack": 99}), clock)
    assert not asyncio.run(forwarder.deliver_one(journal.next_pending())).delivered
    assert journal.pending_count() == 1


def test_stored_media_is_served_by_the_pi(tmp_path):
    journal, clock = _journal(tmp_path), Clock()
    seq = _append(journal, "m0", received_at=clock.value)
    journal.attach_media(seq, b"ogg", "voice.ogg", "audio/ogg")
    envelope = _forwarder(journal, _acking([]), clock).envelope(journal.get(seq))
    assert envelope["event"]["payload"]["media"]["url"] == f"http://pi:8088/media/{seq}/voice.ogg"


def test_deferred_reply_only_for_the_newest_late_mention(tmp_path):
    journal, clock = _journal(tmp_path), Clock()
    old = _append(journal, "old", received_at=clock.value - 3600, addressed=True)
    newer = _append(journal, "newer", received_at=clock.value - 1800, addressed=True)
    other = _append(journal, "other", received_at=clock.value - 1800, addressed=True,
                    sender="b@c.us")
    fresh = _append(journal, "fresh", received_at=clock.value, addressed=True, sender="c@c.us")
    backlog = _append(journal, "wa-backlog", received_at=clock.value - 3600, addressed=True,
                      sender="d@c.us", backlog=True)
    forwarder = _forwarder(journal, _acking([]), clock)

    def deferred(seq):
        return forwarder.envelope(journal.get(seq))["kaya_relay"]["deferred_reply"]

    assert not deferred(old)
    assert deferred(newer)
    assert deferred(other)
    assert not deferred(fresh)
    assert not deferred(backlog)


def test_run_waits_while_the_pc_is_offline(tmp_path):
    journal, clock, received = _journal(tmp_path), Clock(), []
    _append(journal, "m0", received_at=clock.value)
    monitor = PcMonitor("http://pc:7860", now=clock)
    monitor.observe(PcState.OFFLINE)
    forwarder = _forwarder(journal, _acking(received), clock, monitor)

    async def scenario():
        stop = asyncio.Event()
        task = asyncio.create_task(forwarder.run(stop))
        await asyncio.sleep(0.05)
        assert received == []
        monitor.observe(PcState.ONLINE)
        forwarder.wake()
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.02)
        stop.set()
        forwarder.wake()
        await asyncio.wait_for(task, timeout=6)

    asyncio.run(scenario())
    assert [body["kaya_relay"]["seq"] for body in received] == [1]
