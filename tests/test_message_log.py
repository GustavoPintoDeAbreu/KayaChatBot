"""The log is the group's memory, so a damaged line must damage only itself.

One line of the live group log was found holding 159 characters of one record
followed by the whole of the next:

    ..."timestamp": 1787737868, "{"id": "2c914ed2e7...", "chat_id": ...

The first record had been cut off mid-write with no trailing newline. The next
append then continued that same line, and because `read` skips anything it
cannot parse, BOTH messages were lost — the truncated one and the perfectly good
one that landed behind it.

Losing the truncated record is unavoidable; it was never fully written. Losing
its neighbour is not, and that is what these tests are about.
"""
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.message_log import MessageLog


def _lines(path):
    good, bad = [], 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return good, bad


def test_a_truncated_line_does_not_swallow_the_next_message(tmp_path):
    """The exact live failure, reproduced."""
    log = MessageLog(base_dir=str(tmp_path))
    path = tmp_path / "shared.jsonl"

    log.append(chat_id="g@g.us", message_id="A", sender="Rafa", text="primeira",
               timestamp=1700000000, scope="shared")
    # A write cut short mid-flush: valid JSON so far, no trailing newline.
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"id": "574ea1c0", "sender": "Pedro", "text": "Please go to the')
    log.append(chat_id="g@g.us", message_id="B", sender="Gil", text="a seguir",
               timestamp=1700000002, scope="shared")

    good, bad = _lines(path)
    assert [record["text"] for record in good] == ["primeira", "a seguir"]
    assert bad == 1, "only the genuinely truncated record should be unreadable"


def test_a_normal_append_adds_no_blank_lines(tmp_path):
    """The guard must not leave a stray newline behind on the happy path."""
    log = MessageLog(base_dir=str(tmp_path))
    for index in range(5):
        log.append(chat_id="g@g.us", message_id=f"M{index}", sender="Rafa",
                   text=f"mensagem {index}", timestamp=1700000000 + index,
                   scope="shared")

    raw = (tmp_path / "shared.jsonl").read_text(encoding="utf-8")
    assert raw.count("\n\n") == 0
    good, bad = _lines(tmp_path / "shared.jsonl")
    assert len(good) == 5 and bad == 0


def test_the_first_append_to_a_new_file_is_unaffected(tmp_path):
    log = MessageLog(base_dir=str(tmp_path))
    assert log.append(chat_id="g@g.us", message_id="A", sender="Rafa",
                      text="primeira", timestamp=1700000000, scope="shared")

    good, bad = _lines(tmp_path / "shared.jsonl")
    assert len(good) == 1 and bad == 0


def test_concurrent_appends_all_survive(tmp_path):
    """`append` is serialised; every message must land, exactly once, intact."""
    log = MessageLog(base_dir=str(tmp_path))
    count = 120

    def write(index):
        log.append(chat_id="g@g.us", message_id=f"M{index}", sender=f"S{index}",
                   text="x" * 500 + str(index), timestamp=1700000000 + index,
                   scope="shared", reply_to_id=f"R{index}", reply_to_text="y" * 200)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    good, bad = _lines(tmp_path / "shared.jsonl")
    assert bad == 0
    assert len({record["id"] for record in good}) == count


def test_the_same_message_is_logged_once_under_concurrency(tmp_path):
    """WAHA delivers everything twice, so this pair of threads is the normal case."""
    log = MessageLog(base_dir=str(tmp_path))
    barrier = threading.Barrier(16)

    def write():
        barrier.wait()
        log.append(chat_id="g@g.us", message_id="SAME", sender="Rafa", text="olá",
                   timestamp=1700000000, scope="shared")

    threads = [threading.Thread(target=write) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    good, bad = _lines(tmp_path / "shared.jsonl")
    assert len(good) == 1 and bad == 0
