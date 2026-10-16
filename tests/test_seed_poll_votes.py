"""scripts/seed_poll_votes.py: registering a pre-tracking poll and replaying its logged votes."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import poll_votes

_spec = importlib.util.spec_from_file_location(
    "seed_poll_votes", Path(__file__).parent.parent / "scripts" / "seed_poll_votes.py")
seed_poll_votes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed_poll_votes)

CHAT = "120363000000000000@g.us"
POLL = {"poll_id": "true_" + CHAT + "_3EB08ECDE9484D0A232894", "chat_id": CHAT,
        "options": ["Buzio", "Casa Ideal"], "created_at": 1792054339}
# A real line's shape, cut at 300 characters by the adapter's log call.
LINE = ('WAHA reported a failed poll vote: {"vote": {"id": "false_' + CHAT
        + '_AC3FCF6873891778F0D521FA408307E6_184065839865899@lid", "to": "' + CHAT
        + '", "from": "184065839865899@lid", "fromMe": false, "selectedOptions": [], '
        '"timestamp": 1792055501}, "poll": {"id": "true_' + CHAT + '_3EB08ECDE9484D0A232894_')


def test_parse_log_line_reads_a_truncated_line():
    vote = seed_poll_votes.parse_log_line(LINE)
    assert vote.voter_id == "184065839865899@lid" and vote.chat_id == CHAT
    assert vote.poll_id == "3EB08ECDE9484D0A232894" and vote.timestamp == 1792055501
    assert vote.unknown is True
    assert seed_poll_votes.parse_log_line("something else") is None


def test_dry_run_writes_nothing(tmp_path):
    store = tmp_path / "votes.json"
    report = seed_poll_votes.seed(poll_votes.PollRegistry(store), [POLL], [LINE], apply=False)
    assert not store.exists() and len(report) == 2


def test_apply_registers_the_poll_and_replays_votes_once(tmp_path):
    registry = poll_votes.PollRegistry(tmp_path / "votes.json")
    seed_poll_votes.seed(registry, [POLL], [LINE, LINE], apply=True)
    poll = registry.get("3EB08ECDE9484D0A232894")
    assert poll["created_at"] == 1792054339 and poll["source"] == "explicit"
    assert list(poll["votes"]) == ["184065839865899"]
    report = seed_poll_votes.seed(registry, [POLL], [LINE], apply=True)
    assert report[-1].endswith("already there")
