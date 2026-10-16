"""Seed the poll-vote registry with a poll posted before vote tracking was on.

``whatsapp.poll_votes`` was off when the first polls went out (2026-10-15), so
those polls were never registered and their votes, all ``poll.vote.failed``
under NOWEB without its store, were only logged. This registers each poll from
``data/polls.jsonl`` and replays the logged failed votes (one
``WAHA reported a failed poll vote: {...}`` line each, cut at 300 characters)
as "voted, choice unknown", so the reminders skip whoever already voted.

Dry run by default: prints what it would record. ``--apply`` writes. Re-running
is safe (registering keeps the votes; a replayed vote is not stored twice).

Usage (inside the prod container, after the deploy):
  python scripts/seed_poll_votes.py --log data/poll_vote_failed_backfill.log
  python scripts/seed_poll_votes.py --log data/poll_vote_failed_backfill.log --apply
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from src.chat import poll_votes

# The logged payload is truncated, so read the fields by pattern, not as JSON.
_VOTE_FROM = re.compile(r'"vote": \{.*?"from": "([^"]+)"')
_VOTE_TO = re.compile(r'"vote": \{.*?"to": "([^"]+)"')
_VOTE_TS = re.compile(r'"vote": \{.*?"timestamp": (\d+)')
_POLL_ID = re.compile(r'"poll": \{"id": "([^"]+)')


def parse_log_line(line: str) -> Optional[poll_votes.PollVote]:
    """One logged failed vote, or None when the line lacks a field we need."""
    if "failed poll vote" not in line:
        return None
    voter, to, ts, poll = (_VOTE_FROM.search(line), _VOTE_TO.search(line),
                           _VOTE_TS.search(line), _POLL_ID.search(line))
    if not (voter and poll):
        return None
    chat = to.group(1) if to else ""
    return poll_votes.PollVote(
        poll_id=poll_votes.poll_key(poll.group(1)),
        chat_id=chat,
        voter_id=voter.group(1),
        options=[],
        timestamp=int(ts.group(1)) if ts else None,
        poll_from_me=True,
        vote_from_me=False,
        unknown=True,
    )


def seed(registry: poll_votes.PollRegistry, polls: Iterable[dict],
         log_lines: Iterable[str], apply: bool) -> List[str]:
    """Register the polls and replay the votes; returns what was (or would be) done."""
    report = []
    known = {}
    for poll in polls:
        key = poll_votes.poll_key(poll["poll_id"])
        known[key] = poll
        report.append(f"poll {key} in {poll['chat_id']}, created {poll['created_at']}, "
                      f"{len(poll.get('options') or [])} options")
        if apply:
            registry.register(poll["chat_id"], poll["poll_id"], list(poll.get("options") or []),
                              int(poll["created_at"]), source="explicit")
    for line in log_lines:
        vote = parse_log_line(line)
        if vote is None or vote.poll_id not in known:
            continue
        recorded = registry.record_vote(vote, "", int(vote.timestamp or 0)) if apply else None
        report.append(f"vote {vote.voter_id} at {vote.timestamp}"
                      + ("" if not apply else (" recorded" if recorded else " already there")))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--polls", default=str(BASE / "data" / "polls.jsonl"))
    parser.add_argument("--log", required=True, help="grep of the 'failed poll vote' log lines")
    parser.add_argument("--store", default=None, help="default: whatsapp.poll_votes.store")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    from src.config_loader import load_config
    config = load_config()
    store = Path(args.store or poll_votes.settings(config)["store"])
    if not store.is_absolute():
        store = BASE / store
    polls = [json.loads(line) for line in Path(args.polls).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    lines = Path(args.log).read_text(encoding="utf-8").splitlines()
    for row in seed(poll_votes.PollRegistry(store), polls, lines, args.apply):
        print(row)
    print(("written to " if args.apply else "dry run; would write ") + str(store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
