"""The review step: the only thing standing between a model and the profiles.

Accepting rewrites what the bot says about real people, so the destructive paths
are the ones worth pinning down — that a reject changes nothing, that a snapshot
always survives, and that the evidence shown is actually the evidence.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "review_bios", ROOT / "scripts" / "review_bios.py")
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


MEMBERS = {"members": [
    {"name": "Rafa", "key_facts": ["Rafa hosts the group."]},
    {"name": "Gil", "key_facts": ["Gil runs."]},
]}
PROPOSALS = {
    "unresolved_senders": {"Ricardo": 1},
    "proposals": {
        "Rafa": {"current": ["Rafa hosts the group."],
                 "proposed": ["Rafa hosts the group.", "Rafa trains kickboxing."],
                 "added": ["Rafa trains kickboxing."], "removed": [],
                 "new_facts": [{"fact": "Rafa trains kickboxing.", "evidence": ["m1", "m2"],
                                "quote": "comecei kickboxing com o Mateus"}]},
        "Gil": {"current": ["Gil runs."], "proposed": ["Gil runs.", "Gil owns a dog."],
                "added": ["Gil owns a dog."], "removed": [],
                "new_facts": [{"fact": "Gil owns a dog.", "evidence": ["m1"],
                               "quote": "tenho um cão novo"}]},
    },
}


@pytest.fixture
def data(tmp_path):
    (tmp_path / "live_messages").mkdir()
    (tmp_path / "group_members.json").write_text(json.dumps(MEMBERS), encoding="utf-8")
    (tmp_path / "bio_proposals.json").write_text(json.dumps(PROPOSALS), encoding="utf-8")
    (tmp_path / "live_messages" / "shared.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "m1", "sender": "Rafa", "text": "comecei kickboxing com o Mateus"},
        {"id": "m2", "sender": "Gil", "text": "boas malta tudo bem"},
    ]), encoding="utf-8")
    return tmp_path


def run(data, *argv):
    sys.argv = ["review_bios.py", "--data-dir", str(data), *argv]
    return review.main()


def members_of(data):
    doc = json.loads((data / "group_members.json").read_text(encoding="utf-8"))
    return {m["name"]: m for m in doc["members"]}


def pending(data):
    return json.loads((data / "bio_proposals.json").read_text(encoding="utf-8"))["proposals"]


# ── accept ───────────────────────────────────────────────────────────────────
def test_accepting_writes_the_proposed_facts(data):
    run(data, "--accept", "Rafa")

    assert members_of(data)["Rafa"]["key_facts"] == [
        "Rafa hosts the group.", "Rafa trains kickboxing."]


def test_accepting_clears_only_that_proposal(data):
    run(data, "--accept", "Rafa")

    assert list(pending(data)) == ["Gil"]


def test_accepting_leaves_a_snapshot_to_go_back_to(data):
    run(data, "--accept", "Rafa")

    snapshots = list((data / "profile_snapshots").glob("*.json"))
    assert len(snapshots) == 1
    restored = json.loads(snapshots[0].read_text(encoding="utf-8"))
    assert [m for m in restored["members"] if m["name"] == "Rafa"][0]["key_facts"] == \
        ["Rafa hosts the group."]


def test_two_accepts_in_the_same_second_keep_both_snapshots(data):
    """Otherwise the second overwrites the first and the original is gone."""
    run(data, "--accept", "Rafa")
    run(data, "--accept", "Gil")

    assert len(list((data / "profile_snapshots").glob("*.json"))) == 2


def test_accept_all_applies_everything(data):
    run(data, "--accept", "all")

    assert "Rafa trains kickboxing." in members_of(data)["Rafa"]["key_facts"]
    assert "Gil owns a dog." in members_of(data)["Gil"]["key_facts"]
    assert pending(data) == {}


# ── reject ───────────────────────────────────────────────────────────────────
def test_rejecting_changes_no_profile(data):
    before = (data / "group_members.json").read_text(encoding="utf-8")

    run(data, "--reject", "Rafa")

    assert (data / "group_members.json").read_text(encoding="utf-8") == before
    assert list(pending(data)) == ["Gil"]


# ── pin ──────────────────────────────────────────────────────────────────────
def test_pinning_records_and_promotes_the_fact(data):
    run(data, "--member", "Gil", "--pin", "Gil works in sales.")

    gil = members_of(data)["Gil"]
    assert gil["pinned_facts"] == ["Gil works in sales."]
    assert gil["key_facts"][0] == "Gil works in sales."


def test_pinning_needs_a_real_member(data):
    assert run(data, "--member", "Nobody", "--pin", "x") == 1


# ── the evidence shown ───────────────────────────────────────────────────────
def test_the_quote_is_shown_verbatim(data, capsys):
    """The extractor now copies the supporting words out of the messages and
    they are verified before the proposal is written, so the review shows the
    real thing. It used to rank the chunk's messages by word overlap, which
    handed an invented fact an unrelated line as its "evidence"."""
    run(data, "--member", "Rafa")

    assert "comecei kickboxing com o Mateus" in capsys.readouterr().out


def test_a_proposal_with_no_quote_is_flagged(data, capsys):
    """Written before quotes were required — say so rather than showing nothing."""
    document = json.loads((data / "bio_proposals.json").read_text(encoding="utf-8"))
    document["proposals"]["Rafa"]["new_facts"][0].pop("quote", None)
    (data / "bio_proposals.json").write_text(json.dumps(document), encoding="utf-8")

    run(data, "--member", "Rafa")

    assert "no quote recorded" in capsys.readouterr().out


def test_nothing_pending_is_not_an_error(tmp_path):
    (tmp_path / "group_members.json").write_text(json.dumps(MEMBERS), encoding="utf-8")

    assert run(tmp_path) == 0


def test_a_missing_profile_file_is_reported(tmp_path):
    assert run(tmp_path) == 1
