"""Unit tests for src/chat/feedback.py — ratings, reasons, bug reports, aggregation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import feedback


def test_log_rating_roundtrip(tmp_path):
    log = tmp_path / "fb.jsonl"
    fid = feedback.log_rating(
        source="web", rating="down", user_message="Q", assistant_response="A",
        interaction_id="iid-1", path=log,
    )
    rows = feedback.load_feedback(log)
    assert len(rows) == 1
    r = rows[0]
    assert r["feedback_id"] == fid
    assert r["type"] == "rating"
    assert r["rating"] == "down"
    assert r["interaction_id"] == "iid-1"
    assert "timestamp" in r


def test_log_rating_returns_unique_ids(tmp_path):
    log = tmp_path / "fb.jsonl"
    a = feedback.log_rating(source="web", rating="up", path=log)
    b = feedback.log_rating(source="web", rating="up", path=log)
    assert a != b


def test_comment_joins_rating(tmp_path):
    log = tmp_path / "fb.jsonl"
    fid = feedback.log_rating(source="web", rating="down", user_message="Q", path=log)
    feedback.log_comment(feedback_id=fid, source="web", comment="resposta errada", path=log)
    agg = feedback.aggregate_feedback(feedback_path=log, bug_path=tmp_path / "none.jsonl")
    assert agg["down"] == 1
    assert agg["recent_down"][0]["reason"] == "resposta errada"


def test_empty_comment_is_dropped(tmp_path):
    log = tmp_path / "fb.jsonl"
    feedback.log_comment(feedback_id="x", source="web", comment="   ", path=log)
    feedback.log_comment(feedback_id="", source="web", comment="hello", path=log)
    assert feedback.load_feedback(log) == []


def test_extra_fields_recorded(tmp_path):
    log = tmp_path / "fb.jsonl"
    feedback.log_rating(source="whatsapp", rating="up", is_group=True, path=log)
    r = feedback.load_feedback(log)[0]
    assert r["source"] == "whatsapp"
    assert r["is_group"] is True


def test_bug_report_roundtrip(tmp_path):
    bug = tmp_path / "bug.jsonl"
    rid = feedback.log_bug_report(
        source="web", description="página partida", contact="gu",
        env="dev", version="abc123", recent_turns=["User: oi"], path=bug,
    )
    rows = feedback.load_bug_reports(bug)
    assert len(rows) == 1
    r = rows[0]
    assert r["report_id"] == rid
    assert r["description"] == "página partida"
    assert r["contact"] == "gu"
    assert r["recent_turns"] == ["User: oi"]


def test_aggregate_counts_by_source(tmp_path):
    log = tmp_path / "fb.jsonl"
    bug = tmp_path / "bug.jsonl"
    feedback.log_rating(source="web", rating="up", path=log)
    feedback.log_rating(source="web", rating="down", path=log)
    feedback.log_rating(source="whatsapp", rating="up", path=log)
    feedback.log_bug_report(source="web", description="b", path=bug)
    agg = feedback.aggregate_feedback(feedback_path=log, bug_path=bug)
    assert agg["total_ratings"] == 3
    assert agg["up"] == 2 and agg["down"] == 1
    assert agg["by_source"]["web:up"] == 1
    assert agg["by_source"]["web:down"] == 1
    assert agg["by_source"]["whatsapp:up"] == 1
    assert agg["bug_total"] == 1


def test_logging_never_raises(tmp_path):
    # A bad path must not propagate an exception to the caller.
    feedback.log_rating(source="web", rating="up", path=tmp_path / "a" / "b" / "c.jsonl")
    feedback.log_bug_report(source="web", description="x", path=tmp_path / "a" / "b" / "d.jsonl")


def test_config_path_resolution():
    cfg = {"chat": {"feedback": {"log_file": "/tmp/x.jsonl"}, "bug_report": {"log_file": "rel/y.jsonl"}}}
    assert feedback.feedback_log_path(cfg) == Path("/tmp/x.jsonl")
    # Relative paths resolve against the repo root.
    assert feedback.bug_log_path(cfg).name == "y.jsonl"
    assert feedback.bug_log_path(cfg).is_absolute()
