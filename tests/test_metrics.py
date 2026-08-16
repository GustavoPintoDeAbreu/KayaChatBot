"""Unit tests for src/chat/metrics.py — logging + aggregation, no model/network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import metrics


def test_log_and_load_roundtrip(tmp_path):
    log = tmp_path / "interactions.jsonl"
    metrics.log_interaction(
        source="web", user_message="oi", assistant_response="olá tudo bem", latency_ms=123.4, path=log
    )
    rows = metrics.load_interactions(log)
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "web"
    assert r["response_words"] == 3
    assert r["response_chars"] == len("olá tudo bem")
    assert r["latency_ms"] == 123.4
    assert "interaction_id" in r and "timestamp" in r


def test_extra_fields_are_recorded(tmp_path):
    log = tmp_path / "i.jsonl"
    metrics.log_interaction(
        source="whatsapp", user_message="q", assistant_response="a",
        web_search_used=True, is_group=True, path=log,
    )
    r = metrics.load_interactions(log)[0]
    assert r["web_search_used"] is True
    assert r["is_group"] is True


def test_aggregate_computes_expected(tmp_path):
    log = tmp_path / "i.jsonl"
    metrics.log_interaction(source="web", user_message="a", assistant_response="um dois três", latency_ms=100, path=log)
    metrics.log_interaction(source="whatsapp", user_message="b", assistant_response="quatro cinco", latency_ms=300, path=log)
    metrics.log_interaction(source="web", user_message="c", assistant_response="seis", latency_ms=200, web_search_used=True, path=log)
    agg = metrics.aggregate(log)
    assert agg["total"] == 3
    assert agg["by_source"] == {"web": 2, "whatsapp": 1}
    assert agg["avg_latency_ms"] == 200.0
    assert agg["avg_response_words"] == 2.0  # (3 + 2 + 1) / 3
    assert agg["web_search_rate"] == round(1 / 3, 3)


def test_aggregate_empty(tmp_path):
    assert metrics.aggregate(tmp_path / "nope.jsonl") == {
        "total": 0, "by_source": {}, "avg_response_words": 0.0,
        "avg_response_chars": 0.0, "avg_latency_ms": 0.0, "web_search_rate": 0.0, "per_day": {},
    }


def test_logging_never_raises(tmp_path):
    # A bad path must not propagate an exception to the caller.
    metrics.log_interaction(source="web", user_message="x", assistant_response="y", path=tmp_path / "a" / "b" / "c.jsonl")


def test_malformed_lines_skipped(tmp_path):
    log = tmp_path / "i.jsonl"
    log.write_text('{"source":"web","assistant_response":"ok"}\nnot-json\n\n', encoding="utf-8")
    rows = metrics.load_interactions(log)
    assert len(rows) == 1



# ── what gets logged at all ──────────────────────────────────────────────────
# Every result carrying a `command` was excluded, and `_handle_image_request`
# always sets one. So the edit the group complained about left no record: not the
# prompt it was given, not the editor, not the outcome. The exclusion is meant
# for bookkeeping confirmations ("volto a responder por texto"), not for a real
# turn that happens to produce a picture.
def test_an_image_turn_is_logged():
    assert metrics.should_log({"reply": "vou fazer isso", "command": "image"})


def test_bookkeeping_confirmations_are_still_excluded():
    for command in sorted(metrics.BOOKKEEPING_COMMANDS):
        assert not metrics.should_log({"reply": "ok", "command": command})


def test_an_ordinary_turn_is_logged():
    assert metrics.should_log({"reply": "olá", "command": ""})
    assert metrics.should_log({"reply": "olá", "command": None})
    assert metrics.should_log({"reply": "olá"})


def test_nothing_to_say_is_not_logged():
    assert not metrics.should_log({"reply": "", "command": ""})
    assert not metrics.should_log(None)
    assert not metrics.should_log({})


def test_the_speaker_is_recorded(tmp_path):
    """It was missing, so nothing could detect the bot naming the person it was
    answering — the "O Gustavo tem de tratar disso" reply to Gustavo."""
    log = tmp_path / "i.jsonl"
    metrics.log_interaction(source="whatsapp", user_message="q",
                            assistant_response="O Gustavo tem de tratar disso",
                            speaker="Gustavo", reply_members=["Gustavo"], path=log)
    row = metrics.load_interactions(log)[0]
    assert row["speaker"] == "Gustavo"
    assert row["speaker"] in row["reply_members"]


def test_a_render_is_recorded_with_what_produced_it(tmp_path):
    """The translated instruction, the final prompt and the editor only ever
    existed as a logger.info on a logger nobody configures."""
    log = tmp_path / "i.jsonl"
    metrics.log_interaction(
        source="imagegen", user_message="transform the monster cans into dildos",
        assistant_response="Replace the cans. Keep the person's face...",
        image_subject="object", image_editor="flux2-klein",
        image_seed=42, image_edited=True, path=log)
    row = metrics.load_interactions(log)[0]
    assert row["image_subject"] == "object"
    assert row["image_editor"] == "flux2-klein"
    assert row["image_edited"] is True
