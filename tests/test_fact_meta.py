"""Dated profile facts and the weekly "Still true?" (src/data/fact_meta.py, bio_gate).

2026-11-25: a roast built a paragraph on a member's job from years ago. Facts were
plain strings, so nothing could tell an old one from a new one. Fictional members.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.response_utils import build_member_prompt_suffix
from src.data import bio_gate, fact_meta
from src.data.bio_gate import BioQueue, apply_answer, digest_due, digest_text, mark_sent

YEAR = 365 * 86400
NOW = 2_000_000_000.0


def member(name, facts, dates=None):
    person = {"name": name, "aliases": [name.lower()], "key_facts": list(facts)}
    for fact, when in (dates or {}).items():
        fact_meta.stamp(person, fact, "gate", when)
    return person


def write(tmp_path, members):
    path = tmp_path / "group_members.json"
    path.write_text(json.dumps({"members": members}), encoding="utf-8")
    return path


def read(path, name):
    return next(m for m in json.loads(path.read_text())["members"] if m["name"] == name)


# ── kinds and dates ──────────────────────────────────────────────────────────
def test_only_jobs_homes_and_relationships_go_stale():
    assert fact_meta.kind("Ana works at a startup.") == "job"
    assert fact_meta.kind("Ana is an engineer abroad.") == "job"
    assert fact_meta.kind("Bruno lives in the city centre.") == "home"
    assert fact_meta.kind("Carlos is married to Dora.") == "relationship"
    assert fact_meta.kind("Ana owns a dog named Rex.") == ""
    assert fact_meta.kind("Bruno loves fast food.") == ""


def test_undated_is_asked_but_only_a_known_old_date_is_marked():
    old = "Ana works at a startup."
    fresh = "Ana works at a bank."
    undated = "Ana lives in the city centre."
    ana = member("Ana", [old, fresh, undated], {old: NOW - 2 * YEAR, fresh: NOW - 30 * 86400})

    assert fact_meta.is_stale(ana, old, NOW) and fact_meta.is_old(ana, old, NOW)
    assert not fact_meta.is_stale(ana, fresh, NOW)
    assert fact_meta.is_stale(ana, undated, NOW) and not fact_meta.is_old(ana, undated, NOW)
    assert fact_meta.STALE_MARK in fact_meta.mark(ana, old, NOW)
    assert fact_meta.STALE_MARK not in fact_meta.mark(ana, undated, NOW)
    assert fact_meta.STALE_MARK not in fact_meta.mark(ana, "Ana owns a dog.", NOW)


def test_the_card_marks_an_old_job_and_tells_the_model_what_that_means(monkeypatch):
    monkeypatch.setattr(fact_meta, "_now", lambda: NOW)
    old = "Ana works at a startup."
    suffix = build_member_prompt_suffix({"members": [member("Ana", [old], {old: NOW - 2 * YEAR})]})

    assert f"Ana works at a startup {fact_meta.STALE_MARK}." in suffix
    assert "di-lo no passado" in suffix and "nem num roast" in suffix


def test_stale_facts_come_undated_first_then_oldest_and_at_most_three():
    ana = member("Ana", ["Ana works at A.", "Ana works at B.", "Ana lives in C.", "Ana owns a dog."],
                 {"Ana works at A.": NOW - 3 * YEAR, "Ana works at B.": NOW - 5 * YEAR})
    bruno = member("Bruno", ["Bruno is married to Dora.", "Bruno works at E."],
                   {"Bruno works at E.": NOW - 10 * 86400})

    picked = fact_meta.stale_facts([ana, bruno], NOW)

    assert [i["fact"] for i in picked] == ["Ana lives in C.", "Bruno is married to Dora.",
                                           "Ana works at B."]


def test_backfill_dates_only_what_the_gate_applied():
    ana = member("Ana", ["Ana works at a bank.", "Ana lives in the city centre."])
    applied = [{"member": "Ana", "fact": "Ana works at a bank", "applied": "auto", "moved_at": NOW}]

    assert fact_meta.backfill([ana], applied) == 1
    assert fact_meta.learned_at(ana, "Ana works at a bank.") == NOW
    assert fact_meta.learned_at(ana, "Ana lives in the city centre.") is None


# ── the gate dates what it writes ────────────────────────────────────────────
def test_an_applied_fact_is_dated(tmp_path):
    path = write(tmp_path, [member("Ana", [])])
    item = {"id": "x", "member": "Ana", "fact": "Ana works at a bank.", "quotes": [],
            "first_seen": NOW}

    bio_gate.apply_items(path, [item], None, "auto")

    assert fact_meta.learned_at(read(path, "Ana"), "Ana works at a bank.") is not None


def test_his_rewording_moves_the_date_to_the_new_fact(tmp_path):
    old = "Ana works at a startup."
    path = write(tmp_path, [member("Ana", [old], {old: NOW - 2 * YEAR})])

    assert bio_gate._replace_fact(path, "Ana", old, "Ana is an engineer at a big firm.")

    ana = read(path, "Ana")
    assert ana["key_facts"] == ["Ana is an engineer at a big firm."]
    assert old not in ana["fact_meta"]
    assert ana["fact_meta"]["Ana is an engineer at a big firm."]["source"] == "his edit"


# ── the weekly "Still true?" ─────────────────────────────────────────────────
def _sent(tmp_path, members):
    path = write(tmp_path, members)
    queue = BioQueue(tmp_path / "bio_queue.json")
    stale = fact_meta.stale_facts(members)
    assert digest_due(queue, [], stale)
    text, ids, added = digest_text(queue, [], stale)
    mark_sent(queue, ids, added, recheck=stale)
    queue.save()
    return path, BioQueue(tmp_path / "bio_queue.json"), text


def test_the_list_asks_whether_old_facts_are_still_true(tmp_path):
    _path, _queue, text = _sent(tmp_path, [member("Ana", ["Ana works at a startup."])])

    assert "Still true?" in text
    assert "1. Ana works at a startup. (no date)" in text


def test_nothing_waiting_and_nothing_old_sends_nothing(tmp_path):
    queue = BioQueue(tmp_path / "bio_queue.json")
    assert not digest_due(queue, [], fact_meta.stale_facts([member("Ana", ["Ana owns a dog."])]))


def test_keep_dates_it_again(tmp_path):
    path, queue, _ = _sent(tmp_path, [member("Ana", ["Ana works at a startup."])])

    reply = apply_answer(queue, path, "1", 12)

    ana = read(path, "Ana")
    assert ana["key_facts"] == ["Ana works at a startup."]
    assert ana["fact_meta"]["Ana works at a startup."]["source"] == "his yes"
    assert "still true" in reply
    assert BioQueue(tmp_path / "bio_queue.json").data["asked"]["recheck"] == []


def test_his_words_drop_one_and_reword_another(tmp_path):
    members = [member("Ana", ["Ana works at a startup."]), member("Bruno", ["Bruno lives in town."])]
    path, queue, _ = _sent(tmp_path, members)

    def model(system, user):
        return json.dumps({"items": [{"n": 1, "action": "edit",
                                      "fact": "Ana is an engineer at a big firm."},
                                     {"n": 2, "action": "drop"}], "rest": "leave"})

    reply = apply_answer(queue, path, "a ana agora é engenheira, o bruno já não vive lá", 12,
                         generate=model)

    assert read(path, "Ana")["key_facts"] == ["Ana is an engineer at a big firm."]
    assert read(path, "Bruno")["key_facts"] == []
    assert "reworded 1" in reply and "took 1 back off" in reply


def test_a_bare_answer_never_drops_an_old_fact_it_did_not_name(tmp_path):
    members = [member("Ana", ["Ana works at a startup."]), member("Bruno", ["Bruno lives in town."])]
    path, queue, _ = _sent(tmp_path, members)

    apply_answer(queue, path, "1", 12)

    assert read(path, "Bruno")["key_facts"] == ["Bruno lives in town."]
    assert "fact_meta" not in read(path, "Bruno")


# ── review_bios.py hand edits ────────────────────────────────────────────────
def _review(tmp_path, *args):
    script = Path(__file__).parent.parent / "scripts" / "review_bios.py"
    return subprocess.run([sys.executable, str(script), "--data-dir", str(tmp_path), *args],
                          capture_output=True, text=True, timeout=60)


def test_review_bios_add_drop_alias_and_undo(tmp_path):
    path = write(tmp_path, [{"name": "Ana", "aliases": ["ana"], "recent_summary": "old talk",
                             "key_facts": ["Ana works at a startup."]}])
    before = path.read_text()

    run = _review(tmp_path, "--member", "Ana", "--drop", "Ana works at a startup",
                  "--add", "Ana is an engineer at a big firm.", "--alias", "Ana Maria",
                  "--clear-summary")
    assert run.returncode == 0, run.stdout + run.stderr

    ana = read(path, "Ana")
    assert ana["key_facts"] == ["Ana is an engineer at a big firm."]
    assert ana["fact_meta"]["Ana is an engineer at a big firm."]["source"] == "manual"
    assert "ana maria" in ana["aliases"] and ana["recent_summary"] == ""

    assert _review(tmp_path, "--undo", "Ana").returncode == 0
    assert json.loads(path.read_text()) == json.loads(before)
