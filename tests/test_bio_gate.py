"""The profile-fact gate (src/data/bio_gate.py), the weekly list and its answer.

2026-11-19: 455 proposed facts waited for a review nobody was asked to do, and a
"who is he?" got another member's job. Fictional members; no model or network here
(the grounding check is a stub).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import bio_gate
from src.data.bio_gate import BioQueue, apply_answer, decide, digest_text, gate, same_fact

ANA = {"name": "Ana", "aliases": ["ana", "ana maria"], "key_facts": ["Ana supports the local club."]}
BRUNO = {"name": "Bruno", "aliases": ["bruno", "bruno costa"],
          "key_facts": ["Bruno loves football chat."]}
CARLOS = {"name": "Carlos", "aliases": ["carlos"], "key_facts": []}
MEMBERS = [ANA, BRUNO, CARLOS]

LOOKUP = {
    "m1": {"sender": "Bruno", "text": "quem trabalhava na acme era a ana"},
    "m2": {"sender": "Ana", "text": "Idealmente conciliar a parte analítica e comercial como fazia na acme"},
    "m3": {"sender": "Carlos", "text": "o meu carro morreu outra vez, o carro tem 20 anos"},
    "m4": {"sender": "Carlos", "text": "levei o meu carro à oficina, é o meu carro de sempre"},
    "m5": {"sender": "Ana", "text": "votei no partido X outra vez, sou de esquerda desde sempre"},
}


def entry(fact, quote, *ids):
    return {"fact": fact, "quote": quote, "evidence": list(ids)}


def write_members(tmp_path, members=MEMBERS):
    path = tmp_path / "group_members.json"
    path.write_text(json.dumps({"members": json.loads(json.dumps(members))}), encoding="utf-8")
    return path


def facts_of(path, name):
    return next(m for m in json.loads(path.read_text())["members"] if m["name"] == name)["key_facts"]


def yes(system, user):
    return "YES"


def no(system, user):
    return "NO"


def test_a_quote_about_gil_never_becomes_a_fact_about_bruno():
    others = [ANA, CARLOS]
    verdict, reason = decide(entry("Bruno worked at Acme.", "quem trabalhava na acme era a ana", "m1"),
                             BRUNO, others, "Bruno")
    assert verdict == "drop" and "somebody else" in reason
    verdict, _ = decide(entry("Ana used to work at Acme.", "quem trabalhava na acme era a ana", "m1"),
                        ANA, [BRUNO, CARLOS], "Bruno")
    assert verdict == "candidate"


def test_noise_opinions_media_and_the_subjectless_are_dropped():
    others = [BRUNO, CARLOS]
    assert decide(entry("Ana works.", "Back to work nigga"), ANA, others, "Ana")[0] == "drop"
    assert decide(entry("Ana believes the economy is rigged.", "a economia é uma fraude"), ANA, others, "Ana")[0] == "drop"
    assert decide(entry("Ana is in Vila Nova.", "estamos em vila nova"), ANA, others, "Ana")[0] == "drop"
    assert decide(entry("Carlos goes to the gym.", "[Imagem: um homem no ginásio]"), CARLOS, [ANA], "Carlos")[0] == "drop"
    assert decide(entry("Works at a bank in Lisbon.", "trabalho num banco"), ANA, others, "Ana")[0] == "drop"


def test_sensitive_facts_are_asked_and_friday_is_not_sensitive():
    others = [BRUNO, CARLOS]
    assert decide(entry("Ana votes for party X and is left-wing.", "votei no partido X"), ANA, others, "Ana") \
        == ("ask", "sensitive")
    assert decide(entry("Ana goes to the office every Friday morning.", "sexta de manhã no escritório"),
                  ANA, others, "Ana")[0] == "candidate"


def test_same_fact_reworded():
    assert same_fact("Ana used to work at Acme", "Ana worked at Acme, in analytics and sales.")
    assert not same_fact("Ana works at Initech", "Ana worked at Acme, in analytics and sales.")


def test_two_quotes_and_a_yes_apply_one_quote_waits(tmp_path):
    path = write_members(tmp_path)
    queue = BioQueue(tmp_path / "q.json")
    batch = {"Carlos": [entry("Carlos owns a car that is twenty years old.", "o carro tem 20 anos", "m3")]}
    counts = gate(batch, MEMBERS, LOOKUP, queue, yes, path, 12)
    assert counts["applied"] == 0 and len(queue.data["candidates"]) == 1
    batch = {"Carlos": [entry("Carlos owns an old car.", "é o meu carro de sempre", "m4")]}
    counts = gate(batch, MEMBERS, LOOKUP, queue, yes, path, 12)
    assert counts["applied"] == 1
    assert facts_of(path, "Carlos") == ["Carlos owns a car that is twenty years old."]
    assert queue.data["applied"][0]["snapshot"]
    assert (tmp_path / "profile_snapshots").exists()


def test_the_models_no_sends_it_to_his_list_not_the_profile(tmp_path):
    path = write_members(tmp_path)
    queue = BioQueue(tmp_path / "q.json")
    batch = {"Carlos": [entry("Carlos owns a car that is twenty years old.", "o carro tem 20 anos", "m3"),
                       entry("Carlos owns an old car.", "é o meu carro de sempre", "m4")]}
    gate(batch, MEMBERS, LOOKUP, queue, no, path, 12)
    assert facts_of(path, "Carlos") == []
    assert queue.data["ask"][0]["reason"] == "the model wasn't sure"


def test_a_busy_gpu_leaves_candidates_waiting(tmp_path):
    def busy(system, user):
        raise RuntimeError("GPU1 is lent")
    path = write_members(tmp_path)
    queue = BioQueue(tmp_path / "q.json")
    batch = {"Carlos": [entry("Carlos owns a car that is twenty years old.", "o carro tem 20 anos", "m3"),
                       entry("Carlos owns an old car.", "é o meu carro de sempre", "m4")]}
    gate(batch, MEMBERS, LOOKUP, queue, busy, path, 12)
    assert len(queue.data["candidates"]) == 1 and not queue.data["ask"]


def test_a_fact_seen_once_expires_quietly(tmp_path, monkeypatch):
    path = write_members(tmp_path)
    queue = BioQueue(tmp_path / "q.json")
    gate({"Carlos": [entry("Carlos owns a car that is twenty years old.", "o carro tem 20 anos", "m3")]},
         MEMBERS, LOOKUP, queue, yes, path, 12)
    later = time.time() + 29 * 86400
    monkeypatch.setattr(bio_gate, "_now", lambda: later)
    gate({}, MEMBERS, LOOKUP, queue, yes, path, 12)
    assert not queue.data["candidates"] and not queue.data["ask"] and queue.data["dropped"] == 1


def test_a_full_profile_is_not_trimmed(tmp_path):
    full = dict(CARLOS, key_facts=[f"Carlos fact number {n} about him." for n in range(2)])
    path = write_members(tmp_path, [ANA, BRUNO, full])
    queue = BioQueue(tmp_path / "q.json")
    batch = {"Carlos": [entry("Carlos owns a car that is twenty years old.", "o carro tem 20 anos", "m3"),
                       entry("Carlos owns an old car.", "é o meu carro de sempre", "m4")]}
    gate(batch, [ANA, BRUNO, full], LOOKUP, queue, yes, path, 2)
    assert len(facts_of(path, "Carlos")) == 2
    assert queue.data["ask"][0]["reason"] == "his profile is full"


def _two_waiting(tmp_path):
    path = write_members(tmp_path)
    queue = BioQueue(tmp_path / "q.json")
    gate({"Ana": [entry("Ana votes for party X and is left-wing.", "votei no partido X", "m5"),
                  entry("Ana is left-wing and active in politics.", "sou de esquerda desde sempre", "m5")]},
         MEMBERS, LOOKUP, queue, yes, path, 12)
    return path, queue


def _auto_added(queue, path, fact="Carlos owns an old car."):
    item = {"id": "auto1", "member": "Carlos", "fact": fact, "quotes": [], "first_seen": time.time(),
            "applied": "auto", "moved_at": time.time()}
    queue.data["applied"].append(item)
    doc = json.loads(path.read_text())
    next(m for m in doc["members"] if m["name"] == "Carlos")["key_facts"] = [fact]
    path.write_text(json.dumps(doc))
    return item


def test_weekly_list_and_his_numbered_answer(tmp_path):
    path, queue = _two_waiting(tmp_path)
    assert len(queue.data["ask"]) == 2
    assert bio_gate.digest_due(queue, [])
    text, ids, added = digest_text(queue, [{"user_message": "quem é este gajo?",
                                            "assistant_response": "O Bruno trabalha na Acme",
                                            "speaker": "Ana", "comment": "holy allucinacao"}])
    assert text.startswith(bio_gate.BIO_TAG) and "1. Ana votes" in text and "2. Ana is left-wing" in text
    assert "holy allucinacao" in text and "by text or voice" in text and len(ids) == 2 and added == []
    bio_gate.mark_sent(queue, ids, added)
    reply = apply_answer(queue, path, "2", 12)
    assert "applied 1" in reply and "dropped 1" in reply
    assert facts_of(path, "Ana")[-1] == "Ana is left-wing and active in politics."
    assert not queue.data["ask"] and queue.data["asked"]["ids"] == []
    assert "no open list" in apply_answer(queue, path, "1", 12)


def test_a_skipped_week_comes_back_first_and_flagged(tmp_path, monkeypatch):
    path, queue = _two_waiting(tmp_path)
    _text, ids, added = digest_text(queue, [])
    bio_gate.mark_sent(queue, ids, added)
    later = time.time() + 8 * 86400
    monkeypatch.setattr(bio_gate, "_now", lambda: later)
    gate({"Ana": [entry("Ana votes for party Y now and is left-wing.", "votei no partido Y", "m5")]},
         MEMBERS, LOOKUP, queue, yes, path, 12)
    assert bio_gate.digest_due(queue, [])
    text, ids2, _ = digest_text(queue, [])
    assert "You didn't answer last week" in text
    assert "1. ↩ Ana votes for party X" in text and "2. ↩ Ana is left-wing" in text
    bio_gate.mark_sent(queue, ids2, [])
    assert all(i.get("asked_times") == 2 for i in queue.data["ask"] if i["id"] in ids)


def test_plain_words_edit_and_drop_through_the_model(tmp_path):
    path, queue = _two_waiting(tmp_path)
    _auto_added(queue, path)
    _text, ids, added = digest_text(queue, [])
    assert added == ["auto1"]
    bio_gate.mark_sent(queue, ids, added)
    seen = {}

    def model(system, user):
        seen["user"] = user
        return json.dumps({"items": [
            {"n": 1, "action": "drop"},
            {"n": 2, "action": "edit", "fact": "Ana is active in local politics."},
            {"n": 3, "action": "edit", "fact": "Carlos owns a 20-year-old Fiat."}], "rest": "leave"})

    reply = apply_answer(queue, path, "o primeiro está errado, o segundo é política local, "
                         "o carro do Carlos é um Fiat com 20 anos", 12, generate=model)
    assert "3. Carlos owns an old car." in seen["user"] and "His answer" in seen["user"]
    assert "applied 1 (1 with your wording)" in reply and "dropped 1" in reply and "reworded 1" in reply
    assert facts_of(path, "Ana")[-1] == "Ana is active in local politics."
    assert facts_of(path, "Carlos") == ["Carlos owns a 20-year-old Fiat."]


def test_everythings_right_keeps_all_without_the_model(tmp_path):
    path, queue = _two_waiting(tmp_path)
    _auto_added(queue, path)
    _text, ids, added = digest_text(queue, [])
    bio_gate.mark_sent(queue, ids, added)

    def model(system, user):
        raise AssertionError("not needed")

    reply = apply_answer(queue, path, "All good!", 12, generate=model)
    assert "applied 2" in reply and facts_of(path, "Carlos") == ["Carlos owns an old car."]


def test_an_auto_added_fact_he_calls_wrong_comes_off(tmp_path):
    path, queue = _two_waiting(tmp_path)
    _auto_added(queue, path)
    _text, ids, added = digest_text(queue, [])
    bio_gate.mark_sent(queue, ids, added)
    reply = apply_answer(queue, path, "number three is wrong, the rest is fine", 12,
                         generate=lambda s, u: '{"items": [{"n": 3, "action": "drop"}], "rest": "keep"}')
    assert "took 1 back" in reply and "applied 2" in reply
    assert facts_of(path, "Carlos") == []


def test_an_unusable_model_answer_changes_nothing(tmp_path):
    path, queue = _two_waiting(tmp_path)
    _text, ids, added = digest_text(queue, [])
    bio_gate.mark_sent(queue, ids, added)
    reply = apply_answer(queue, path, "hmm talvez", 12, generate=lambda s, u: "não sei")
    assert "nothing was changed" in reply and len(queue.data["ask"]) == 2
    assert apply_answer(queue, path, "hmm talvez", 12) .startswith(bio_gate.BIO_TAG + " I couldn't read")


def test_answers_that_are_not_answers():
    assert bio_gate.parse_answer("all", 3) == [1, 2, 3]
    assert bio_gate.parse_answer("nenhum", 3) == []
    assert bio_gate.parse_answer("1, 3 e 9", 3) == [1, 3]
    assert bio_gate.parse_answer("what is this?", 3) is None


def test_undo_restores_the_snapshot(tmp_path):
    path = write_members(tmp_path)
    queue = BioQueue(tmp_path / "q.json")
    batch = {"Carlos": [entry("Carlos owns a car that is twenty years old.", "o carro tem 20 anos", "m3"),
                       entry("Carlos owns an old car.", "é o meu carro de sempre", "m4")]}
    gate(batch, MEMBERS, LOOKUP, queue, yes, path, 12)
    queue.save()
    assert facts_of(path, "Carlos")
    assert bio_gate.undo(BioQueue(tmp_path / "q.json"), path, "Carlos")
    assert facts_of(path, "Carlos") == []


def test_called_wrong_rows_are_read_back(tmp_path):
    log = tmp_path / "fb.jsonl"
    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    log.write_text("\n".join(json.dumps(r) for r in [
        {"kind": "called_wrong", "timestamp": now, "comment": "holy allucinacao"},
        {"type": "rating", "timestamp": now, "rating": "down"},
        {"kind": "called_wrong", "timestamp": "2020-01-01T00:00:00+00:00", "comment": "old"},
    ]), encoding="utf-8")
    rows = bio_gate.recent_called_wrong(log)
    assert [r["comment"] for r in rows] == ["holy allucinacao"]
