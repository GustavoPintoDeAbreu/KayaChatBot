"""The biography refresh: what it may read, and what it may not do.

Profiles are injected into EVERY chat's system prompt, so anything this learns
is effectively public to the group — which makes the scope rule and the
propose-never-write rule properties to test, not implementation details.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import bio_refresh

MEMBERS = [
    {"name": "Rafa", "aliases": ["rafa", "chamusca"], "key_facts": ["Rafa hosts the group."]},
    {"name": "Gil", "aliases": ["gil"], "key_facts": []},
    {"name": "Romano", "aliases": ["romano", "ricardo romano"], "key_facts": []},
]


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "data" / "live_messages").mkdir(parents=True)
    members = tmp_path / "data" / "group_members.json"
    members.write_text(json.dumps({"members": MEMBERS}), encoding="utf-8")
    return tmp_path


def write_log(workspace, rows, name="shared.jsonl"):
    path = workspace / "data" / "live_messages" / name
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def msg(mid, sender, text, ts=1000, scope="shared"):
    return {"id": mid, "chat_id": "g@g.us", "sender": sender, "text": text,
            "timestamp": ts, "scope": scope}


def make_config(workspace, **over):
    config = {
        "whatsapp": {"bot_jid": "351900000000@c.us", "bot_lid": "237065786642635@lid",
                     "message_log_dir": str(workspace / "data" / "live_messages")},
        "data": {"group_members_file": str(workspace / "data" / "group_members.json"),
                 "sender_aliases": {}},
        "rag": {"max_facts_per_member": 4},
        "chat": {"concurrency": {"acquire_timeout": 5}},
        "bio_refresh": {"state_file": str(workspace / "state.json"),
                        "proposals_file": str(workspace / "proposals.json")},
    }
    config["bio_refresh"].update(over)
    return config


class ScriptedBackend:
    """Returns queued JSON payloads; records what it was asked."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def generate(self, messages, *, max_new_tokens=None, sampling=None):
        self.calls.append(messages)
        return self.answers.pop(0) if self.answers else "{}"


def resolver_for(workspace):
    from src.data.identity_resolver import SenderResolver

    return SenderResolver(workspace / "data" / "group_members.json", {})


def run(workspace, backend, **over):
    config = make_config(workspace, **over)
    return bio_refresh.run_cycle(
        config, backend, members_file=workspace / "data" / "group_members.json",
        state_path=str(workspace / "state.json"),
        proposals_path=str(workspace / "proposals.json"))


# ── what it is allowed to read ───────────────────────────────────────────────
def test_a_dm_scoped_row_is_never_read(workspace):
    """A profile reaches every chat, so a DM fact would become group-readable.
    The scope is checked per row rather than trusted from the filename."""
    write_log(workspace, [msg("m1", "Rafa", "o Rafa comprou uma carrinha", scope="dm:abc")])

    kept, _, _ = bio_refresh.read_new_messages(
        make_config(workspace), bio_refresh.BioRefreshState(str(workspace / "s.json")),
        resolver_for(workspace))

    assert kept == []


def test_a_message_aimed_at_the_bot_is_dropped(workspace):
    write_log(workspace, [
        msg("m1", "Gil", "@237065786642635 quem é o mais burro?"),
        msg("m2", "Gil", "o Gil foi a Hvar em agosto"),
    ])

    kept, _, _ = bio_refresh.read_new_messages(
        make_config(workspace), bio_refresh.BioRefreshState(str(workspace / "s.json")),
        resolver_for(workspace))

    assert [m["id"] for m in kept] == ["m2"]


def test_a_failed_image_description_is_skipped(workspace):
    write_log(workspace, [
        msg("m1", "Gil", "[Imagem: Por favor, fornece o vídeo ou a imagem.]"),
        msg("m2", "Gil", "o Gil adoptou uma cadela"),
    ])

    kept, _, _ = bio_refresh.read_new_messages(
        make_config(workspace), bio_refresh.BioRefreshState(str(workspace / "s.json")),
        resolver_for(workspace))

    assert [m["id"] for m in kept] == ["m2"]


def test_an_unresolved_sender_is_reported_not_invented(workspace):
    """An unmapped display name is a mapping waiting to be added. Treating it as
    a new person is how a phantom member gets a profile."""
    write_log(workspace, [msg("m1", "fredericop167", "comprei um carro"),
                          msg("m2", "Rafa", "o Rafa treina kickboxing")])

    kept, unresolved, _ = bio_refresh.read_new_messages(
        make_config(workspace), bio_refresh.BioRefreshState(str(workspace / "s.json")),
        resolver_for(workspace))

    assert [m["sender"] for m in kept] == ["Rafa"]
    assert unresolved == {"fredericop167": 1}


def test_an_ambiguous_sender_is_not_guessed(workspace):
    """Two members answer to "Ricardo"; attributing to either would be a lie."""
    write_log(workspace, [msg("m1", "Ricardo", "fui a Espanha")])

    kept, unresolved, _ = bio_refresh.read_new_messages(
        make_config(workspace), bio_refresh.BioRefreshState(str(workspace / "s.json")),
        resolver_for(workspace))

    assert kept == []
    assert unresolved == {"Ricardo": 1}


def test_the_watermark_stops_a_message_being_read_twice(workspace):
    write_log(workspace, [msg("m1", "Rafa", "o Rafa mudou de casa", ts=500),
                          msg("m2", "Rafa", "o Rafa comprou uma bicicleta", ts=1500)])
    state = bio_refresh.BioRefreshState(str(workspace / "s.json"))
    state.advance(1000, 0)

    kept, _, newest = bio_refresh.read_new_messages(
        make_config(workspace), state, resolver_for(workspace))

    assert [m["id"] for m in kept] == ["m2"]
    assert newest == 1500


# ── the cycle ────────────────────────────────────────────────────────────────
EXTRACTION = json.dumps({"members": {"Rafa": {"facts": ["Rafa trains kickboxing."]}}})
DISTIL = json.dumps(["Rafa trains kickboxing.", "Rafa hosts the group."])


def test_a_cycle_proposes_without_touching_the_profiles(workspace):
    """The whole point of propose-only: these are statements about real people,
    made by a model, with nobody watching."""
    write_log(workspace, [msg("m1", "Rafa", "o Rafa anda a treinar kickboxing")])
    before = (workspace / "data" / "group_members.json").read_text(encoding="utf-8")

    report = run(workspace, ScriptedBackend(EXTRACTION, DISTIL))

    assert "Rafa" in report["proposals"]
    assert report["proposals"]["Rafa"]["added"] == ["Rafa trains kickboxing."]
    assert (workspace / "data" / "group_members.json").read_text(encoding="utf-8") == before


def test_the_proposal_carries_the_messages_behind_it(workspace):
    """The review has to show why, or it is just a model's word for it."""
    write_log(workspace, [msg("m1", "Rafa", "o Rafa anda a treinar kickboxing")])

    report = run(workspace, ScriptedBackend(EXTRACTION, DISTIL))

    assert report["proposals"]["Rafa"]["new_facts"][0]["evidence"] == ["m1"]


def test_proposals_are_written_to_disk(workspace):
    write_log(workspace, [msg("m1", "Rafa", "o Rafa anda a treinar kickboxing")])

    run(workspace, ScriptedBackend(EXTRACTION, DISTIL))

    saved = json.loads((workspace / "proposals.json").read_text(encoding="utf-8"))
    assert "Rafa" in saved["proposals"]


def test_a_second_cycle_does_not_discard_the_first(workspace):
    """Review happens when somebody gets round to it."""
    write_log(workspace, [msg("m1", "Rafa", "o Rafa treina kickboxing", ts=1000)])
    run(workspace, ScriptedBackend(EXTRACTION, DISTIL))

    write_log(workspace, [msg("m2", "Gil", "o Gil mudou-se para Lisboa", ts=2000)])
    run(workspace, ScriptedBackend(
        json.dumps({"members": {"Gil": {"facts": ["Gil moved to Lisbon."]}}}),
        json.dumps(["Gil moved to Lisbon."])))

    saved = json.loads((workspace / "proposals.json").read_text(encoding="utf-8"))
    assert set(saved["proposals"]) == {"Rafa", "Gil"}


def test_nothing_new_is_not_an_error(workspace):
    write_log(workspace, [])

    report = run(workspace, ScriptedBackend())

    assert report["skipped"] == "nothing new"
    assert report["proposals"] == {}


def test_a_busy_gpu_leaves_the_watermark_alone(workspace, monkeypatch):
    """Being late costs nothing; taking the lock from a live reply does."""
    from src.chat.gpu_lock import GpuBusyError

    write_log(workspace, [msg("m1", "Rafa", "o Rafa treina kickboxing", ts=1000)])

    def boom(*a, **kw):
        raise GpuBusyError("busy")

    monkeypatch.setattr(bio_refresh, "_generate", boom)

    report = run(workspace, ScriptedBackend())

    assert report["skipped"] == "GPU busy"
    assert bio_refresh.BioRefreshState(str(workspace / "state.json")).watermark == 0


def test_unparseable_model_output_does_not_crash_the_cycle(workspace):
    write_log(workspace, [msg("m1", "Rafa", "o Rafa treina kickboxing")])

    report = run(workspace, ScriptedBackend("not json at all"))

    assert report["proposals"] == {}


# ── distillation ─────────────────────────────────────────────────────────────
def test_a_pinned_fact_survives_a_model_that_drops_it():
    """"Pinned" has to mean pinned, or the flag is decoration."""
    member = {"name": "Gil", "key_facts": ["Gil runs."],
              "pinned_facts": ["Gil works in sales."]}

    facts = bio_refresh.distil_key_facts(
        ScriptedBackend(json.dumps(["Gil runs.", "Gil likes techno."])),
        {"bio_refresh": {}, "chat": {"concurrency": {"acquire_timeout": 5}}},
        member, ["Gil likes techno."], limit=4)

    assert "Gil works in sales." in facts


def test_the_cap_is_respected():
    member = {"name": "Gil", "key_facts": []}
    many = json.dumps([f"Gil fact {n}." for n in range(10)])

    facts = bio_refresh.distil_key_facts(
        ScriptedBackend(many),
        {"bio_refresh": {}, "chat": {"concurrency": {"acquire_timeout": 5}}},
        member, ["Gil fact 0."], limit=4)

    assert len(facts) == 4


def test_a_member_with_nothing_new_is_left_alone():
    """No new evidence, no proposal — the cycle must not churn every profile."""
    member = {"name": "Gil", "key_facts": ["Gil runs."]}

    assert bio_refresh.distil_key_facts(
        ScriptedBackend(), {"bio_refresh": {}}, member, [], limit=4) is None


# ── keeping the review readable, and the facts clean ─────────────────────────
def test_a_restatement_keeps_the_existing_wording():
    """Every removal in the first two dry runs was a parenthetical being
    stripped, not a fact changing. A review that is all noise is not read."""
    current = ["Rafa (Rafael, Chamusca) is married to Mel and has a son named Martim."]

    settled = bio_refresh.settle_wording(
        ["Rafa is married to Mel and has a son named Martim."], current)

    assert settled == current


def test_a_genuine_change_is_not_collapsed():
    current = ["Rafa lives in Lisbon."]

    assert bio_refresh.settle_wording(["Rafa now lives in Porto."], current) == \
        ["Rafa now lives in Porto."]


def test_a_fact_that_names_nobody_is_dropped():
    """The member list is shuffled per prompt, so "Wedding planned for next
    September" attaches to nobody. Dropped, not repaired: guessing the subject
    is how a fact lands on the wrong person."""
    facts = bio_refresh.distil_key_facts(
        ScriptedBackend(json.dumps(["Wedding planned for September.", "Manuel is a pilot."])),
        {"bio_refresh": {}, "chat": {"concurrency": {"acquire_timeout": 5}}},
        {"name": "Manuel", "key_facts": []}, ["Manuel is a pilot."], limit=8)

    assert facts == ["Manuel is a pilot."]


def test_a_blocked_term_never_reaches_a_proposal():
    """The deterministic half of "never record this". The prompt asks; this does
    not, and data.blocked_terms is the list the batch pipeline already uses."""
    config = {"bio_refresh": {}, "chat": {"concurrency": {"acquire_timeout": 5}},
              "data": {"blocked_terms": ["Dolby Atmos"]}}

    facts = bio_refresh.distil_key_facts(
        ScriptedBackend(json.dumps(["Gil demos Dolby Atmos setups.", "Gil runs."])),
        config, {"name": "Gil", "key_facts": []}, ["Gil runs."], limit=8)

    assert facts == ["Gil runs."]


def test_storage_is_wider_than_the_prompt_shows(workspace):
    """rag.max_facts_per_member caps what the PROMPT shows and is applied at
    injection. Distilling to it would make every accepted proposal delete
    history — the first dry run cut a member from six facts to four."""
    write_log(workspace, [msg("m1", "Rafa", "o Rafa treina kickboxing")])
    many = json.dumps([f"Rafa fact {n}." for n in range(9)])

    report = run(workspace, ScriptedBackend(EXTRACTION, many))

    assert len(report["proposals"]["Rafa"]["proposed"]) == 8
