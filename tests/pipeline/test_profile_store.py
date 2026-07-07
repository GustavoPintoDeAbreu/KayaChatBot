"""Provenance/attribution audit for member facts (W3)."""

import json

import pytest

from src.data.profile_store import (
    MemberEvidenceIndex,
    salient_terms,
    audit_member_key_facts,
)


@pytest.fixture
def members_file(tmp_path):
    data = {
        "members": [
            {"name": "Gil", "aliases": ["gil", "gilão"],
             "key_facts": ["Gil owns a dog named Carlota.",
                           "Gil supports Sporting."]},
            {"name": "Peter", "aliases": ["peter"], "key_facts": []},
            {"name": "Carnall", "aliases": ["carnall"], "key_facts": []},
        ]
    }
    p = tmp_path / "members.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _msgs():
    return [
        {"id": "m1", "sender": "Gil", "timestamp": "2024-10-19T10:00:00", "text": "a minha Carlota é anti-social"},
        {"id": "m2", "sender": "Gil", "timestamp": "2024-10-20T10:00:00", "text": "vou passear a Carlota"},
        {"id": "m3", "sender": "Peter", "timestamp": "2024-11-01T10:00:00", "text": "o Sporting jogou bem"},
        {"id": "m4", "sender": "Carnall", "timestamp": "2024-11-02T10:00:00", "text": "vou levar a cuca ao jardim"},
        {"id": "m5", "sender": "Gil", "timestamp": "2024-11-03T10:00:00", "text": "o Sporting outra vez"},
    ]


def test_salient_terms_drops_member_name_and_stopwords():
    terms = salient_terms("Gil owns a dog named Carlota.", ["gil", "gilão"])
    assert "carlota" in terms
    assert "gil" not in terms
    assert "dog" not in terms  # generic stopword


def test_index_attributes_by_sender_and_mention(members_file):
    idx = MemberEvidenceIndex(members_file).index(_msgs())
    ev = idx.audit_fact("Gil", ["Carlota"])
    assert ev.support_count == 2  # two Gil messages about Carlota
    assert ev.associated_members.get("Gil") == 2
    assert "m1" in ev.sample_msg_ids
    assert not ev.attribution_ambiguous


def test_cross_attribution_flagged(members_file):
    # "Sporting" appears for Gil (1) and Peter (1) — tie, not ambiguous; add a
    # second Peter mention to make Peter dominate.
    msgs = _msgs() + [{"id": "m6", "sender": "Peter", "timestamp": "2024-11-04T10:00:00",
                       "text": "grande Sporting"}]
    idx = MemberEvidenceIndex(members_file).index(msgs)
    ev = idx.audit_fact("Gil", ["Sporting"])
    # Peter (2) now beats Gil (2)? Gil has m5 + ... recount: Gil m5 only =1? m5 has Sporting.
    # Gil: m5 (1). Peter: m3, m6 (2). So Peter dominates → ambiguous for Gil.
    assert ev.associated_members["Peter"] >= ev.associated_members["Gil"]
    assert ev.attribution_ambiguous


def test_audit_member_key_facts_verdicts(members_file):
    members = json.loads(members_file.read_text())["members"]
    gil = next(m for m in members if m["name"] == "Gil")
    idx = MemberEvidenceIndex(members_file).index(_msgs())
    rows = audit_member_key_facts(gil, idx)
    by_fact = {r["fact"]: r["verdict"] for r in rows}
    assert by_fact["Gil owns a dog named Carlota."] == "ok"
