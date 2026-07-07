"""Person-filter canonicalisation: raw sender-string variants must resolve to
the right member, without cross-member collisions (the W1b retrieval fix)."""

import json

import pytest

from src.chat.retriever import ConversationRetriever


@pytest.fixture
def retriever(tmp_path):
    members = {
        "group_name": "Test",
        "members": [
            {"name": "Gil", "aliases": ["gil", "gilão"],
             "sender_aliases": ["Gil João", "João Gil"]},
            {"name": "Frederico", "aliases": ["frederico", "fred"],
             "sender_aliases": ["fredericop167"]},
            {"name": "Murgeiro", "aliases": ["murgeiro", "joão", "jao"]},
            {"name": "Bernardo", "aliases": ["bernardo", "benny", "benny pereira"]},
        ],
    }
    path = tmp_path / "members.json"
    path.write_text(json.dumps(members), encoding="utf-8")
    return ConversationRetriever({"data": {"group_members_file": str(path)}, "rag": {}})


def test_sender_variants_resolve_to_member(retriever):
    assert retriever._canonical_members("Gil João") == {"Gil"}
    assert retriever._canonical_members("João Gil") == {"Gil"}
    assert retriever._canonical_members("fredericop167") == {"Frederico"}


def test_no_cross_member_collision(retriever):
    # "Gil João" contains "joão" (a Murgeiro alias) but must NOT map to Murgeiro.
    assert "Murgeiro" not in retriever._canonical_members("Gil João")
    # Murgeiro's own sender string still resolves to Murgeiro.
    assert retriever._canonical_members("Murgeiro") == {"Murgeiro"}


def test_multiword_alias_wins(retriever):
    assert retriever._canonical_members("benny pereira") == {"Bernardo"}


def test_non_member_is_unmapped(retriever):
    assert retriever._canonical_members("Zezinha") == set()


def test_person_in_chunk_matches_via_participants(retriever):
    query_members = {"Gil"}
    meta = {"participants": "Gil João,Rafa", "mentioned": ""}
    assert retriever._person_in_chunk(query_members, meta)
    # A chunk with neither Gil as sender nor mention is excluded.
    assert not retriever._person_in_chunk(query_members, {"participants": "Rafa", "mentioned": "bernardo"})
