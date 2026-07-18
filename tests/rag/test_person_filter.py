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
            {"name": "Gil", "aliases": ["gil", "gilão"]},
            {"name": "Frederico", "aliases": ["frederico", "fred"]},
            {"name": "Murgeiro", "aliases": ["murgeiro", "joão", "jao"]},
            {"name": "Bernardo", "aliases": ["bernardo", "benny", "benny pereira"]},
        ],
    }
    path = tmp_path / "members.json"
    path.write_text(json.dumps(members), encoding="utf-8")
    config = {
        "data": {
            "group_members_file": str(path),
            "sender_aliases": {"Gil João": "Gil", "João Gil": "Gil", "fredericop167": "Frederico"},
        },
        "rag": {},
    }
    return ConversationRetriever(config)


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


def _chunk(text, sim):
    return {"text": text, "similarity_score": sim}


def test_rrf_boosts_rare_term_chunk(retriever):
    # A rare proper noun (high idf) should be pulled up past a higher-cosine
    # chunk that shares no query term.
    retriever._lex_idf = {"fiveguys": 6.0, "cheese": 2.0, "gil": 1.0}
    candidates = [
        _chunk("gil disse qualquer coisa no chat", 0.55),  # higher cosine, no query term
        _chunk("fiveguys extra cheese", 0.40),             # lower cosine, rare term
    ]
    ranked = retriever._rrf_rerank("fiveguys cheese", candidates, rrf_k=60, lexical_weight=0.5)
    assert ranked[0]["text"] == "fiveguys extra cheese"


def test_rrf_lexical_weight_keeps_dense_winner(retriever):
    # With a low lexical weight, a strong dense match is not displaced by a
    # common-word lexical overlap.
    retriever._lex_idf = {"cao": 0.2, "gil": 0.2}
    candidates = [
        _chunk("gil adotou a cuca no canil", 0.62),  # the relevant dense winner
        _chunk("alguem tem um cao? o gil?", 0.30),   # common-word overlap only
    ]
    ranked = retriever._rrf_rerank("o cao do gil", candidates, rrf_k=60, lexical_weight=0.5)
    assert ranked[0]["text"] == "gil adotou a cuca no canil"
