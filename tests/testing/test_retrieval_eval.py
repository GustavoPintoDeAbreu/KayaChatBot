"""Unit tests for the retrieval-eval scoring primitives (no DB / no GPU)."""

from src.testing.retrieval_eval import (
    _norm,
    _spec_matches_chunk,
    _first_hit_rank,
    _specs_covered,
)


def test_norm_strips_accents_and_case():
    assert _norm("Marginalíssimo") == "marginalissimo"
    assert _norm("PRAGA") == _norm("praga")


def test_spec_requires_all_terms_cooccur():
    chunk = "Peter: o Kobe veio comigo para Queijas"
    assert _spec_matches_chunk(["Kobe"], chunk)
    assert _spec_matches_chunk(["kobe", "queijas"], chunk)  # accent/case-insensitive AND
    assert not _spec_matches_chunk(["Kobe", "Cuca"], chunk)  # Cuca absent


def test_first_hit_rank_and_mrr():
    chunks = [
        {"text": "unrelated chatter"},
        {"text": "someone mentions Cuca the dog"},
        {"text": "Peter and Kobe at the beach"},
    ]
    gold = [["Kobe"]]
    assert _first_hit_rank(chunks, gold) == 3
    assert _first_hit_rank(chunks, [["nonexistent"]]) is None


def test_specs_covered_counts_distinct_gold():
    chunks = [
        {"text": "Fuel TV job"},
        {"text": "random"},
    ]
    gold = [["Fuel"], ["Dazn"]]  # two relevant chunks; only one present
    assert _specs_covered(chunks, gold) == 1
    assert _specs_covered(chunks, [["Fuel"]]) == 1
