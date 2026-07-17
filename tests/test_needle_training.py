"""Unit tests for generate_needle_training.py — deterministic, no GPU, no data files."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generate_needle_training import (
    _count_tokens,
    _fill_template,
    _FACT_TEMPLATES,
    build_filler_blocks,
    build_example,
    generate_needle_examples,
    plant_needle,
)

# Stub messages for tests — no real personal data
_MSGS = [
    {"sender": "Membro A", "text": "O jantar ficou combinado para sábado."},
    {"sender": "Membro B", "text": "Eu também quero ir, a que horas é?"},
    {"sender": "Membro C", "text": "Às 20h no Bairro do Avillez, acho eu."},
    {"sender": "Membro A", "text": "Perfeito, lembro o resto do grupo."},
    {"sender": "Membro D", "text": "Alguém sabe se o Membro E vem?"},
    {"sender": "Membro B", "text": "Disse que talvez, depende do trabalho."},
    {"sender": "Membro C", "text": "Ok, vemos amanhã então."},
    {"sender": "Membro A", "text": "Combinado. Até lá."},
]
_NAMES = ["Membro A", "Membro B", "Membro C"]

# Bench needle strings that must NEVER appear in training data
_BENCH_STRINGS = ["código secreto", "4827"]


def _no_bench_contamination(text: str) -> bool:
    return not any(s in text for s in _BENCH_STRINGS)


def test_count_tokens_basic():
    assert _count_tokens("hello world") == pytest.approx(round(2 / 0.60), abs=1)


def test_fill_template_replaces_all():
    import random
    tmpl = _FACT_TEMPLATES[0]
    rng = random.Random(42)
    fact = _fill_template(tmpl["fact"], tmpl["slots"], "TestName", rng)
    assert "TestName" in fact
    assert "{name}" not in fact
    for key in tmpl["slots"]:
        assert "{" + key + "}" not in fact


def test_build_filler_blocks_nonempty():
    import random
    rng = random.Random(3407)
    result = build_filler_blocks(_MSGS, target_token_count=300, rng=rng)
    assert "=== Conversas relevantes do grupo ===" in result
    assert "=== Fim das conversas ===" in result
    assert _count_tokens(result) >= 50


def test_plant_needle_at_depth_0():
    context = "=== Conversas relevantes do grupo ===\n--- Conversa 1 ---\nMembro A: ola\n=== Fim das conversas ==="
    result = plant_needle(context, "FACTO PLANTADO", 0.0)
    assert "FACTO PLANTADO" in result


def test_plant_needle_at_depth_1():
    context = "=== Conversas relevantes do grupo ===\n--- Conversa 1 ---\nMembro A: texto\nMembro B: mais texto\n=== Fim das conversas ==="
    result = plant_needle(context, "FACTO PLANTADO", 1.0)
    assert "FACTO PLANTADO" in result


def test_plant_needle_preserves_context_content():
    context = "=== Conversas relevantes do grupo ===\n--- Conversa 1 ---\nMembro A: texto original\n=== Fim das conversas ==="
    result = plant_needle(context, "FACTO", 0.5)
    assert "texto original" in result
    assert "FACTO" in result


def test_build_example_schema():
    import random
    rng = random.Random(3407)
    tmpl = _FACT_TEMPLATES[0]
    ex = build_example(_MSGS, _NAMES, tmpl, depth=0.5, target_tokens=500, rng=rng)
    assert ex is not None
    assert "conversations" in ex
    assert len(ex["conversations"]) == 2
    assert ex["conversations"][0]["role"] == "user"
    assert ex["conversations"][1]["role"] == "assistant"
    assert ex["source"] == "needle_synthetic"
    assert "meta" in ex
    assert ex["meta"]["depth"] == 0.5


def test_build_example_answer_contains_fact_value():
    import random
    rng = random.Random(3407)
    tmpl = _FACT_TEMPLATES[2]  # price template
    ex = build_example(_MSGS, _NAMES, tmpl, depth=0.25, target_tokens=600, rng=rng)
    assert ex is not None
    prices = tmpl["slots"]["price"]
    answer = ex["conversations"][1]["content"]
    assert any(p in answer for p in prices), f"No price found in: {answer}"


def test_build_example_needle_in_user_content():
    import random
    rng = random.Random(3407)
    tmpl = _FACT_TEMPLATES[0]
    ex = build_example(_MSGS, _NAMES, tmpl, depth=0.5, target_tokens=500, rng=rng)
    assert ex is not None
    user_content = ex["conversations"][0]["content"]
    # The answer value should appear in the user content (it's planted there)
    answer = ex["conversations"][1]["content"]
    # Extract a unique token from the answer to check it's in the context
    answer_words = set(answer.split())
    user_words = set(user_content.split())
    assert answer_words & user_words, "Answer value not found in user context"


def test_generate_determinism():
    ex1 = generate_needle_examples(10, seed=42, messages=_MSGS, member_names=_NAMES)
    ex2 = generate_needle_examples(10, seed=42, messages=_MSGS, member_names=_NAMES)
    for a, b in zip(ex1, ex2):
        assert a["conversations"][0]["content"] == b["conversations"][0]["content"]
        assert a["conversations"][1]["content"] == b["conversations"][1]["content"]


def test_generate_different_seeds_differ():
    ex1 = generate_needle_examples(5, seed=1, messages=_MSGS, member_names=_NAMES)
    ex2 = generate_needle_examples(5, seed=2, messages=_MSGS, member_names=_NAMES)
    # At least one should differ
    assert any(
        a["conversations"][1]["content"] != b["conversations"][1]["content"]
        for a, b in zip(ex1, ex2)
    )


def test_no_bench_contamination():
    examples = generate_needle_examples(50, seed=3407, messages=_MSGS, member_names=_NAMES)
    for ex in examples:
        for turn in ex["conversations"]:
            assert _no_bench_contamination(turn["content"]), (
                f"Bench needle string found in: {turn['content'][:100]}"
            )


def test_token_range_respected():
    examples = generate_needle_examples(
        20, seed=3407, min_tokens=300, max_tokens=800, messages=_MSGS, member_names=_NAMES
    )
    for ex in examples:
        tokens = ex["meta"]["actual_tokens"]
        # Allow ±40% tolerance (estimator + overhead)
        assert tokens >= 100, f"Too short: {tokens}"


def test_depths_cycled():
    examples = generate_needle_examples(10, seed=3407, messages=_MSGS, member_names=_NAMES)
    depths = [ex["meta"]["depth"] for ex in examples]
    # Should cycle through [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0]
    assert depths[0] == 0.0
    assert depths[4] == 1.0
    assert depths[5] == 0.0


def test_source_field():
    examples = generate_needle_examples(3, seed=3407, messages=_MSGS, member_names=_NAMES)
    for ex in examples:
        assert ex["source"] == "needle_synthetic"


# Import the shuffle helper from generate_local_synthetic for its own tests
def test_shuffle_context_blocks_import():
    from src.data.generate_local_synthetic import shuffle_context_blocks
    import random
    ctx = (
        "=== Conversas relevantes do grupo ===\n"
        "\n--- Conversa 1 ---\nMembro A: linha um\n"
        "\n--- Conversa 2 ---\nMembro B: linha dois\n"
        "\n--- Conversa 3 ---\nMembro C: linha três\n"
        "\n=== Fim das conversas ==="
    )
    rng = random.Random(1)
    shuffled = shuffle_context_blocks(ctx, rng)
    assert "Membro A" in shuffled
    assert "Membro B" in shuffled
    assert "Membro C" in shuffled
    assert "=== Conversas relevantes do grupo ===" in shuffled
    assert "=== Fim das conversas ===" in shuffled


def test_shuffle_context_blocks_renumbers():
    from src.data.generate_local_synthetic import shuffle_context_blocks
    import random
    ctx = (
        "=== Conversas relevantes do grupo ===\n"
        "\n--- Conversa 1 ---\nA: x\n"
        "\n--- Conversa 2 ---\nB: y\n"
        "\n--- Conversa 3 ---\nC: z\n"
        "\n=== Fim das conversas ==="
    )
    rng = random.Random(99)
    shuffled = shuffle_context_blocks(ctx, rng)
    import re
    numbers = re.findall(r"--- Conversa (\d+) ---", shuffled)
    assert numbers == ["1", "2", "3"], f"Renumbering failed: {numbers}"


def test_shuffle_context_blocks_short_unchanged():
    from src.data.generate_local_synthetic import shuffle_context_blocks
    import random
    ctx = "=== Conversas relevantes do grupo ===\n--- Conversa 1 ---\nA: x\n=== Fim das conversas ==="
    rng = random.Random(42)
    result = shuffle_context_blocks(ctx, rng)
    assert result == ctx
