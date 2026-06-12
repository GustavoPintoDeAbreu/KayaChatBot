"""Orchestration test for generate_local_synthetic.generate_dataset.

Proves the full generate→filter→format loop end-to-end with stubbed teacher and
retriever (no model load), and that bad outputs (refusal, echo, emoji-only) are
filtered while good ones are formatted for the merge pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generate_local_synthetic import generate_dataset, build_user_turn


class StubRetriever:
    def __init__(self, context=""):
        self._ctx = context

    def retrieve_all(self, query, knowledge_approach="json_only"):
        return self._ctx


class StubTeacher:
    """Returns a scripted answer per question (by substring match)."""

    def __init__(self, mapping, default="resposta genérica"):
        self.mapping = mapping
        self.default = default
        self.calls = []

    def generate(self, system_prompt, user_message):
        self.calls.append((system_prompt, user_message))
        for key, val in self.mapping.items():
            if key in user_message:
                return val
        return self.default


def test_build_user_turn_includes_context():
    r = StubRetriever("=== ctx ===")
    turn, ctx = build_user_turn("Quem é o Gil?", r, "json_only")
    assert "=== ctx ===" in turn and turn.endswith("Quem é o Gil?")
    assert ctx == "=== ctx ==="


def test_build_user_turn_no_retriever():
    turn, ctx = build_user_turn("Olá", None, "json_only")
    assert turn == "Olá" and ctx == ""


def test_generate_dataset_filters_and_formats():
    questions = ["good", "refusal", "echo", "emoji"]
    teacher = StubTeacher({
        "good": "O Gustavo é claramente o mais convencido, dá para ver pelo tom dele.",
        "refusal": "Como assistente, não tenho opiniões sobre isso.",
        "echo": "isto é exatamente o contexto recortado",
        "emoji": "😊😊😊",
    })
    retriever = StubRetriever("isto é exatamente o contexto recortado")

    stats = generate_dataset(
        questions, retriever, teacher,
        base_system_prompt="PERSONA", member_suffix=" SUFFIX",
        knowledge_approach="json_only", min_words=5,
    )

    assert stats["asked"] == 4
    assert stats["accepted"] == 1          # only "good" survives
    assert stats["rejected"] == 3
    ex = stats["examples"][0]
    assert ex["source"] == "synthetic_local"
    assert ex["conversations"][0]["role"] == "user"
    assert ex["conversations"][1]["role"] == "assistant"
    assert "Gustavo" in ex["conversations"][1]["content"]
    # generation instruction + persona were sent to the teacher
    assert "PERSONA SUFFIX" in teacher.calls[0][0]
    assert "INSTRUÇÕES DE GERAÇÃO" in teacher.calls[0][0]


def test_generate_dataset_streams_via_callback():
    captured = []
    teacher = StubTeacher({}, default="Uma resposta sintetizada com algum detalhe e opinião.")
    stats = generate_dataset(
        ["q1", "q2"], StubRetriever(""), teacher,
        base_system_prompt="P", on_example=captured.append, min_words=4,
    )
    assert len(captured) == 2
    assert stats["accepted"] == 2
    assert stats["examples"] == []  # streamed, not collected
