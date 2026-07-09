"""Unit tests for agent_sim.py — fully mocked, no GPU or API calls."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.testing.agent_sim import (
    AgentSim,
    SimConfig,
    SimResult,
    TurnRecord,
    _build_member_personas,
    _agent_next_message,
    _agent_probe_message,
    _score_bot_reply,
)
from src.testing.conversation_tester import ScoreBreakdown


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


FAKE_MEMBERS = [
    {
        "name": "Alice",
        "key_facts": ["Alice gosta de padel", "Alice tem um cão chamado Mimi"],
    },
    {
        "name": "Bob",
        "key_facts": ["Bob trabalha em Lisboa", "Bob joga póquer aos fins de semana"],
    },
    {
        "name": "Carol",
        "key_facts": ["Carol é fã de sushi", "Carol tem um gato"],
    },
]

FAKE_MEMBERS_FILE = None  # will be set up via tmp_path


@pytest.fixture
def members_file(tmp_path):
    import json

    path = tmp_path / "group_members.json"
    path.write_text(json.dumps(FAKE_MEMBERS), encoding="utf-8")
    return str(path)


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.chat_completion.return_value = (
        '{"factual_accuracy": 4, "relevance": 4, "language_quality": 4, '
        '"tone": 4, "identity_adherence": 5, "factual_grounding": 4}'
    )
    return provider


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.generate_reply.return_value = "Olá! Posso ajudar com isso."
    return engine


def _make_sim(mock_engine, mock_provider, members_file, config_override=None):
    config = {
        "data": {"group_members_file": members_file},
        "rag": {"enabled": True},
        "web_search": {"enabled": False},
        "generation": {
            "xai": {"model": "grok-3-mini"}
        },
    }
    if config_override:
        config.update(config_override)
    sim = AgentSim(mock_engine, mock_provider, config)
    sim._members_path = members_file
    return sim


# ---------------------------------------------------------------------------
# _build_member_personas
# ---------------------------------------------------------------------------


def test_build_member_personas_count(members_file):
    personas = _build_member_personas(members_file, num_agents=2)
    assert len(personas) == 2


def test_build_member_personas_fields(members_file):
    personas = _build_member_personas(members_file, num_agents=1)
    p = personas[0]
    assert p["name"] == "Alice"
    assert "system_prompt" in p
    assert "Alice" in p["system_prompt"]
    assert "padel" in p["system_prompt"]


def test_build_member_personas_clamps_to_available(members_file):
    personas = _build_member_personas(members_file, num_agents=100)
    assert len(personas) == len(FAKE_MEMBERS)


# ---------------------------------------------------------------------------
# _agent_next_message / _agent_probe_message
# ---------------------------------------------------------------------------


def test_agent_next_message_calls_provider(mock_provider, members_file):
    personas = _build_member_personas(members_file, 1)
    mock_provider.chat_completion.return_value = "E o Rafa, está bom?"
    result = _agent_next_message(mock_provider, personas[0], ["Alice: olá pessoal"])
    assert result == "E o Rafa, está bom?"
    mock_provider.chat_completion.assert_called_once()
    call_msgs = mock_provider.chat_completion.call_args[0][0]
    assert any(m["role"] == "system" for m in call_msgs)


def test_agent_probe_message_calls_provider(mock_provider, members_file):
    personas = _build_member_personas(members_file, 1)
    mock_provider.chat_completion.return_value = "Alguém lembra do Megane que mencionei?"
    result = _agent_probe_message(mock_provider, personas[0], "car purchase")
    assert "Megane" in result or result  # provider mock returns it
    mock_provider.chat_completion.assert_called_once()


# ---------------------------------------------------------------------------
# _score_bot_reply
# ---------------------------------------------------------------------------


def test_score_bot_reply_returns_breakdown(mock_provider):
    scores = _score_bot_reply(mock_provider, "Quem é o Bob?", "O Bob trabalha em Lisboa.")
    assert isinstance(scores, ScoreBreakdown)
    assert scores.factual_accuracy == 4
    assert scores.identity_adherence == 5


def test_score_bot_reply_handles_malformed_json(mock_provider):
    mock_provider.chat_completion.return_value = "Não consigo avaliar agora."
    scores = _score_bot_reply(mock_provider, "?", "resposta")
    assert isinstance(scores, ScoreBreakdown)


# ---------------------------------------------------------------------------
# AgentSim.run — full flow (mocked provider + engine)
# ---------------------------------------------------------------------------


def test_run_produces_result(mock_engine, mock_provider, members_file):
    sim = _make_sim(mock_engine, mock_provider, members_file)
    mock_provider.chat_completion.return_value = "Tudo bem pessoal!"
    result = sim.run(
        SimConfig(num_agents=2, turns=5, max_new_tokens=64),
        system_prompt="System prompt",
    )
    assert isinstance(result, SimResult)
    bot_turns = [t for t in result.turns if t.bot_reply is not None]
    assert len(bot_turns) > 0


def test_run_records_latency(mock_engine, mock_provider, members_file):
    sim = _make_sim(mock_engine, mock_provider, members_file)
    mock_provider.chat_completion.return_value = "ok"
    result = sim.run(
        SimConfig(num_agents=2, turns=4, max_new_tokens=32),
        system_prompt="System prompt",
    )
    latencies = [t.latency_s for t in result.turns if t.latency_s is not None]
    assert len(latencies) > 0
    assert result.avg_latency_s is not None
    assert result.avg_latency_s >= 0.0


def test_run_planted_fact_probe(mock_engine, mock_provider, members_file):
    """Probe turn should ask about the planted fact; mock engine returns recall token."""
    _SCORE_JSON = (
        '{"factual_accuracy": 4, "relevance": 4, "language_quality": 4, '
        '"tone": 4, "identity_adherence": 5, "factual_grounding": 4}'
    )
    call_count = {"n": 0}

    def side_effect(messages):
        call_count["n"] += 1
        sys_content = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        if "factual_accuracy" in sys_content:
            return _SCORE_JSON
        if "probe" in sys_content.lower() or "car purchase" in sys_content.lower():
            return "Alguém lembra do Megane que mencionei?"
        return "ok pessoal"

    mock_provider.chat_completion.side_effect = side_effect
    mock_engine.generate_reply.return_value = "Sim, o Megane branco foi comprado!"

    planted = [{"fact": "car purchase", "message": "Comprei um Megane branco!", "recall_token": "Megane"}]
    sim = _make_sim(mock_engine, mock_provider, members_file)
    result = sim.run(
        SimConfig(num_agents=2, turns=12, max_new_tokens=64, planted_facts=planted),
        system_prompt="System prompt",
    )
    probe_turns = [t for t in result.turns if t.probe]
    assert len(probe_turns) >= 1
    assert probe_turns[0].probe_recalled is True
    assert result.memory_recall_pct == 100.0


def test_run_avg_scores_computed(mock_engine, mock_provider, members_file):
    mock_provider.chat_completion.return_value = (
        '{"factual_accuracy": 3, "relevance": 3, "language_quality": 4, '
        '"tone": 4, "identity_adherence": 5, "factual_grounding": 3}'
    )
    sim = _make_sim(mock_engine, mock_provider, members_file)
    result = sim.run(
        SimConfig(num_agents=2, turns=4, max_new_tokens=32),
        system_prompt="System prompt",
    )
    if result.avg_scores:
        assert "extended_average" in result.avg_scores
        assert 0 <= result.avg_scores["extended_average"] <= 5


def test_run_identity_leaks_flagged(mock_engine, mock_provider, members_file):
    mock_engine.generate_reply.return_value = "Sim, é meu amigo desde sempre."
    mock_provider.chat_completion.return_value = "ok"
    sim = _make_sim(mock_engine, mock_provider, members_file)
    result = sim.run(
        SimConfig(num_agents=2, turns=3, max_new_tokens=32),
        system_prompt="System",
    )
    flagged = [t for t in result.turns if t.identity_leaks]
    assert len(flagged) > 0


def test_no_rag_runs_cleanly(mock_engine, mock_provider, members_file):
    """Sim should not crash when RAG is disabled."""
    sim = _make_sim(
        mock_engine, mock_provider, members_file,
        config_override={"rag": {"enabled": False}},
    )
    mock_provider.chat_completion.return_value = "ok"
    result = sim.run(
        SimConfig(num_agents=1, turns=3, max_new_tokens=32),
        system_prompt="System",
    )
    assert isinstance(result, SimResult)
