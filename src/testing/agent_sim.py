"""Grok-driven multi-agent conversation simulator.

Simulates a small group of "member persona" agents (powered by Grok/XAI) having a
natural conversation with the live KayaBot (local GPU model + RAG). Measures:
- Per-turn latency for the local model
- Memory retention: planted facts are injected early; a probe question is asked later
- Answer quality: every bot turn is scored by the Grok judge on 6 dimensions

Each agent has a persona system prompt built from group_members.json key_facts. The
bot's `generate_reply` path is used (web-search is disabled so the local model always
responds). History accumulates as "<who>: <text>" lines, mirroring the production path.

Usage (from scripts/run_agent_sim.py):
    from src.testing.agent_sim import AgentSim, SimConfig
    sim = AgentSim(engine, provider, config)
    result = sim.run(SimConfig(agents=3, turns=20))
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.testing.conversation_tester import (
    SCORING_SYSTEM_PROMPT,
    ScoreBreakdown,
    check_identity_leaks,
    parse_scores,
)


@dataclass
class SimConfig:
    num_agents: int = 3
    turns: int = 20
    max_new_tokens: int = 256
    history_window: int = 20
    planted_facts: Optional[List[Dict[str, str]]] = None


@dataclass
class TurnRecord:
    turn: int
    speaker: str
    message: str
    bot_reply: Optional[str] = None
    latency_s: Optional[float] = None
    scores: Optional[ScoreBreakdown] = None
    identity_leaks: List[str] = field(default_factory=list)
    probe: bool = False
    probe_recalled: Optional[bool] = None


@dataclass
class SimResult:
    config: SimConfig
    turns: List[TurnRecord]
    memory_recall_pct: Optional[float] = None
    avg_latency_s: Optional[float] = None
    avg_scores: Optional[Dict[str, float]] = None
    raw_transcript: List[str] = field(default_factory=list)


_AGENT_SYSTEM_TMPL = """\
You are {name}, a member of the Kaya friend group having a casual WhatsApp group chat.
Speak naturally in European Portuguese (with occasional English). You are NOT the bot.
Keep messages short (1-3 sentences) as in a real chat.

Key things about you:
{key_facts}

Current context: you are chatting in the group. Sometimes ask questions that only
the bot (Kaya) can answer (about other members, group history, events). Sometimes
make normal chat small talk. Never speak *as* the bot.
"""

_PROBE_SYSTEM_TMPL = """\
You are {name}. Ask a single natural, curiosity-driven question about what you said
earlier — specifically about the detail: {fact}. Make it sound like you mentioned it
to the group and are now checking if anyone (or the bot) remembers.
Keep it to one sentence, natural WhatsApp style in European Portuguese.
"""

_NEXT_MESSAGE_PROMPT = """\
Continue the group chat as {name}. Reply to the last message or start a new topic.
Keep it short (1-3 sentences), natural, in European Portuguese.

Chat so far:
{history}

Your message:"""

_JUDGE_CONVERSATION_SYSTEM = """\
You are evaluating KayaBot's responses in a simulated group chat.
KayaBot is a bot (NOT a group member) with access to group history.

Score each response using the standard 6-dimension rubric.
""" + SCORING_SYSTEM_PROMPT


def _build_member_personas(members_file: str, num_agents: int) -> List[Dict[str, Any]]:
    """Load member profiles and build agent persona dicts."""
    data = json.loads(Path(members_file).read_text(encoding="utf-8"))
    personas = []
    for member in data[:num_agents]:
        name = member.get("name", "Membro")
        key_facts = member.get("key_facts", [])
        if isinstance(key_facts, list):
            facts_str = "\n".join(f"- {f}" for f in key_facts[:6])
        else:
            facts_str = str(key_facts)
        personas.append({
            "name": name,
            "system_prompt": _AGENT_SYSTEM_TMPL.format(name=name, key_facts=facts_str),
            "key_facts": key_facts,
        })
    return personas


def _agent_next_message(provider, persona: Dict, history: List[str]) -> str:
    """Drive a persona agent to produce its next chat message."""
    history_str = "\n".join(history[-12:]) if history else "(conversa vazia)"
    messages = [
        {"role": "system", "content": persona["system_prompt"]},
        {"role": "user", "content": _NEXT_MESSAGE_PROMPT.format(
            name=persona["name"], history=history_str
        )},
    ]
    return provider.chat_completion(messages).strip()


def _agent_probe_message(provider, persona: Dict, fact: str) -> str:
    """Ask the persona to generate a probe question about a planted fact."""
    system = _PROBE_SYSTEM_TMPL.format(name=persona["name"], fact=fact)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Generate your probe question:"},
    ]
    return provider.chat_completion(messages).strip()


def _score_bot_reply(provider, question: str, reply: str, context: str = "") -> ScoreBreakdown:
    """Judge a single bot reply using the standard 6-dim rubric."""
    reference = context or "Grupo de amigos portugueses; resposta deve ser em terceira pessoa."
    user_msg = (
        f"Reference knowledge:\n{reference}\n\n"
        f"Question asked:\n{question}\n\n"
        f"Response to evaluate:\n{reply}\n\n"
        "Provide your scores as a JSON object."
    )
    messages = [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = provider.chat_completion(messages)
    return parse_scores(raw)


class AgentSim:
    """Runs a simulated group conversation and collects quality metrics."""

    def __init__(self, engine, provider, config: Dict[str, Any]):
        self.engine = engine
        self.provider = provider
        self.config = config
        members_file = config.get("data", {}).get("group_members_file", "data/group_members.json")
        config_path = str(Path(__file__).parent.parent.parent / "config.yaml")
        self._members_path = members_file if Path(members_file).is_absolute() else str(
            Path(config_path).parent / members_file
        )

    def run(self, sim_config: SimConfig, system_prompt: str) -> SimResult:
        personas = _build_member_personas(self._members_path, sim_config.num_agents)
        planted = sim_config.planted_facts or self._default_planted_facts(personas)

        history: List[str] = []
        turn_records: List[TurnRecord] = []
        raw_transcript: List[str] = []

        probe_turn = max(10, int(sim_config.turns * 0.7))
        planted_at_turn = 3
        probes_done: List[Dict] = []

        for turn_idx in range(sim_config.turns):
            persona = personas[turn_idx % len(personas)]

            if turn_idx == planted_at_turn and planted:
                for pf in planted:
                    plant_msg = pf["message"]
                    history.append(f"{persona['name']}: {plant_msg}")
                    raw_transcript.append(f"[PLANTED] {persona['name']}: {plant_msg}")
                    turn_records.append(TurnRecord(turn=turn_idx, speaker=persona["name"], message=plant_msg))

            is_probe = (turn_idx == probe_turn and planted)
            if is_probe:
                pf = planted[0]
                message = _agent_probe_message(self.provider, persona, pf["fact"])
            else:
                message = _agent_next_message(self.provider, persona, history)

            history.append(f"{persona['name']}: {message}")
            raw_transcript.append(f"{persona['name']}: {message}")

            t0 = time.perf_counter()
            recent_lines = history[-sim_config.history_window:]
            bot_reply = self.engine.generate_reply(
                message,
                speaker=persona["name"],
                recent_lines=recent_lines,
                system_prompt=system_prompt,
                max_new_tokens=sim_config.max_new_tokens,
            )
            latency_s = time.perf_counter() - t0

            history.append(f"Kaya: {bot_reply}")
            raw_transcript.append(f"Kaya: {bot_reply}")

            leaks = check_identity_leaks(bot_reply)
            scores = None
            try:
                scores = _score_bot_reply(self.provider, message, bot_reply)
            except Exception as exc:
                print(f"  ⚠️  scoring failed turn {turn_idx}: {exc}")

            probe_recalled = None
            if is_probe and planted:
                probe_recalled = planted[0]["recall_token"] in bot_reply
                probes_done.append({"fact": planted[0]["fact"], "recalled": probe_recalled})

            rec = TurnRecord(
                turn=turn_idx,
                speaker=persona["name"],
                message=message,
                bot_reply=bot_reply,
                latency_s=round(latency_s, 3),
                scores=scores,
                identity_leaks=leaks,
                probe=is_probe,
                probe_recalled=probe_recalled,
            )
            turn_records.append(rec)

            status = "✓" if (scores and not scores.failed) else "✗"
            probe_str = f" [PROBE recalled={'✓' if probe_recalled else '✗'}]" if is_probe else ""
            print(
                f"  turn {turn_idx:3d} | {persona['name'][:10]:10s} | "
                f"lat={latency_s:.1f}s | judge={status}"
                f"{probe_str}",
                flush=True,
            )

        scored = [r for r in turn_records if r.scores is not None]
        dims = ["factual_accuracy", "relevance", "language_quality", "tone", "identity_adherence", "factual_grounding"]
        avg_scores = None
        if scored:
            avg_scores = {}
            for dim in dims:
                avg_scores[dim] = round(sum(getattr(r.scores, dim) for r in scored) / len(scored), 3)
            avg_scores["extended_average"] = round(sum(r.scores.extended_average for r in scored) / len(scored), 3)

        avg_latency = None
        latencies = [r.latency_s for r in turn_records if r.latency_s is not None]
        if latencies:
            avg_latency = round(sum(latencies) / len(latencies), 3)

        memory_recall_pct = None
        if probes_done:
            memory_recall_pct = round(
                100.0 * sum(1 for p in probes_done if p["recalled"]) / len(probes_done), 1
            )

        return SimResult(
            config=sim_config,
            turns=turn_records,
            memory_recall_pct=memory_recall_pct,
            avg_latency_s=avg_latency,
            avg_scores=avg_scores,
            raw_transcript=raw_transcript,
        )

    def _default_planted_facts(self, personas: List[Dict]) -> List[Dict]:
        if not personas:
            return []
        return [
            {
                "fact": "car purchase",
                "message": f"Pessoal, finalmente comprei o meu primeiro carro — um Renault Megane branco!",
                "recall_token": "Megane",
            }
        ]
