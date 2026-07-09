#!/usr/bin/env python3
"""CLI runner for the Grok-driven multi-agent conversation simulator.

Fires up N member persona agents (Grok/XAI) that chat with the live KayaBot
(local GPU model + RAG), measuring memory retention, answer quality, and
per-turn latency. Results are written as JSON to reports/benchmarks/.

    # needs GPU free + XAI_API_KEY in .env
    kaya_chatbot_env/bin/python scripts/run_agent_sim.py
    kaya_chatbot_env/bin/python scripts/run_agent_sim.py --agents 3 --turns 20
    kaya_chatbot_env/bin/python scripts/run_agent_sim.py --no-rag   # ablation
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.chat.engine import get_engine, build_system_prompt
from src.testing.conversation_tester import load_provider
from src.testing.agent_sim import AgentSim, SimConfig


def _log(msg: str) -> None:
    print(msg, flush=True)


def _anonymize_transcript(transcript: list) -> list:
    """Return transcript with real member names replaced by Member A/B/C etc.

    Names are loaded from group_members.json at runtime and swapped consistently.
    Falls back to the raw transcript if the file can't be read.
    """
    try:
        import json as _json
        members_file = Path(__file__).parent.parent / "data" / "group_members.json"
        data = _json.loads(members_file.read_text(encoding="utf-8"))
        labels = [chr(65 + i) for i in range(len(data))]
        mapping = {m["name"]: f"Member {labels[i]}" for i, m in enumerate(data)}
        result = []
        for line in transcript:
            for real, anon in mapping.items():
                line = line.replace(real, anon)
            result.append(line)
        return result
    except Exception:
        return transcript


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-agent conversation simulator")
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--history-window", type=int, default=20)
    ap.add_argument("--no-rag", action="store_true", help="Disable RAG for this run")
    args = ap.parse_args()

    config_path = str(Path(__file__).parent.parent / "config.yaml")
    config = load_config(config_path)

    config["web_search"] = {**config.get("web_search", {}), "enabled": False}
    if args.no_rag:
        config["rag"] = {**config.get("rag", {}), "enabled": False}

    _log("Loading engine …")
    engine = get_engine(config)
    system_prompt = build_system_prompt(config, config_path, include_uncensored=False)

    _log("Loading XAI provider …")
    provider = load_provider("xai", config)

    sim = AgentSim(engine, provider, config)
    sim_config = SimConfig(
        num_agents=args.agents,
        turns=args.turns,
        max_new_tokens=args.max_new_tokens,
        history_window=args.history_window,
    )

    _log(f"\n=== Agent Simulation ({args.agents} agents, {args.turns} turns) ===")
    result = sim.run(sim_config, system_prompt)

    _log("\n" + "=" * 60)
    _log("RESULTS")
    _log("=" * 60)
    _log(f"  Memory recall:    {result.memory_recall_pct}%")
    _log(f"  Avg latency:      {result.avg_latency_s}s / turn")
    if result.avg_scores:
        _log("  Judge scores (avg):")
        for dim, val in result.avg_scores.items():
            _log(f"    {dim:<25} {val:.3f}")
    _log("=" * 60)

    turn_dicts = []
    for tr in result.turns:
        d: dict = {
            "turn": tr.turn,
            "speaker": tr.speaker,
            "message": tr.message,
            "probe": tr.probe,
        }
        if tr.bot_reply is not None:
            d["bot_reply"] = tr.bot_reply
        if tr.latency_s is not None:
            d["latency_s"] = tr.latency_s
        if tr.scores is not None:
            d["scores"] = tr.scores.to_dict()
        if tr.identity_leaks:
            d["identity_leaks"] = tr.identity_leaks
        if tr.probe_recalled is not None:
            d["probe_recalled"] = tr.probe_recalled
        turn_dicts.append(d)

    out_dir = Path(config.get("benchmark", {}).get("output_dir", "reports/benchmarks/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"agent_sim_{stamp}.json"

    anon_transcript = _anonymize_transcript(result.raw_transcript)

    out_path.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "config": {
                    "num_agents": args.agents,
                    "turns": args.turns,
                    "max_new_tokens": args.max_new_tokens,
                    "history_window": args.history_window,
                    "rag_enabled": config.get("rag", {}).get("enabled", True),
                    "web_search_enabled": False,
                },
                "summary": {
                    "memory_recall_pct": result.memory_recall_pct,
                    "avg_latency_s": result.avg_latency_s,
                    "avg_scores": result.avg_scores,
                },
                "anonymized_transcript": anon_transcript,
                "turns": turn_dicts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
