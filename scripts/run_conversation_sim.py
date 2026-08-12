#!/usr/bin/env python3
"""Run a simulated group conversation against the bot, and report what broke.

Drives the `kaya-sim` instance — the production webhook path in mock mode — with
invented people played by a cheap cloud model, and asserts on the routing
decisions and replies that come back. See `src/testing/persona_sim.py` for why
the spine is scripted and only the filler is improvised.

    # once, before the first run
    kaya_chatbot_env/bin/python scripts/seed_sim_data.py
    docker compose --profile sim up -d kaya-sim

    kaya_chatbot_env/bin/python scripts/run_conversation_sim.py --preset smoke
    kaya_chatbot_env/bin/python scripts/run_conversation_sim.py --preset standard
    kaya_chatbot_env/bin/python scripts/run_conversation_sim.py --preset long_haul
    kaya_chatbot_env/bin/python scripts/run_conversation_sim.py --only images,audio

Exits non-zero if any assertion failed, so it can gate a deploy the way
`preflight_e2e.py` does. Reports land in reports/sim/<stamp>/.

Cost: personas run on grok-4.20-0309-non-reasoning ($1.25/$2.50 per M). A smoke
run is a few cents; long_haul is around a euro. Wall time is dominated by image
rendering (~90s each), not by tokens.
"""

import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent

# The persona model needs XAI_API_KEY, which lives in .env like every other
# secret here. Loaded explicitly so the script works without `set -a; . .env`.
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:  # pragma: no cover - dotenv is a dependency, but do not die on it
    pass

# Personas do not need to reason — they write "boa, bora" — and reasoning tokens
# bill as output. The judge is where reasoning earns its keep.
PERSONA_MODEL = "grok-4.20-0309-non-reasoning"
JUDGE_MODEL = "grok-4.20-0309-reasoning"


def log(message: str) -> None:
    print(message, flush=True)


def _check_generation_reachable(client, url: str) -> None:
    """Fail fast if the sim cannot actually generate.

    A sim that cannot reach its llama.cpp server does not error — it answers
    every beat with an empty string, and the run reports "0/6 turns answered, 6
    beats failed" as though the bot had misbehaved. The real cause was one line
    buried in the container log: `Failed to resolve 'llama'`, because kaya-sim
    runs in THIS compose project while the llama service belongs to the prod one
    (~/kaya-prod), so the two are on different networks and the service name does
    not resolve. One probe up front turns an hour of confusion into a message.
    """
    probe = {
        "event": "message",
        "payload": {"id": f"preflight_{uuid.uuid4().hex[:10]}",
                    "from": "349000000001@c.us", "body": "olá",
                    "notifyName": "preflight", "fromMe": False,
                    "timestamp": int(time.time())},
    }
    result, _elapsed = client.send(probe)
    if ((result or {}).get("reply") or "").strip():
        return
    log("!! the sim accepted a message but produced an empty reply.")
    log("   It usually means the app cannot reach its llama.cpp server. Check:")
    log("     docker logs kaya-sim 2>&1 | tail -20")
    log("   If it says \"Failed to resolve 'llama'\", kaya-sim is on this compose")
    log("   project's network while llama belongs to the prod one. Join them:")
    log("     docker network connect kaya-prod_default kaya-sim && docker restart kaya-sim")
    sys.exit(2)


def make_provider(config: Dict[str, Any], model: str):
    """A provider pinned to one model, without disturbing the shared config."""
    import copy

    from src.testing.conversation_tester import load_provider

    local = copy.deepcopy(config)
    local.setdefault("generation", {}).setdefault("xai", {})["model"] = model
    return load_provider("xai", local)


def summarise(result, elapsed: float) -> Dict[str, Any]:
    """Latency by beat kind, drop rates, and the assertion tally."""
    answered = [t for t in result.turns if t.handled]
    by_kind: Dict[str, List[float]] = {}
    for turn in result.turns:
        if turn.handled:
            by_kind.setdefault(turn.beat, []).append(turn.seconds)

    bursts = result.metrics.get("bursts", [])
    metrics = {
        "turns": len(result.turns),
        "answered": len(answered),
        "assertions_failed": sum(len(t.failures) for t in result.turns),
        "beats_failed": len(result.failures),
        "latency_by_beat": {
            kind: {"n": len(values), "median": round(statistics.median(values), 1),
                   "max": round(max(values), 1)}
            for kind, values in by_kind.items()
        },
        "burst_drop_rate": (
            round(sum(b["dropped"] for b in bursts) / max(sum(b["senders"] for b in bursts), 1), 3)
            if bursts else None
        ),
        "images": result.images,
        "wall_seconds": round(elapsed, 1),
    }
    metrics.update(result.metrics)
    return metrics


def write_report(run_dir: Path, result, config_note: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": result.scenario,
        "started": result.started,
        "metrics": result.metrics,
        "turns": [vars(t) for t in result.turns],
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                         encoding="utf-8")

    rows = []
    for turn in result.turns:
        status = "fail" if turn.failures else ("skip" if not turn.handled else "ok")
        detail = "<br>".join(turn.failures) if turn.failures else ""
        media = f'<span class="media">{turn.media}</span>' if turn.media else ""
        rows.append(
            f'<tr class="{status}"><td>{turn.index}</td><td>{turn.beat}</td>'
            f'<td>{turn.speaker}</td><td>{turn.chat}</td>'
            f'<td class="msg">{(turn.text or "")[:200]} {media}</td>'
            f'<td class="msg">{(turn.reply or "")[:400]}</td>'
            f'<td>{turn.command or ""}{("/" + turn.image) if turn.image else ""}</td>'
            f'<td>{turn.seconds}s</td><td class="why">{detail}{"<i>" + turn.note + "</i>" if turn.note else ""}</td></tr>'
        )

    metrics = result.metrics
    latency = "".join(
        f"<tr><td>{k}</td><td>{v['n']}</td><td>{v['median']}s</td><td>{v['max']}s</td></tr>"
        for k, v in (metrics.get("latency_by_beat") or {}).items())

    html = f"""<!doctype html><meta charset="utf-8">
<title>Kaya conversation sim — {run_dir.name}</title>
<style>
 body{{font:14px/1.55 system-ui,sans-serif;margin:2rem;background:#0f1115;color:#e6e6e6}}
 h1{{font-size:1.35rem;margin-bottom:.25rem}} .sub{{color:#8b95a5;margin-bottom:1.5rem}}
 table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}
 td,th{{padding:.45rem .6rem;border-bottom:1px solid #232833;vertical-align:top;text-align:left}}
 th{{color:#8b95a5;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
 .msg{{max-width:34rem}} .why{{color:#ff8080;font-size:12px;max-width:18rem}}
 .why i{{color:#6f7a8a;font-style:normal;display:block}}
 tr.fail td{{background:#2a1416}} tr.skip td{{color:#6f7a8a}}
 .media{{color:#7fb0ff;font-size:11px}}
 .cards{{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem}}
 .card{{background:#161a21;border:1px solid #232833;border-radius:8px;padding:.75rem 1rem;min-width:9rem}}
 .card b{{display:block;font-size:1.5rem;font-weight:600}}
 .card span{{color:#8b95a5;font-size:12px}}
</style>
<h1>Kaya conversation simulation</h1>
<div class="sub">{result.scenario} — {result.started} — {config_note}</div>
<div class="cards">
  <div class="card"><b>{metrics.get('turns', 0)}</b><span>turns</span></div>
  <div class="card"><b>{metrics.get('answered', 0)}</b><span>answered</span></div>
  <div class="card"><b>{metrics.get('beats_failed', 0)}</b><span>beats failed</span></div>
  <div class="card"><b>{metrics.get('assertions_failed', 0)}</b><span>assertions failed</span></div>
  <div class="card"><b>{metrics.get('burst_drop_rate')}</b><span>burst drop rate</span></div>
  <div class="card"><b>{metrics.get('wall_seconds', 0)}s</b><span>wall clock</span></div>
</div>
<table><tr><th>kind</th><th>n</th><th>median</th><th>max</th></tr>{latency}</table>
<table>
<tr><th>#</th><th>beat</th><th>who</th><th>chat</th><th>said</th><th>bot replied</th>
<th>route</th><th>took</th><th>failures / what this tests</th></tr>
{''.join(rows)}
</table>
"""
    (run_dir / "index.html").write_text(html, encoding="utf-8")
    log(f"\nreport: file://{run_dir / 'index.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a group conversation.")
    parser.add_argument("--preset", default="smoke",
                        choices=["smoke", "standard", "long_haul"])
    parser.add_argument("--only", default="", help="comma-separated scenario names")
    parser.add_argument("--url", default="http://127.0.0.1:7862")
    parser.add_argument("--media-host", default="172.18.0.1",
                        help="address the sim container reaches this host on")
    parser.add_argument("--media-port", type=int, default=8899)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import random

    random.seed(args.seed)

    from src.config_loader import load_config
    from src.testing import scenarios
    from src.testing.persona_sim import (
        MediaPool, MediaServer, PersonaDriver, SimClient, SimResult, SimRunner)
    from src.testing.sim_world import personas

    config = load_config(str(BASE_DIR / "config.yaml"))
    client = SimClient(args.url)

    try:
        health = client.health()
    except Exception as exc:  # noqa: BLE001
        log(f"!! cannot reach the sim at {args.url}: {exc}")
        log("   start it with: docker compose --profile sim up -d kaya-sim")
        sys.exit(2)
    if not health.get("mock"):
        log("!! the target is NOT in mock mode — refusing to run.")
        log("   A non-mock target would send these messages to real people.")
        sys.exit(2)
    _check_generation_reachable(client, args.url)
    log(f"target {args.url} — mock={health.get('mock')}")

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    beats = scenarios.build(args.preset, only)
    people = personas(scenarios.PRESETS[args.preset]["people"])
    log(f"preset {args.preset}: {len(beats)} beats, {len(people)} people "
        f"({', '.join(p.name for p in people)})")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = BASE_DIR / "reports" / "sim" / stamp

    server = MediaServer(run_dir / "media", host=args.media_host, port=args.media_port)
    server.start()
    pool = MediaPool(server)
    log(f"media: {len(pool.photos)} photos, {len(pool.voices)} voice notes")

    driver = PersonaDriver(make_provider(config, PERSONA_MODEL), PERSONA_MODEL)
    runner = SimRunner(client, driver, pool, people, log=log)
    result = SimResult(scenario=f"{args.preset}:{','.join(only or ['all'])}",
                       started=stamp)

    started = time.time()
    try:
        runner.run(beats, result)
    except KeyboardInterrupt:
        log("\ninterrupted — writing what we have")
    finally:
        server.stop()

    result.metrics = summarise(result, time.time() - started)
    write_report(run_dir, result,
                 f"personas: {PERSONA_MODEL}, {driver.calls} calls")

    log(f"\n{result.metrics['answered']}/{result.metrics['turns']} turns answered, "
        f"{result.metrics['beats_failed']} beats failed "
        f"({result.metrics['assertions_failed']} assertions)")
    for turn in result.failures:
        log(f"  ✗ [{turn.index}] {turn.beat} {turn.speaker}: {turn.text[:60]!r}")
        for failure in turn.failures:
            log(f"      {failure}")
        if turn.note:
            log(f"      (testing: {turn.note})")

    sys.exit(1 if result.failures else 0)


if __name__ == "__main__":
    main()
