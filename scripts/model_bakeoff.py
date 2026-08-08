#!/usr/bin/env python3
"""Unattended serving bake-off across candidate models and GPU configurations.

Answers, with measurements rather than intuition: what is the best model this box
can actually serve, now that both 3090s can hold one model (~45GB of weights+KV
via llama.cpp layer split)?

For each arm it brings up the `llama-bench` compose service on the arm's GGUF and
flags, waits for /health, runs the three existing harnesses against it, records
throughput and per-card VRAM, then tears the server down before the next arm.
Strictly sequential — the two GPUs are a single shared resource.

Modelled on scripts/heretic_sweep.py (same run()/newest() subprocess-and-report
pattern), with the additions a multi-hour unattended run needs: the scorecard is
written after every arm, arms are resumable, a missing GGUF or a dead server is
recorded and skipped instead of aborting the sweep.

    kaya_chatbot_env/bin/python scripts/model_bakeoff.py               # all arms
    kaya_chatbot_env/bin/python scripts/model_bakeoff.py --only 31b-q8,70b-abl-iq4
    kaya_chatbot_env/bin/python scripts/model_bakeoff.py --resume reports/benchmarks/bakeoff_<stamp>.json
    kaya_chatbot_env/bin/python scripts/model_bakeoff.py --list

The `base-e4b` arm is current prod, re-scored under the same judge as every
candidate — that row, not the historical BASELINE_SUMMARY.md numbers, is what
candidates must beat. (Those were scored by the xai judge, which is out of
credits; judges are not interchangeable, so the old 3.068 is context only.)

Reading the scorecard: every arm except `base-e4b` is a STOCK instruct model with
no LoRA, so it will lose on the golden identity/voice dimensions and refuse more
often. Select on knowledge, reasoning, context recall and tok/s — the composite
only becomes decisive after the finalists are fine-tuned.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

BASE_DIR = Path(__file__).parent.parent
PYTHON = str(BASE_DIR / "kaya_chatbot_env" / "bin" / "python")
REPORTS = BASE_DIR / "reports" / "benchmarks"
LOGS = BASE_DIR / "logs" / "bakeoff"
GGUF_DIR = BASE_DIR / "models" / "gguf"

BENCH_URL = "http://127.0.0.1:8081"
BENCH_CONTAINER = "kaya-llama-bench"

# Historical reference from BASELINE_SUMMARY.md (current prod, E4B + LoRA at Q6_K).
# NOTE: those golden numbers were produced by the *xai* judge, which is now out of
# credits. Judges are not interchangeable — an azure-scored run cannot be compared
# to an xai-scored one. That is why `base-e4b` (current prod) is an arm in this
# sweep: it re-scores the live model under the same judge as every candidate, and
# IT is the number to compare against. The values below are context, not the gate.
BASELINE = {"golden_extended_average": 3.068, "context_recall_pct": 95.0,
            "refusal_rate": 0.15, "golden_judge": "xai"}
REFERENCE_ARM = "base-e4b"


@dataclass
class Arm:
    """One serving configuration to score.

    gguf        filename inside models/gguf (what llama-bench mounts at /models)
    tokenizer   HF repo id or local dir — used ONLY for chat templating under the
                gguf backend, so a stock repo works with no adapter on disk
    ctx         llama-server -c
    fa          flash attention: "off" for Gemma-4 (its KV layout requires it),
                "on" for Llama / MoE (faster, smaller KV cache)
    extra       extra llama-server flags (-ts split bias, --parallel, --n-cpu-moe)
    """
    tag: str
    tier: str
    gguf: str
    tokenizer: str
    ctx: int = 4096
    fa: str = "off"
    extra: str = ""
    est_gb: float = 0.0
    seq_lengths: List[int] = field(default_factory=lambda: [2048, 4096])
    note: str = ""
    sm: str = "layer"      # "none" = keep the whole model on one card


# GPU0 drives the desktop, is capped at 250W and burn-in measured its sustained
# clocks ~15% below GPU1's (1238 vs 1448 MHz avg). It is therefore the slow half
# of any layer split, so two-card arms give it slightly fewer layers.
TS_BIAS = "-ts 0.45,0.55"
# Tier A exists to answer "what does ONE card give you", so those arms keep the
# whole model on GPU1 (-sm none). Left to split, llama.cpp spreads even a 10GB
# model over both cards and charges a cross-PCIe hop per token, which understates
# single-card throughput. GPU1 is the compute-clean card (no desktop, 280W).
SINGLE_GPU = "--main-gpu 1"
# One slot, so the whole KV budget serves a single request instead of being
# divided four ways (llama-server defaults to 4 slots).
ONE_SLOT = "--parallel 1"

# The server's -c must exceed the largest sequence a recall sweep builds, or the
# deepest cells overflow the context and are recorded as failures rather than as
# genuine recall misses. Arms therefore run with roughly 2x headroom over their
# max seq_length.
ARMS: List[Arm] = [
    # ---- Tier A: fits one card. Establishes what the second GPU is worth. ----
    Arm("base-e4b", "A", "kaya-wpp-Q6_K.gguf",
        "models/kaya_gemma4_heretic_seq4096_wpp", 8192, "off", f"{ONE_SLOT} {SINGLE_GPU}", 5.8,
        note="REFERENCE: current prod, E4B + WhatsApp LoRA, abliterated base", sm="none"),
    Arm("12b-q6", "A", "gemma-4-12b-it-Q6_K.gguf",
        "unsloth/gemma-4-12b-it", 8192, "off", f"{ONE_SLOT} {SINGLE_GPU}", 9.8, sm="none"),
    Arm("26b-a4b-q4", "A", "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        "unsloth/gemma-4-26B-A4B-it", 8192, "off", f"{ONE_SLOT} {SINGLE_GPU}", 14.2,
        note="MoE ~4B active: near-E4B speed, far more knowledge", sm="none"),
    Arm("26b-a4b-q6", "A", "gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf",
        "unsloth/gemma-4-26B-A4B-it", 8192, "off", f"{ONE_SLOT} {SINGLE_GPU}", 23.3, sm="none"),
    Arm("31b-q4", "A", "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        "unsloth/gemma-4-31B-it", 8192, "off", f"{ONE_SLOT} {SINGLE_GPU}", 17.3,
        note="QAT: Q4 footprint at near-BF16 quality", sm="none"),

    # ---- Tier B: needs both cards. The reason the second GPU was bought. ----
    Arm("31b-q5", "B", "gemma-4-31B-it-Q5_K_M.gguf",
        "unsloth/gemma-4-31B-it", 8192, "off", f"{TS_BIAS} {ONE_SLOT}", 21.7),
    Arm("31b-q8", "B", "gemma-4-31B-it-Q8_0.gguf",
        "unsloth/gemma-4-31B-it", 8192, "off", f"{TS_BIAS} {ONE_SLOT}", 32.6,
        note="precision arm: what 2 cards buy in quality at fixed size"),
    # Two context arms off the same file. Gemma's KV cost at 64K is uncertain
    # (it alternates local sliding-window and global attention, so the naive
    # per-layer estimate badly overshoots), and 17.3GB of weights + a 64K cache
    # may not fit in ~45GB. Running 32K as well means a 64K OOM still leaves a
    # measured answer instead of a hole.
    Arm("31b-ctx32k", "B", "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        "unsloth/gemma-4-31B-it", 32768, "off", f"{TS_BIAS} {ONE_SLOT}", 17.3,
        seq_lengths=[4096, 8192, 16384, 24576],
        note="CONTEXT arm: rag.max_context_tokens is 2500 today and the measured "
             "reliable window was ~2360 — this tests whether that ceiling moves."),
    Arm("31b-ctx64k", "B", "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        "unsloth/gemma-4-31B-it", 65536, "off", f"{TS_BIAS} {ONE_SLOT}", 17.3,
        seq_lengths=[16384, 32768, 49152],
        note="CONTEXT arm, upper bound: if the server OOMs here, 31b-ctx32k is "
             "the answer and that is itself the finding."),
    # The 31B held 100% needle recall out to 27,413 tokens at -c 32768, ~11.6x
    # prod's measured ~2360 window, using 27.0GB (both cards). These ask whether
    # the cheaper models do the same: 12B at Q6 is 11.8GB, so 12B + a 32K cache
    # should fit ONE card — which would mean the big context costs no second GPU
    # at all. Same seq ladder as 31b-ctx32k so the numbers are comparable.
    Arm("12b-ctx32k", "B", "gemma-4-12b-it-Q6_K.gguf",
        "unsloth/gemma-4-12b-it", 32768, "off", f"{ONE_SLOT} {SINGLE_GPU}", 9.8,
        seq_lengths=[4096, 8192, 16384, 24576], sm="none",
        note="CONTEXT on one card: quality leader + 32K cache, single GPU."),
    Arm("26b-a4b-ctx32k", "B", "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        "unsloth/gemma-4-26B-A4B-it", 32768, "off", f"{TS_BIAS} {ONE_SLOT}", 14.2,
        seq_lengths=[4096, 8192, 16384, 24576],
        note="CONTEXT + speed: the 153 tok/s MoE with a 32K cache."),
    Arm("70b-abl-iq4", "B", "Llama-3.3-70B-Instruct-abliterated-IQ4_XS.gguf",
        "huihui-ai/Llama-3.3-70B-Instruct-abliterated", 8192, "on",
        f"{TS_BIAS} {ONE_SLOT}", 37.9,
        note="raw-parameter arm; abliterated to keep the low-refusal property "
             "without a fine-tune"),
    Arm("70b-abl-q4km", "B", "Llama-3.3-70B-Instruct-abliterated-Q4_K_M.gguf",
        "huihui-ai/Llama-3.3-70B-Instruct-abliterated", 8192, "on",
        f"{TS_BIAS} {ONE_SLOT}", 42.5,
        seq_lengths=[2048, 4096],
        note="largest that fits in VRAM at all: 42.5GB of ~45GB usable. If the "
             "server OOMs at this ctx, that IS the finding — record and move on."),

    # ---- Tier C: bigger than VRAM; experts spill into the 64GB of system RAM. ----
    # ~63GB of weights against ~45GB of VRAM, so some expert layers must live in
    # system RAM. KAYA_MOE_N tunes how many: higher N = less VRAM, more CPU work.
    # There is no way to know the right N without trying, so the driver retries
    # with a larger offload when the server fails to come up.
    Arm("gptoss-120b", "C", "gpt-oss-120b-Q4_K_M-00001-of-00002.gguf",
        "unsloth/gpt-oss-120b", 8192, "on",
        f"{TS_BIAS} {ONE_SLOT} --n-cpu-moe {os.environ.get('KAYA_MOE_N', '12')}", 62.7,
        note="MoE ~5B active, ~63GB total; experts spill into the 64GB of RAM."),
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None) -> int:
    """Run a subprocess, teeing combined output to log_path. Returns exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"$ {' '.join(cmd)}  (→ {log_path.name})")
    full_env = {**os.environ, **(env or {})}
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=full_env)
        try:
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            fh.write("\n*** killed: timeout ***\n")
            return 124
    return proc.returncode


def newest(pattern: str, since: float) -> Optional[str]:
    """Newest file under REPORTS matching pattern, modified after `since`."""
    hits = [p for p in glob.glob(str(REPORTS / pattern)) if os.path.getmtime(p) >= since - 1]
    return max(hits, key=os.path.getmtime) if hits else None


def gpu_snapshot() -> List[Dict]:
    """Per-card VRAM/temp/throttle right now."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,temperature.gpu,power.draw,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except Exception:
        return []
    cards = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        cards.append({"gpu": int(parts[0]), "vram_mib": int(parts[1]),
                      "temp_c": int(parts[2]), "power_w": float(parts[3]),
                      "throttle": parts[4]})
    return cards


def compose_up(arm: Arm) -> None:
    env = {
        "KAYA_BENCH_GGUF": arm.gguf,
        "KAYA_BENCH_CTX": str(arm.ctx),
        "KAYA_BENCH_FA": arm.fa,
        "KAYA_BENCH_EXTRA": arm.extra,
        "KAYA_BENCH_SM": arm.sm,
    }
    subprocess.run(["docker", "rm", "-f", BENCH_CONTAINER], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "compose", "--profile", "bench", "up", "-d", "llama-bench"],
                   cwd=str(BASE_DIR), check=True, env={**os.environ, **env},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compose_down() -> None:
    subprocess.run(["docker", "rm", "-f", BENCH_CONTAINER], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)


def wait_healthy(timeout_s: float) -> bool:
    """Poll /health. A 42GB model off NVMe plus CUDA init can take minutes."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", BENCH_CONTAINER],
                          capture_output=True, text=True).stdout.strip() != "true":
            log("  server container exited — see its docker logs")
            return False
        try:
            if requests.get(f"{BENCH_URL}/health", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(10)
    return False


def server_props() -> Dict:
    try:
        return requests.get(f"{BENCH_URL}/props", timeout=10).json()
    except requests.RequestException:
        return {}


def measure_throughput(prompt_tokens: int = 512, n_predict: int = 128) -> Dict:
    """tok/s and time-to-first-token, straight from llama.cpp's own timings."""
    prompt = ("Explica em português de Portugal, com algum detalhe, o que é um grupo de amigos "
              "e porque é importante manter memória partilhada das suas histórias. ") * 6
    try:
        t0 = time.perf_counter()
        resp = requests.post(f"{BENCH_URL}/completion",
                             json={"prompt": prompt, "n_predict": n_predict,
                                   "temperature": 0.0, "cache_prompt": False},
                             timeout=600)
        resp.raise_for_status()
        wall = time.perf_counter() - t0
        tim = resp.json().get("timings", {})
        return {
            "predicted_per_second": round(tim.get("predicted_per_second", 0.0), 2),
            "prompt_per_second": round(tim.get("prompt_per_second", 0.0), 2),
            "prompt_ms": round(tim.get("prompt_ms", 0.0), 1),
            "wall_s": round(wall, 2),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}


# The judge's per-conversation dimensions. Split deliberately: a stock base with
# no LoRA cannot score on voice, but that says nothing about whether it knows more.
GOLDEN_DIMS = ("factual_accuracy", "relevance", "language_quality", "tone",
               "identity_adherence", "factual_grounding", "average", "extended_average")
# What a stock (un-fine-tuned) arm can fairly be judged on.
KNOWLEDGE_DIMS = ("factual_accuracy", "factual_grounding", "relevance")
# What only a fine-tune can deliver — expected to be poor on every stock arm.
VOICE_DIMS = ("tone", "identity_adherence")


def _golden_scores(report: Dict) -> Dict:
    """Per-dimension means over judge-scored conversations.

    Mirrors how BASELINE_SUMMARY.md derived 3.068: averaged over the conversations
    that actually came back scored, since the judge intermittently drops a few.
    Also rolls up a knowledge score and a voice score, so a stock base is not
    dismissed on a composite it structurally cannot win.
    """
    buckets: Dict[str, List[float]] = {d: [] for d in GOLDEN_DIMS}
    for conv in report.get("results", []) or []:
        scores = conv.get("scores")
        if not isinstance(scores, dict):
            continue
        for dim in GOLDEN_DIMS:
            v = scores.get(dim)
            if isinstance(v, (int, float)):
                buckets[dim].append(float(v))

    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else None

    out = {d: mean(v) for d, v in buckets.items()}
    out["scored_n"] = len(buckets["extended_average"])
    knowledge = [x for d in KNOWLEDGE_DIMS for x in buckets[d]]
    voice = [x for d in VOICE_DIMS for x in buckets[d]]
    out["knowledge_score"] = mean(knowledge)
    out["voice_score"] = mean(voice)
    return out


def _recall_pct(report: Dict, max_tokens: int = 3600) -> Dict:
    """Recall over cells at or below max_tokens (the prod envelope), and overall."""
    rows = report.get("rows", [])
    scored = [r for r in rows if not r.get("oom") and "recalled" in r]
    env = [r for r in scored if r.get("actual_tokens", 0) <= max_tokens]

    def pct(rs):
        return round(100.0 * sum(1 for r in rs if r["recalled"]) / len(rs), 1) if rs else None

    deepest = max((r["actual_tokens"] for r in scored if r["recalled"]), default=0)
    return {"envelope_pct": pct(env), "envelope_n": len(env),
            "overall_pct": pct(scored), "overall_n": len(scored),
            "deepest_recall_tokens": deepest,
            "oom_cells": sum(1 for r in rows if r.get("oom"))}


def score_arm(arm: Arm, judge: str, quick: bool) -> Dict:
    """Run the three harnesses against the live llama-bench server."""
    env = {"KAYA_INFERENCE_BACKEND": "gguf", "KAYA_LLAMA_URL": BENCH_URL}
    row: Dict = {}

    # 1. Golden regression (LLM judge). The long one.
    t0 = time.time()
    rc = run([PYTHON, "scripts/run_golden.py", "--model-dir", arm.tokenizer, "--judge", judge],
             LOGS / f"{arm.tag}_golden.log", env=env, timeout=7200)
    path = newest("golden_*.json", t0)
    if rc == 0 and path:
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        total = rep.get("golden_tests_run", 0)
        row["golden"] = {
            **_golden_scores(rep),
            "passed": total - rep.get("golden_tests_failed", 0),
            "run": total,
            "identity_failures": rep.get("identity_failures"),
            "report": os.path.basename(path),
            "minutes": round((time.time() - t0) / 60, 1),
        }
    else:
        row["golden"] = {"error": f"rc={rc}"}
        log(f"  ⚠ golden failed for {arm.tag} (rc={rc})")

    # 2. Needle-in-haystack context recall.
    t0 = time.time()
    cmd = [PYTHON, "scripts/bench_context_recall.py", "--model-dir", arm.tokenizer,
           "--seq-lengths", *[str(s) for s in arm.seq_lengths]]
    if quick:
        cmd += ["--fracs", "0.85", "--depths", "0.0", "0.5", "1.0"]
    rc = run(cmd, LOGS / f"{arm.tag}_recall.log", env=env, timeout=7200)
    path = newest("context_recall_*.json", t0)
    if rc == 0 and path:
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        row["recall"] = {**_recall_pct(rep), "report": os.path.basename(path),
                         "minutes": round((time.time() - t0) / 60, 1)}
    else:
        row["recall"] = {"error": f"rc={rc}"}
        log(f"  ⚠ context recall failed for {arm.tag} (rc={rc})")

    # 3. Offensive / refusal probe.
    t0 = time.time()
    rc = run([PYTHON, "scripts/run_offensive_probe.py", "--model-dir", arm.tokenizer,
              "--tag", arm.tag], LOGS / f"{arm.tag}_offensive.log", env=env, timeout=3600)
    path = newest(f"offensive_{arm.tag}_*.json", t0)
    if rc == 0 and path:
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        row["refusal_rate"] = rep.get("refusal_rate")
        row["offensive_report"] = os.path.basename(path)
    else:
        row["refusal_rate"] = None
        log(f"  ⚠ offensive probe failed for {arm.tag} (rc={rc})")

    return row


def run_arm(arm: Arm, judge: str, quick: bool, load_timeout: float,
            throughput_only: bool = False) -> Dict:
    """Score one arm. throughput_only skips the three harnesses.

    Split mode changes speed but not answers, so re-measuring tok/s after a
    topology change (e.g. single-card vs layer-split) does not need the full
    quality suite re-run — that would cost ~15 min per arm for numbers that
    cannot have moved.
    """
    row: Dict = {"tag": arm.tag, "tier": arm.tier, "gguf": arm.gguf,
                 "tokenizer": arm.tokenizer, "ctx": arm.ctx, "fa": arm.fa,
                 "extra": arm.extra, "sm": arm.sm, "est_gb": arm.est_gb, "note": arm.note,
                 "error": None}

    gguf_path = GGUF_DIR / arm.gguf
    if not gguf_path.exists():
        row["error"] = "gguf missing — run scripts/fetch_bakeoff_models.sh"
        log(f"  ✗ {arm.tag}: {gguf_path.name} not present; skipping")
        return row
    row["gguf_gb"] = round(gguf_path.stat().st_size / 1e9, 1)

    log(f"  starting llama-bench: -c {arm.ctx} -fa {arm.fa} {arm.extra}")
    try:
        compose_up(arm)
    except subprocess.CalledProcessError as exc:
        row["error"] = f"compose up failed: {exc}"
        return row

    try:
        if not wait_healthy(load_timeout):
            row["error"] = f"server not healthy within {load_timeout:.0f}s"
            logs = subprocess.run(["docker", "logs", "--tail", "40", BENCH_CONTAINER],
                                  capture_output=True, text=True).stdout
            (LOGS / f"{arm.tag}_server_fail.log").write_text(logs, encoding="utf-8")
            log(f"  ✗ {arm.tag}: server never became healthy")
            return row

        props = server_props()
        row["server_n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
        row["total_slots"] = props.get("total_slots")
        row["gpu_loaded"] = gpu_snapshot()
        vram = sum(c["vram_mib"] for c in row["gpu_loaded"]) / 1024
        both = sum(1 for c in row["gpu_loaded"] if c["vram_mib"] > 1024)
        row["vram_total_gb"] = round(vram, 1)
        row["cards_used"] = both
        log(f"  loaded: {vram:.1f}GB across {both} card(s), n_ctx={row['server_n_ctx']}")

        row["throughput"] = measure_throughput()
        log(f"  throughput: {row['throughput'].get('predicted_per_second')} tok/s")

        if not throughput_only:
            row.update(score_arm(arm, judge, quick))
        row["gpu_after"] = gpu_snapshot()
    finally:
        compose_down()

    return row


def write_scorecard(path: Path, rows: List[Dict], started: float) -> None:
    path.write_text(json.dumps({
        "baseline": BASELINE,
        "elapsed_min": round((time.time() - started) / 60, 1),
        "generated": datetime.now(timezone.utc).isoformat(),
        "scorecard": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt(value, spec: str = "", dash: str = "—") -> str:
    """Format a value, or a dash when it is missing."""
    if value is None:
        return dash
    return format(value, spec) if spec else str(value)


def print_table(rows: List[Dict]) -> None:
    header = (f"{'TAG':<14}{'TIER':<5}{'GB':>6}{'CARDS':>6}{'TOK/S':>8}"
              f"{'GOLDEN':>8}{'KNOW':>7}{'VOICE':>7}"
              f"{'REC<=3.6k':>10}{'DEEPEST':>9}{'REFUSAL':>8}  NOTE")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in rows:
        golden = r.get("golden") or {}
        recall = r.get("recall") or {}
        thr = r.get("throughput") or {}
        size = r.get("vram_total_gb") or r.get("est_gb") or 0.0
        envelope = recall.get("envelope_pct")
        note = r.get("error") or (r.get("note") or "")[:32]

        print(
            f"{r['tag']:<14}{r.get('tier', ''):<5}"
            f"{size:>6.1f}"
            f"{_fmt(r.get('cards_used')):>6}"
            f"{_fmt(thr.get('predicted_per_second')):>8}"
            f"{_fmt(golden.get('extended_average'), '.3f'):>8}"
            f"{_fmt(golden.get('knowledge_score'), '.2f'):>7}"
            f"{_fmt(golden.get('voice_score'), '.2f'):>7}"
            f"{(f'{envelope:.0f}%' if envelope is not None else '—'):>10}"
            f"{_fmt(recall.get('deepest_recall_tokens')):>9}"
            f"{_fmt(r.get('refusal_rate'), '.0%'):>8}"
            f"  {note}"
        )

    print("=" * len(header))

    ref = next((r for r in rows if r["tag"] == REFERENCE_ARM and not r.get("error")), None)
    if ref:
        rg = (ref.get("golden") or {})
        rr = (ref.get("recall") or {})
        print(f"REFERENCE ({REFERENCE_ARM}, current prod, same judge): "
              f"golden {_fmt(rg.get('extended_average'), '.3f')}  "
              f"know {_fmt(rg.get('knowledge_score'), '.2f')}  "
              f"voice {_fmt(rg.get('voice_score'), '.2f')}  "
              f"recall {_fmt(rr.get('envelope_pct'), '.0f')}%  "
              f"refusal {_fmt(ref.get('refusal_rate'), '.0%')}")
        print("Compare candidates to THAT row, not to the historical numbers below.")
    else:
        print(f"⚠ reference arm '{REFERENCE_ARM}' did not complete — candidates have "
              f"no same-judge baseline to be compared against.")

    b = BASELINE
    print(f"historical ({b['golden_judge']}-judged, BASELINE_SUMMARY.md): "
          f"golden {b['golden_extended_average']}  recall {b['context_recall_pct']}%  "
          f"refusal {b['refusal_rate']:.0%}  — different judge, NOT comparable to GOLDEN above.")
    print("KNOW = factual_accuracy/grounding/relevance — what a stock base can win on.")
    print("VOICE = tone/identity_adherence — only a fine-tune delivers these; stock arms")
    print("        score low here by construction, so do not rank on GOLDEN alone.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Serving bake-off across models and GPU configs.")
    ap.add_argument("--only", default=None, help="Comma-separated arm tags (default: all).")
    ap.add_argument("--tier", default=None, help="Comma-separated tiers to run, e.g. A,B.")
    ap.add_argument("--judge", default="azure",
                    help="LLM judge for the golden suite (default azure; xai is "
                         "out of credits). Must be the SAME judge for every arm.")
    ap.add_argument("--quick", action="store_true",
                    help="Fewer context-recall cells; same three harnesses.")
    ap.add_argument("--load-timeout", type=float, default=900.0,
                    help="Seconds to wait for a server to become healthy (default 900).")
    ap.add_argument("--resume", default=None,
                    help="Existing scorecard JSON; arms already scored are skipped.")
    ap.add_argument("--list", action="store_true", help="Print the matrix and exit.")
    ap.add_argument("--throughput-only", action="store_true",
                    help="Measure load + tok/s only, skipping the three harnesses. "
                         "For re-measuring speed after a topology change.")
    args = ap.parse_args()

    arms = ARMS
    if args.only:
        want = {t.strip() for t in args.only.split(",")}
        arms = [a for a in arms if a.tag in want]
    if args.tier:
        tiers = {t.strip().upper() for t in args.tier.split(",")}
        arms = [a for a in arms if a.tier in tiers]

    if args.list:
        for a in arms:
            present = "✓" if (GGUF_DIR / a.gguf).exists() else "MISSING"
            print(f"  [{a.tier}] {a.tag:<14} {a.est_gb:>5.1f}GB  ctx={a.ctx:<6} "
                  f"fa={a.fa:<4} {present:<8} {a.gguf}")
        return

    done: Dict[str, Dict] = {}
    rows: List[Dict] = []
    if args.resume:
        prev = json.loads(Path(args.resume).read_text(encoding="utf-8"))
        for r in prev.get("scorecard", []):
            if not r.get("error"):
                done[r["tag"]] = r
                rows.append(r)
        log(f"resuming: {len(done)} arm(s) already scored → {sorted(done)}")

    LOGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.resume) if args.resume else REPORTS / f"bakeoff_{stamp}.json"

    started = time.time()
    todo = [a for a in arms if a.tag not in done]
    log(f"bake-off: {len(todo)} arm(s) to run → {[a.tag for a in todo]}")
    log(f"scorecard → {out}")

    for i, arm in enumerate(todo, 1):
        log("=" * 78)
        log(f"ARM {i}/{len(todo)}: {arm.tag}  [tier {arm.tier}]  {arm.gguf}")
        try:
            row = run_arm(arm, args.judge, args.quick, args.load_timeout,
                          throughput_only=args.throughput_only)
        except KeyboardInterrupt:
            log("interrupted — tearing down and saving what we have")
            compose_down()
            break
        except Exception as exc:  # one bad arm must not end a multi-hour sweep
            row = {"tag": arm.tag, "tier": arm.tier,
                   "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            log(f"  ✗ {arm.tag} raised: {exc}")
            compose_down()
        rows.append(row)
        write_scorecard(out, rows, started)   # after every arm, not just at the end
        log(f"  → {arm.tag} done ({round((time.time()-started)/60,1)} min elapsed)")

    write_scorecard(out, rows, started)
    print_table(rows)
    print(f"\nscorecard saved → {out}")


if __name__ == "__main__":
    main()
