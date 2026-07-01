#!/usr/bin/env python3
"""Overnight heretic base-model sweep.

For the stock baseline (existing v16 adapter) and each heretic base, (re)train the
LoRA on the already-built train_synthetic.jsonl, then score the result on the golden
benchmark and the offensive/refusal probe. Runs strictly sequentially (one GPU),
each step in its own subprocess so the model is loaded cleanly per evaluation.

Frees the GPU by stopping kaya-prod first. Does NOT restart prod — the winner is
chosen from the scorecard and deployed separately.

    kaya_chatbot_env/bin/python scripts/heretic_sweep.py
    kaya_chatbot_env/bin/python scripts/heretic_sweep.py --only pew,ultra   # subset
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PYTHON = str(BASE_DIR / "kaya_chatbot_env" / "bin" / "python")
REPORTS = BASE_DIR / "reports" / "benchmarks"
LOGS = BASE_DIR / "logs"

# tag, base repo (None = baseline: eval the existing adapter, no retrain), model_dir
CANDIDATES = [
    ("baseline", None, "models/kaya_gemma4_synth_v16"),
    ("pew", "p-e-w/gemma-4-E4B-it-heretic", "models/kaya_heretic_pew"),
    ("ultra", "llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic", "models/kaya_heretic_ultra"),
    ("coder3101", "coder3101/gemma-4-E4B-it-heretic", "models/kaya_heretic_coder3101"),
    ("umbra", "TheUmbraWalker/gemma-4-E4B-it-2-heretic", "models/kaya_heretic_umbra"),
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, log_path: Path) -> int:
    """Run a subprocess, teeing combined output to log_path. Returns the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"$ {' '.join(cmd)}  (→ {log_path.name})")
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd, cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
        proc.wait()
    return proc.returncode


def newest(pattern: str, since: float):
    """Newest file matching pattern (under REPORTS) modified after `since`, or None."""
    hits = [p for p in glob.glob(str(REPORTS / pattern)) if os.path.getmtime(p) >= since - 1]
    return max(hits, key=os.path.getmtime) if hits else None


def free_gpu() -> None:
    log("Stopping kaya-prod to free the GPU …")
    subprocess.run(["docker", "stop", "kaya-prod"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser(description="Heretic base-model sweep.")
    ap.add_argument("--only", default=None, help="Comma-separated tags to run (default: all).")
    ap.add_argument("--skip-train-if-exists", action="store_true", default=True,
                    help="Skip training a candidate whose adapter_config.json already exists.")
    args = ap.parse_args()

    selected = set(args.only.split(",")) if args.only else None
    candidates = [c for c in CANDIDATES if selected is None or c[0] in selected]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    LOGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    free_gpu()

    scorecard = []
    sweep_start = time.time()
    for tag, repo, model_dir in candidates:
        log("=" * 70)
        log(f"CANDIDATE: {tag}  base={repo or '(stock v16)'}  dir={model_dir}")
        abs_dir = BASE_DIR / model_dir
        row = {"tag": tag, "base": repo, "model_dir": model_dir,
               "trained": False, "golden": None, "refusal_rate": None, "error": None}

        # 1. Train (skip baseline; skip if already present)
        if repo is not None:
            if args.skip_train_if_exists and (abs_dir / "adapter_config.json").exists():
                log(f"  adapter already exists at {model_dir} — skipping training.")
                row["trained"] = True
            else:
                t0 = time.time()
                rc = run([PYTHON, "src/finetuning/train.py",
                          "--profile", "gemma4-e4b", "--model-id", repo, "--output-dir", model_dir],
                         LOGS / f"heretic_{tag}_train.log")
                if rc != 0 or not (abs_dir / "adapter_config.json").exists():
                    row["error"] = f"train failed (rc={rc})"
                    log(f"  ✗ training FAILED for {tag} (rc={rc}); skipping eval.")
                    scorecard.append(row)
                    continue
                row["trained"] = True
                log(f"  ✓ trained in {(time.time()-t0)/60:.0f} min")
        else:
            if not (abs_dir / "adapter_config.json").exists():
                row["error"] = "baseline adapter missing"
                scorecard.append(row)
                continue
            row["trained"] = True

        # 2. Golden benchmark
        t0 = time.time()
        rc = run([PYTHON, "scripts/run_golden.py", "--model-dir", model_dir],
                 LOGS / f"heretic_{tag}_golden.log")
        gpath = newest("golden_*.json", t0)
        if rc == 0 and gpath:
            rep = json.loads(Path(gpath).read_text())
            run_n = rep.get("golden_tests_run", 0)
            failed = rep.get("golden_tests_failed", 0)
            row["golden"] = {"passed": run_n - failed, "run": run_n,
                             "pass_rate": round((run_n - failed) / run_n, 4) if run_n else None,
                             "report": os.path.basename(gpath)}
        else:
            log(f"  ⚠ golden eval issue for {tag} (rc={rc})")

        # 3. Offensive / refusal probe
        t0 = time.time()
        rc = run([PYTHON, "scripts/run_offensive_probe.py", "--model-dir", model_dir, "--tag", tag],
                 LOGS / f"heretic_{tag}_offensive.log")
        opath = newest(f"offensive_{tag}_*.json", t0)
        if rc == 0 and opath:
            rep = json.loads(Path(opath).read_text())
            row["refusal_rate"] = rep.get("refusal_rate")
            row["offensive_report"] = os.path.basename(opath)
        else:
            log(f"  ⚠ offensive probe issue for {tag} (rc={rc})")

        scorecard.append(row)
        log(f"  → {tag}: golden={row['golden']}  refusal_rate={row['refusal_rate']}")

    # Aggregate + recommend
    out = REPORTS / f"heretic_sweep_{stamp}.json"
    out.write_text(json.dumps({"scorecard": scorecard,
                               "elapsed_min": round((time.time() - sweep_start) / 60, 1)},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    base_pass = next((r["golden"]["pass_rate"] for r in scorecard
                      if r["tag"] == "baseline" and r.get("golden")), None)

    def eligible(r):
        if r["tag"] == "baseline" or r["refusal_rate"] is None or not r.get("golden"):
            return False
        if base_pass is None:
            return True
        return r["golden"]["pass_rate"] is not None and r["golden"]["pass_rate"] >= base_pass - 0.10

    winners = sorted([r for r in scorecard if eligible(r)],
                     key=lambda r: (r["refusal_rate"], -(r["golden"]["pass_rate"] or 0)))
    recommended = winners[0]["tag"] if winners else None

    print("\n" + "=" * 72)
    print(f"{'TAG':<12}{'TRAINED':<9}{'GOLDEN PASS':<14}{'REFUSAL RATE':<14}{'NOTE'}")
    print("-" * 72)
    for r in scorecard:
        g = r.get("golden")
        gp = f"{g['passed']}/{g['run']} ({g['pass_rate']:.0%})" if g and g.get("pass_rate") is not None else "—"
        rr = f"{r['refusal_rate']:.0%}" if r["refusal_rate"] is not None else "—"
        note = r["error"] or ("◀ recommended" if r["tag"] == recommended else "")
        print(f"{r['tag']:<12}{str(r['trained']):<9}{gp:<14}{rr:<14}{note}")
    print("=" * 72)
    print(f"RECOMMENDED WINNER: {recommended or '(none — review manually)'}")
    print(f"scorecard saved → {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
