#!/usr/bin/env python3
"""Fine-tune the bake-off finalists, walking the OOM ladder automatically.

The serving bake-off ranks STOCK bases, which carry no group voice. This trains
the finalists on the existing WhatsApp dataset so the composite golden score
becomes meaningful, then exports each to GGUF ready to be scored as a new arm.

Every run is single-GPU on purpose: Unsloth 2026.4.5's multi-GPU path is DDP (a
full model copy per card), so the second 3090 adds no training capacity — see the
GPU topology section in CLAUDE.md. CUDA_VISIBLE_DEVICES is pinned to one card
here, which also keeps HF Trainer out of DataParallel.

On CUDA OOM it retries down the ladder from CLAUDE.md rather than giving up:
    lora_r 16 -> 8, then max_seq_length 4096 -> 2048.

    kaya_chatbot_env/bin/python scripts/train_bakeoff.py                 # all finalists
    kaya_chatbot_env/bin/python scripts/train_bakeoff.py --only gemma4-12b-wpp
    kaya_chatbot_env/bin/python scripts/train_bakeoff.py --skip-export
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).parent.parent
PYTHON = str(BASE_DIR / "kaya_chatbot_env" / "bin" / "python")
LOGS = BASE_DIR / "logs" / "train_bakeoff"
REPORTS = BASE_DIR / "reports" / "benchmarks"

# profile, quant to export at. Quant is chosen so the merged model still fits the
# serving envelope the winning arm used — a 31B at Q6_K is 25GB and will not fit
# one card, so the dense 31B exports at Q4_K_M.
FINALISTS = [
    ("gemma4-12b-wpp", "Q6_K"),
    ("gemma4-26b-a4b-wpp", "Q4_K_M"),
    ("gemma4-31b-wpp", "Q4_K_M"),
]

# (lora_r, max_seq_length); None = use the profile's value.
OOM_LADDER = [(None, None), (8, None), (8, 2048)]

OOM_MARKERS = ("out of memory", "CUDA out of memory", "OutOfMemoryError",
               "torch.OutOfMemoryError")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"$ {' '.join(cmd)}  (→ {log_path.name})")
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                env={**os.environ, **(env or {})})
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
        proc.wait()
    return proc.returncode


def looks_like_oom(log_path: Path) -> bool:
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
    except OSError:
        return False
    return any(m.lower() in tail.lower() for m in OOM_MARKERS)


def train_one(profile: str, gpu: str) -> Dict:
    """Train one profile, descending the OOM ladder until it fits."""
    row = {"profile": profile, "trained": False, "attempts": [], "error": None}

    for lora_r, seq in OOM_LADDER:
        label = f"r{lora_r or 'cfg'}_seq{seq or 'cfg'}"
        cmd = [PYTHON, "src/finetuning/train.py", "--profile", profile]
        if lora_r:
            cmd += ["--lora-r", str(lora_r)]
        if seq:
            cmd += ["--max-seq-length", str(seq)]

        t0 = time.time()
        log_path = LOGS / f"{profile}_{label}.log"
        # Pin to one card: Unsloth only installs its DistributedType.NO patch at
        # device_count()==1, and its multi-GPU path is DDP anyway.
        rc = run(cmd, log_path, env={"CUDA_VISIBLE_DEVICES": gpu})
        mins = round((time.time() - t0) / 60, 1)
        attempt = {"lora_r": lora_r, "max_seq_length": seq, "rc": rc, "minutes": mins,
                   "log": log_path.name}
        row["attempts"].append(attempt)

        if rc == 0:
            row.update(trained=True, lora_r=lora_r, max_seq_length=seq, minutes=mins)
            log(f"  ✓ {profile} trained in {mins} min ({label})")
            return row

        if looks_like_oom(log_path):
            log(f"  ⚠ {profile} OOM at {label} after {mins} min — stepping down the ladder")
            continue

        row["error"] = f"train failed (rc={rc}, not OOM) — see {log_path.name}"
        log(f"  ✗ {profile} failed for a non-OOM reason; not retrying")
        return row

    row["error"] = "OOM at every rung of the ladder"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the bake-off finalists.")
    ap.add_argument("--only", default=None, help="Comma-separated profile names.")
    ap.add_argument("--gpu", default="0", help="Which GPU index to train on (default 0).")
    ap.add_argument("--skip-export", action="store_true",
                    help="Train only; do not merge + quantize to GGUF.")
    args = ap.parse_args()

    finalists = FINALISTS
    if args.only:
        want = {p.strip() for p in args.only.split(",")}
        finalists = [f for f in finalists if f[0] in want]

    LOGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORTS / f"train_bakeoff_{stamp}.json"

    rows: List[Dict] = []
    started = time.time()
    for i, (profile, quant) in enumerate(finalists, 1):
        log("=" * 78)
        log(f"FINALIST {i}/{len(finalists)}: {profile}  (export {quant})")
        row = train_one(profile, args.gpu)
        row["export_quant"] = quant

        if row["trained"] and not args.skip_export:
            t0 = time.time()
            rc = run([PYTHON, "scripts/export_gguf.py", "--profile", profile,
                      "--quant", quant], LOGS / f"{profile}_export.log")
            row["exported"] = rc == 0
            row["export_minutes"] = round((time.time() - t0) / 60, 1)
            if rc != 0:
                log(f"  ⚠ export failed for {profile} (rc={rc})")

        rows.append(row)
        out.write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(),
                                   "elapsed_min": round((time.time() - started) / 60, 1),
                                   "results": rows}, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print(f"{'PROFILE':<24}{'TRAINED':<9}{'r':>4}{'SEQ':>7}{'MIN':>7}{'EXPORTED':>10}  NOTE")
    print("-" * 84)
    for r in rows:
        print(f"{r['profile']:<24}{str(r['trained']):<9}"
              f"{str(r.get('lora_r') or 'cfg'):>4}{str(r.get('max_seq_length') or 'cfg'):>7}"
              f"{str(r.get('minutes') or '—'):>7}{str(r.get('exported', '—')):>10}  "
              f"{r.get('error') or ''}")
    print("=" * 84)
    print(f"report → {out}")


if __name__ == "__main__":
    main()
