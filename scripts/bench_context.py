#!/usr/bin/env python3
"""Context-capacity benchmark for the Kaya serving box.

Answers "how far can we push this machine?" so the RAG/history/output knobs are
tuned to the real ceiling instead of guessed. Loads the live engine (same model
the app serves) and, on the real generation path, sweeps:

  1. input context length (padded prompt) at a fixed output length, and
  2. output length (max_new_tokens) at a fixed context,

recording per point: peak VRAM, prefill latency, decode throughput (tokens/s),
total latency, and the OOM / failure point. Prints a safe operating envelope
(largest context+output under the VRAM budget) and recommended config values.

Synchronous + OOM-safe (a failed generation is caught, not left hanging) and
line-buffered so progress is visible even under nohup.

    # needs the GPU free (stop prod first)
    kaya_chatbot_env/bin/python scripts/bench_context.py
    kaya_chatbot_env/bin/python scripts/bench_context.py --vram-budget-gb 22
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config_loader import load_config
from src.chat.engine import get_engine, build_system_prompt

_FILLER = (
    "O grupo combinou um jantar no Marginalíssimo e depois um poker em casa do Rafa. "
    "Peter disse que ia levar o Kobe, Gil falou da Cuca, e o Bernardo perguntou pelo padel. "
    "Anyway, the plan for the weekend is still up in the air, we'll see who shows up. "
)

CONTEXT_SIZES = [512, 1024, 2048, 3072, 4096, 6144, 8192]
OUTPUT_SIZES = [64, 128, 256, 512, 768]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _tokenized_len(engine, system_prompt: str, user_text: str) -> int:
    tokenizer = engine.tokenizer
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return int(tokenizer(text=[prompt], return_tensors="pt")["input_ids"].shape[1])


def _build_inputs(engine, system_prompt: str, context_tokens: int):
    """Grow a padded user turn until the *actual* prompt hits context_tokens.

    Measures the real chat-template-tokenised length each step (the ground truth,
    robust to the Gemma4Processor's output shape) rather than trusting a single
    pre-tokenisation, so the sweep genuinely varies context length.
    """
    tokenizer = engine.tokenizer
    base = "Resume as novidades do grupo. "
    reps = 1
    user_text = base + _FILLER
    # Exponentially grow until we exceed the target, then it's close enough.
    for _ in range(20):
        n = _tokenized_len(engine, system_prompt, user_text)
        if n >= context_tokens:
            break
        reps = max(reps * 2, 2)
        user_text = base + (_FILLER * reps)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[prompt], return_tensors="pt").to("cuda")
    return inputs, int(inputs["input_ids"].shape[1])


def _timed_generate(engine, inputs, max_new_tokens: int):
    """Synchronous generate; returns (elapsed_s, peak_gb) or raises OOM."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        engine.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=engine._inf.get("temperature", 1.0),
            top_p=engine._inf.get("top_p", 0.95), top_k=engine._inf.get("top_k", 64),
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return elapsed, peak_gb


def _measure(engine, system_prompt, context_tokens, max_new_tokens):
    """Prefill = a 1-token generate; decode rate from the full generate."""
    try:
        inputs, prompt_tokens = _build_inputs(engine, system_prompt, context_tokens)
        prefill_s, _ = _timed_generate(engine, inputs, 1)
        total_s, peak_gb = _timed_generate(engine, inputs, max_new_tokens)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            return {"oom": True}
        raise
    decode_s = max(total_s - prefill_s, 1e-6)
    return {
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "peak_vram_gb": round(peak_gb, 2),
        "prefill_s": round(prefill_s, 3),
        "decode_tok_s": round((max_new_tokens - 1) / decode_s, 1) if max_new_tokens > 1 else None,
        "total_s": round(total_s, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Context-capacity benchmark")
    ap.add_argument("--vram-budget-gb", type=float, default=22.0)
    ap.add_argument("--fixed-output", type=int, default=96)
    ap.add_argument("--fixed-context", type=int, default=2048)
    args = ap.parse_args()

    config_path = str(Path(__file__).parent.parent / "config.yaml")
    config = load_config(config_path)
    _log(f"Loading engine ({config['training']['output_dir']}) …")
    engine = get_engine(config)
    system_prompt = build_system_prompt(config, config_path, include_uncensored=False)

    results = {"context_sweep": [], "output_sweep": []}

    _log(f"\n=== Context-length sweep (max_new_tokens={args.fixed_output}) ===")
    _log(f"{'ctx':>6} {'prompt':>7} {'peak_GB':>8} {'prefill_s':>9} {'dec_t/s':>8} {'total_s':>8}")
    for ctx in CONTEXT_SIZES:
        m = _measure(engine, system_prompt, ctx, args.fixed_output)
        if m.get("oom"):
            _log(f"{ctx:>6}   OOM — capacity ceiling reached")
            results["context_sweep"].append({"target_ctx": ctx, "oom": True})
            break
        results["context_sweep"].append({"target_ctx": ctx, **m})
        _log(f"{ctx:>6} {m['prompt_tokens']:>7} {m['peak_vram_gb']:>8} "
             f"{m['prefill_s']:>9} {str(m['decode_tok_s']):>8} {m['total_s']:>8}")

    _log(f"\n=== Output-length sweep (context≈{args.fixed_context} tok) ===")
    _log(f"{'out':>6} {'peak_GB':>8} {'prefill_s':>9} {'dec_t/s':>8} {'total_s':>8}")
    for out in OUTPUT_SIZES:
        m = _measure(engine, system_prompt, args.fixed_context, out)
        if m.get("oom"):
            _log(f"{out:>6}   OOM")
            results["output_sweep"].append({"max_new_tokens": out, "oom": True})
            break
        results["output_sweep"].append(m)
        _log(f"{out:>6} {m['peak_vram_gb']:>8} {m['prefill_s']:>9} "
             f"{str(m['decode_tok_s']):>8} {m['total_s']:>8}")

    ok = [r for r in results["context_sweep"]
          if not r.get("oom") and r.get("peak_vram_gb", 1e9) <= args.vram_budget_gb]
    safe_ctx = max((r["prompt_tokens"] for r in ok), default=0)
    _log("\n" + "=" * 60)
    _log(f"Safe context ceiling under {args.vram_budget_gb} GB: ~{safe_ctx} prompt tokens")
    if ok:
        typical = ok[len(ok) // 2]
        _log(f"Typical decode throughput: ~{typical.get('decode_tok_s')} tok/s")
    _log("Recommended (leave headroom below the ceiling):")
    _log(f"  rag.max_context_tokens ≈ {max(0, int(safe_ctx * 0.5))}  (RAG ~half the prompt; rest is system+history)")
    _log("=" * 60)

    out_dir = Path(config.get("benchmark", {}).get("output_dir", "reports/benchmarks/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"context_capacity_{stamp}.json"
    out_path.write_text(json.dumps({
        "timestamp": stamp,
        "model_dir": config["training"]["output_dir"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_budget_gb": args.vram_budget_gb,
        "safe_context_tokens": safe_ctx,
        **results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"report saved → {out_path}")


if __name__ == "__main__":
    main()
