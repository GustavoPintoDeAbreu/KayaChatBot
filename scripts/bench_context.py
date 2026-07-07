#!/usr/bin/env python3
"""Context-capacity benchmark for the Kaya serving box.

Answers "how far can we push this machine?" so the RAG/history/output knobs are
tuned to the real ceiling instead of guessed. Loads the live engine (same model
the app serves) and, on the real generation path, sweeps:

  1. input context length (padded prompt) at a fixed output length, and
  2. output length (max_new_tokens) at a fixed context,

recording per point: peak VRAM, prefill latency (time-to-first-token), decode
throughput (tokens/s), total latency, and the OOM / failure point. It then
prints a safe operating envelope (largest context+output under the VRAM budget)
and recommended config values.

    # needs the GPU free (stop prod first)
    kaya_chatbot_env/bin/python scripts/bench_context.py
    kaya_chatbot_env/bin/python scripts/bench_context.py --vram-budget-gb 22
"""

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import TextIteratorStreamer

from src.config_loader import load_config
from src.chat.engine import get_engine, build_system_prompt

# A block of realistic code-switched PT/EN filler to pad prompts to a target
# token count (so the measurement reflects the real tokenizer, not synthetic ids).
_FILLER = (
    "O grupo combinou um jantar no Marginalíssimo e depois um poker em casa do Rafa. "
    "Peter disse que ia levar o Kobe, Gil falou da Cuca, e o Bernardo perguntou pelo padel. "
    "Anyway, the plan for the weekend is still up in the air, we'll see who shows up. "
)

CONTEXT_SIZES = [512, 1024, 2048, 3072, 4096, 6144, 8192]
OUTPUT_SIZES = [64, 128, 256, 512, 768]


def _pad_to_tokens(tokenizer, base_text: str, target_tokens: int) -> str:
    """Grow ``base_text`` with filler until it tokenizes to ~target_tokens."""
    text = base_text
    while len(tokenizer(text=text)["input_ids"]) < target_tokens:
        text += _FILLER
    ids = tokenizer(text=text)["input_ids"][:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _run_once(engine, system_prompt: str, context_tokens: int, max_new_tokens: int):
    """One generation; return metrics dict or {'oom': True}."""
    tokenizer, model = engine.tokenizer, engine.model
    user_turn = _pad_to_tokens(tokenizer, "Resume as novidades do grupo. " + _FILLER, context_tokens)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_turn},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[prompt], return_tensors="pt").to("cuda")
    prompt_tokens = int(inputs["input_ids"].shape[1])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=engine._inf.get("temperature", 1.0),
        top_p=engine._inf.get("top_p", 0.95), top_k=engine._inf.get("top_k", 64),
        use_cache=True, streamer=streamer,
    )

    try:
        t0 = time.perf_counter()
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        t_first = None
        n_tokens = 0
        for _ in streamer:
            if t_first is None:
                t_first = time.perf_counter()
            n_tokens += 1
        thread.join()
        t_end = time.perf_counter()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"oom": True}

    prefill_s = (t_first - t0) if t_first else (t_end - t0)
    decode_s = (t_end - t_first) if (t_first and n_tokens > 1) else 0.0
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return {
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "gen_tokens": n_tokens,
        "peak_vram_gb": round(peak_gb, 2),
        "prefill_s": round(prefill_s, 3),
        "decode_tok_s": round((n_tokens - 1) / decode_s, 1) if decode_s > 0 else None,
        "total_s": round(t_end - t0, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Context-capacity benchmark")
    ap.add_argument("--vram-budget-gb", type=float, default=22.0,
                    help="VRAM ceiling for the safe envelope (24 GB card − headroom).")
    ap.add_argument("--fixed-output", type=int, default=128,
                    help="max_new_tokens used during the context-length sweep.")
    ap.add_argument("--fixed-context", type=int, default=2048,
                    help="Prompt tokens used during the output-length sweep.")
    args = ap.parse_args()

    config_path = str(Path(__file__).parent.parent / "config.yaml")
    config = load_config(config_path)
    print(f"Loading engine ({config['training']['output_dir']}) …")
    engine = get_engine(config)
    system_prompt = build_system_prompt(config, config_path, include_uncensored=False)

    results = {"context_sweep": [], "output_sweep": []}

    print(f"\n=== Context-length sweep (max_new_tokens={args.fixed_output}) ===")
    print(f"{'ctx_tok':>8} {'prompt_tok':>10} {'peak_GB':>8} {'prefill_s':>9} {'dec_tok/s':>9} {'total_s':>8}")
    for ctx in CONTEXT_SIZES:
        m = _run_once(engine, system_prompt, ctx, args.fixed_output)
        if m.get("oom"):
            print(f"{ctx:>8}  OOM — capacity ceiling reached")
            results["context_sweep"].append({"target_ctx": ctx, "oom": True})
            break
        results["context_sweep"].append({"target_ctx": ctx, **m})
        print(f"{ctx:>8} {m['prompt_tokens']:>10} {m['peak_vram_gb']:>8} "
              f"{m['prefill_s']:>9} {str(m['decode_tok_s']):>9} {m['total_s']:>8}")

    print(f"\n=== Output-length sweep (context≈{args.fixed_context} tok) ===")
    print(f"{'out_tok':>8} {'peak_GB':>8} {'prefill_s':>9} {'dec_tok/s':>9} {'total_s':>8}")
    for out in OUTPUT_SIZES:
        m = _run_once(engine, system_prompt, args.fixed_context, out)
        if m.get("oom"):
            print(f"{out:>8}  OOM")
            results["output_sweep"].append({"max_new_tokens": out, "oom": True})
            break
        results["output_sweep"].append(m)
        print(f"{out:>8} {m['peak_vram_gb']:>8} {m['prefill_s']:>9} "
              f"{str(m['decode_tok_s']):>9} {m['total_s']:>8}")

    # Safe envelope: largest context whose peak VRAM stays under the budget.
    ok = [r for r in results["context_sweep"]
          if not r.get("oom") and r.get("peak_vram_gb", 1e9) <= args.vram_budget_gb]
    safe_ctx = max((r["prompt_tokens"] for r in ok), default=0)
    print("\n" + "=" * 60)
    print(f"Safe context ceiling under {args.vram_budget_gb} GB: ~{safe_ctx} prompt tokens")
    if ok:
        typical = ok[len(ok) // 2]
        print(f"Typical decode throughput: ~{typical.get('decode_tok_s')} tok/s")
    print("Recommended (leave headroom below the ceiling):")
    print(f"  rag.max_context_tokens  ≈ {max(0, int(safe_ctx * 0.5))}  (RAG is ~half the prompt; rest is system+history)")
    print(f"  inference.max_new_tokens keep ≤ the output sweep's stable range")
    print("=" * 60)

    out_dir = Path(config.get("benchmark", {}).get("output_dir", "reports/benchmarks/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"context_capacity_{stamp}.json"
    payload = {
        "timestamp": stamp,
        "model_dir": config["training"]["output_dir"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_budget_gb": args.vram_budget_gb,
        "safe_context_tokens": safe_ctx,
        **results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report saved → {out_path}")


if __name__ == "__main__":
    main()
