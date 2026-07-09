#!/usr/bin/env python3
"""Needle-in-haystack recall sweep for KayaChatBot.

Tests how far the model's *usable* context extends — not just whether it fits in
VRAM, but whether facts planted at various depths are actually recalled.

For each candidate max_seq_length in [2048, 4096, 8192], loads the model at that
window, inserts a synthetic needle at depths {0%, 25%, 50%, 75%, 100%} in growing
filler, asks a direct recall question, and scores deterministically. Also records
prefill latency and peak VRAM. Writes a timestamped JSON + table to
reports/benchmarks/.

    # needs GPU free (stop prod first if running)
    kaya_chatbot_env/bin/python scripts/bench_context_recall.py
    kaya_chatbot_env/bin/python scripts/bench_context_recall.py --seq-lengths 2048 4096
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config_loader import load_config
from src.chat.engine import build_system_prompt
import src.chat.engine as engine_module

_NEEDLE = "NOTA IMPORTANTE: o código secreto do Rafa é 4827."
_QUESTION = "Qual é o código secreto do Rafa?"
_ANSWER_TOKEN = "4827"

_FILLER = (
    "O grupo combinou um jantar no Marginalíssimo e depois um poker em casa do Rafa. "
    "O Peter disse que ia levar o Kobe, o Gil falou da Cuca, e o Bernardo perguntou pelo padel. "
    "Anyway, the plan for the weekend is still up in the air, we'll see who shows up. "
    "Foram ao futebol no sábado e o Frederico marcou o melhor golo da tarde. "
    "A Mimi e o Kobe andaram à luta no jardim enquanto discutiam onde jantar. "
)

SEQ_LENGTHS = [2048, 4096, 8192]
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
TARGET_PROMPT_FRACTIONS = [0.5, 0.85, 0.98]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _free_model() -> None:
    """Release GPU memory after one model load."""
    engine_module._engine_instance = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_model_at_seq(config, seq_len: int):
    """Load model+tokenizer at a specific max_seq_length, bypassing the singleton."""
    import json as _json
    from pathlib import Path as _Path

    _free_model()
    model_dir = config["training"]["output_dir"]
    adapter_cfg = _json.loads((_Path(model_dir) / "adapter_config.json").read_text())
    base_name = adapter_cfg["base_model_name_or_path"]
    is_gemma4 = "gemma-4" in base_name.lower() or "gemma4" in base_name.lower()

    if not is_gemma4:
        raise RuntimeError("bench_context_recall only supports the Gemma-4 path")

    from unsloth import FastModel

    _log(f"  loading model at max_seq_length={seq_len} …")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    FastModel.for_inference(model)
    _log(f"  ✓ loaded (max_seq_length={seq_len})")
    return model, tokenizer


def _tokenized_len(tokenizer, system_prompt: str, user_text: str) -> int:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return int(tokenizer(text=[prompt], return_tensors="pt")["input_ids"].shape[1])


def _build_prompt_with_needle(
    tokenizer,
    system_prompt: str,
    target_tokens: int,
    depth: float,
) -> Tuple[dict, int, int]:
    """Build a prompt of ~target_tokens with the needle at `depth` fraction.

    Returns (inputs_dict, actual_prompt_tokens, needle_position_tokens).
    """
    base_question = f"{_QUESTION}\n\n"
    reps = 1
    filler_block = _FILLER
    for _ in range(25):
        n = _tokenized_len(tokenizer, system_prompt, base_question + filler_block)
        if n >= target_tokens:
            break
        reps = max(reps * 2, 2)
        filler_block = _FILLER * reps

    split_at = int(len(filler_block) * depth)
    user_text = base_question + filler_block[:split_at] + "\n" + _NEEDLE + "\n" + filler_block[split_at:]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[prompt], return_tensors="pt").to("cuda")
    actual_tokens = int(inputs["input_ids"].shape[1])

    needle_position = int(actual_tokens * depth)
    return inputs, actual_tokens, needle_position


def _synchronous_generate(model, tokenizer, inputs: dict, max_new_tokens: int = 64):
    """Synchronous generate; returns (answer_text, prefill_s, peak_gb)."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer, round(elapsed, 3), round(peak_gb, 2)


def sweep_seq_length(
    model,
    tokenizer,
    system_prompt: str,
    seq_len: int,
    target_fractions: List[float],
    depths: List[float],
) -> List[dict]:
    rows = []
    for frac in target_fractions:
        target_tokens = int(seq_len * frac)
        for depth in depths:
            try:
                inputs, actual_tokens, needle_pos = _build_prompt_with_needle(
                    tokenizer, system_prompt, target_tokens, depth
                )
                answer, elapsed_s, peak_gb = _synchronous_generate(model, tokenizer, inputs)
                recalled = _ANSWER_TOKEN in answer
                row = {
                    "seq_len": seq_len,
                    "target_frac": frac,
                    "target_tokens": target_tokens,
                    "actual_tokens": actual_tokens,
                    "depth": depth,
                    "needle_pos_approx": needle_pos,
                    "recalled": recalled,
                    "answer_snippet": answer[:120],
                    "elapsed_s": elapsed_s,
                    "peak_vram_gb": peak_gb,
                }
                _log(
                    f"  seq={seq_len:5d} frac={frac:.2f} depth={depth:.2f} "
                    f"tok={actual_tokens:5d} recall={'✓' if recalled else '✗'} "
                    f"{elapsed_s:.1f}s {peak_gb:.1f}GB"
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    row = {
                        "seq_len": seq_len, "target_frac": frac,
                        "target_tokens": target_tokens, "depth": depth,
                        "oom": True,
                    }
                    _log(f"  seq={seq_len:5d} frac={frac:.2f} depth={depth:.2f}  OOM")
                else:
                    raise
            rows.append(row)
    return rows


def print_summary(all_rows: List[dict]) -> None:
    _log("\n" + "=" * 70)
    _log("RECALL SUMMARY (depth=rows, tokens=cols)")
    _log("=" * 70)

    seq_lens = sorted({r["seq_len"] for r in all_rows if not r.get("oom")})
    fracs = sorted({r["target_frac"] for r in all_rows if not r.get("oom")})
    depths_sorted = sorted({r["depth"] for r in all_rows if not r.get("oom")})

    for seq_len in seq_lens:
        _log(f"\n  max_seq_length = {seq_len}")
        header = f"  {'depth':>6} | " + " | ".join(f"  frac={f:.2f}" for f in fracs)
        _log(header)
        for depth in depths_sorted:
            cells = []
            for frac in fracs:
                matching = [
                    r for r in all_rows
                    if r.get("seq_len") == seq_len
                    and r.get("target_frac") == frac
                    and r.get("depth") == depth
                ]
                if not matching:
                    cells.append("  ---   ")
                elif matching[0].get("oom"):
                    cells.append("  OOM   ")
                else:
                    r = matching[0]
                    cells.append(f"{'✓' if r['recalled'] else '✗'} {r['actual_tokens']:5d}t")
            _log(f"  {depth:>6.2f} | " + " | ".join(cells))

    _log("\nLatency at depth=0.5 (needle in middle):")
    _log(f"  {'seq_len':>8} {'frac':>6} {'tokens':>7} {'elapsed_s':>10} {'VRAM_GB':>8}")
    for r in sorted(all_rows, key=lambda x: (x.get("seq_len", 0), x.get("target_frac", 0))):
        if r.get("depth") == 0.5 and not r.get("oom"):
            _log(
                f"  {r['seq_len']:>8} {r['target_frac']:>6.2f} {r['actual_tokens']:>7} "
                f"{r['elapsed_s']:>10.2f} {r['peak_vram_gb']:>8.1f}"
            )
    _log("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description="Needle-in-haystack recall benchmark")
    ap.add_argument("--seq-lengths", type=int, nargs="+", default=SEQ_LENGTHS)
    ap.add_argument("--depths", type=float, nargs="+", default=DEPTHS)
    ap.add_argument("--fracs", type=float, nargs="+", default=TARGET_PROMPT_FRACTIONS)
    args = ap.parse_args()

    config_path = str(Path(__file__).parent.parent / "config.yaml")
    config = load_config(config_path)
    system_prompt = build_system_prompt(config, config_path, include_uncensored=False)

    all_rows: List[dict] = []

    for seq_len in args.seq_lengths:
        _log(f"\n=== max_seq_length = {seq_len} ===")
        try:
            model, tokenizer = _load_model_at_seq(config, seq_len)
        except Exception as exc:
            _log(f"  FAILED to load at seq_len={seq_len}: {exc}")
            continue

        rows = sweep_seq_length(model, tokenizer, system_prompt, seq_len, args.fracs, args.depths)
        all_rows.extend(rows)

        del model, tokenizer
        _free_model()

    print_summary(all_rows)

    out_dir = Path(config.get("benchmark", {}).get("output_dir", "reports/benchmarks/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"context_recall_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "model_dir": config["training"]["output_dir"],
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "needle": _NEEDLE,
                "question": _QUESTION,
                "answer_token": _ANSWER_TOKEN,
                "seq_lengths_tested": args.seq_lengths,
                "depths": args.depths,
                "target_fractions": args.fracs,
                "rows": all_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
