# Baseline anchor — before Unsloth-update experiments

Captured 2026-07-20 on the **current prod model** `models/kaya_gemma4_heretic_seq4096_wpp`
(active profile `gemma4-e4b-seq4096-wpp`), stack `unsloth==2026.4.5` / `transformers==5.5.0` / `trl==0.24.0`, RTX 3090.
Prod was stopped for a clean-GPU run, then restarted (site down ~15 min).

These are the numbers every experiment (A: training upgrade, B: GGUF serving) must be compared against.

| Harness | Metric | Baseline | Source JSON |
|---|---|---|---|
| `run_golden.py --judge xai` | mean `extended_average` | **3.068** (floor ≈3.046) | `BASELINE_golden.json` |
| | mean `average` (4-dim) | 3.130 | |
| | tests passed | 7/33 (27/33 judge-scored) | |
| | identity_failures | 1 | |
| `bench_context_recall.py` | recall @≤3600 tok (prod envelope) | **19/20 = 95%** | `BASELINE_context_recall.json` |
| | recall overall (incl. 8192) | 40/45 = 89% | |
| | avg latency / peak VRAM | 8.8 s / 13.2 GB | |
| `run_offensive_probe.py` | refusal_rate | **15%** (3/20) | `BASELINE_offensive.json` |
| `metrics.aggregate()` (live) | avg_latency_ms | **~40,126 ms** end-to-end | `data/feedback/live_interactions.jsonl` |
| | avg_response_words / web_search_rate | 69.7 / 8.1% (n=393, 334 WhatsApp) | |

## Notes / caveats
- 6 of 33 golden conversations came back without judge scores (likely transient xAI judge failures); `extended_average` is the mean over the 27 scored. Re-runs may vary ±.
- Context recall is **95% (not 100%) at ≤3600 tok** even on the current prod model — one miss. Treat 95% as the bar to hold, not 100%.
- Live `avg_latency_ms ≈ 40 s` is the end-to-end WhatsApp figure (includes retrieval, generation, web-search on 8% of turns) — this is the number Experiment B (GGUF inference speedup) must move.
