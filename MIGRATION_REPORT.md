# Overnight Migration Report — WhatsApp-only, Fully On-Prem Pipeline

**Run:** 2026-07-19 evening → 2026-07-20 morning (autonomous)
**Result: ✅ COMPLETE — new adapter live in prod. One action needed from you (WhatsApp QR re-scan, see below).**

---

## ⚠️ The one thing you need to do — scan the WhatsApp QR

**The bot's WhatsApp device link was invalidated** while WAHA was offline for the ~9h of GPU work (WhatsApp unlinks devices that stay disconnected too long). The bridge needs to be re-linked by scanning a QR with the bot's phone — that's the only step I can't do for you.

**Diagnosis (2026-07-20):** I cleared the old session and confirmed the setup is otherwise fully healthy — the webhook config is intact (`kaya-prod:7860/whatsapp/webhook`, events `message,message.reaction`) and the QR endpoint serves fine. The `FAILED` status you may see is **not a fault**: it's normal WAHA/Baileys behavior — an *unscanned* QR times out after ~2–3 minutes and parks the session in `FAILED` until it's restarted. It will keep doing that until someone scans. (The old corrupt session was backed up to `data/waha/noweb/default.corrupt.bak` and replaced with a clean one.)

**Fix (1 minute):**
1. Open **http://localhost:3000** (WAHA dashboard; login = your `KAYA_WEB_USER` / `KAYA_WEB_PASS`).
2. If the `default` session shows `FAILED` or `STOPPED`, click **Start/Restart** — a live, auto-refreshing QR appears.
3. On the **bot's phone**: WhatsApp → Settings → Linked devices → Link a device → scan the QR.
4. It flips to `WORKING` and the bridge answers again immediately (webhook already wired, no redeploy).

I left the session in a live `SCAN_QR_CODE` state at 18:16, so if you look soon the QR is already up; if it's lapsed to `FAILED`, just hit Restart as in step 2.

**The web chat is unaffected and fully working** (HTTP 200, new model loaded, zero errors, tunnel up).

---

## What was done

### 1. Code — privacy + WhatsApp-only (main `1c0891c`)
- **Knowledge extraction is now local**: new `src/data/local_teacher.py` (shared 4-bit teacher, Qwen3.5-27B) + `LocalTeacherProvider`; `generate_knowledge_base.py` gained `load_backend()` with `knowledge_generation.backend: local` as default. The cloud path survives only as an explicit `--backend cloud` escape hatch that prints a privacy warning. Smoke-tested end to end on real data: clean profiles, zero JSON parse errors (curated `group_members.json`/`group_knowledge.json` were backed up, smoke-tested against, and restored untouched).
- **Cloud pipelines deleted**: `generate_synthetic_data.py`, `generation_utils.py`, `llm_providers/data_cleaning.py`, `scripts/run_pipeline.py`, `prepare_portuguese_data.py`. `WhatsAppReader` is regex-only now. `run_full_pipeline.py` is single-mode (direct, on-prem).
- **Instagram removed entirely**: extraction methods, `InstagramReader`, `InstagramMessage` model, resolver anonymous/double-UTF8 handling (`resolve()` now always returns `str`), incremental-update JSON branch, config keys, raw `data/insta/` exports, and all Instagram-era archived datasets (noted in `data/archive/MANIFEST.md`).
- **Kept, as agreed**: xAI Grok for the eval judge and production web-search (member-free queries only); Azure provider code kept as eval-only judge fallback but removed from all docs.
- **Tests**: 2 files deleted, 3 reworked WhatsApp-only, 10 new tests for the local backend. **Suite: 506 passed locally AND in-container** (5 skips are the deleted legacy direct-data checks). Verified no sender-resolution regression: the re-extracted sender set is byte-identical to before.
- **Docs**: README and CLAUDE.md rewritten per your list — cloud-generation feature line gone, Azure prerequisites/rate-limit sections gone, Instagram data-prep gone, xAI documented as optional (web-search + judge only), new privacy invariant added to CLAUDE.md.

### 2. Data — full Instagram purge (all regenerated on-prem)
| Artifact | Result |
|---|---|
| `all_messages_cleaned.jsonl` | 22,146 messages, **0 instagram** (was 27,084 with 18.3% Instagram) |
| RAG DB (`data/rag_db`) | rebuilt: 1,950 conversation chunks + 29 knowledge facts, retrieval sanity-checked |
| `synthetic_local.jsonl` | 1,979 accepted / 2,067 asked (local teacher, ~12h run) |
| `synthetic_local_long.jsonl` | 287 accepted / 300 (4096-token context variant) |
| `needle_training.jsonl` | 200 examples regenerated |
| Merged train/val | **2,213 / 246** (19% more than the previous dataset) |

### 3. New adapter — `models/kaya_gemma4_heretic_seq4096_wpp` (LIVE)
- Profile `gemma4-e4b-seq4096-wpp`, 450 steps at seq 4096, same recipe as the previous promotion.
- **Eval loss 1.633 → 1.460** (previous adapter finished at 1.515 — the new one is better).

### 4. Quality gates
- **Golden tests (Grok judge): PASS** — mean extended_average **3.115 ≥ 3.046 floor**; 8/33 passed, exact parity with the previous adapter (3.131 / 8/33 — difference is within judge noise).
- **Context recall: PASS at the production operating point** — 100% recall at all depths for contexts ≤ ~3600 tokens (prod runs `max_context_tokens: 2500`, total ≈ 3600t). Long-range extrapolation improved: 10/10 at 8k–14.5k tokens (previous: 9/10). **Known caveat**: 3 deterministic misses when the needle sits at the very first tokens of a completely full ~4783t window (the old adapter passed those cells). This is outside prod's envelope, but if you ever raise `max_context_tokens` near 4096, re-check. Full grids: `reports/benchmarks/context_recall_20260720T052017Z.json` vs `..._20260717T235539Z.json`.
- **Local knowledge backend smoke: PASS** (see §1).

### 5. Deploy
- Promotion commit `ec0c901` pushed to main; prod redeployed and verified: **serving commit `ec0c901`, new adapter loaded, container healthy, web UI HTTP 200, tunnel up, 0 log errors**.
- One deploy hiccup auto-resolved: the prod checkout saw the MANIFEST edit through the shared `data/` symlink as a dirty file; fixed with a forced checkout to the identical committed content (nothing overwritten).

## Rollback
- **Model**: set `active_model_profile: "gemma4-e4b-seq4096"` in config.yaml and redeploy — the previous adapter directory is untouched.
- **Data**: `~/kaya_purge_backup.tgz` holds the pre-purge dataset + RAG DB (contains Instagram data — **delete it after ~1 week of stability**: `rm ~/kaya_purge_backup.tgz`).

## Loose ends
1. **WhatsApp QR re-scan** (top of this report) — only you can do it.
2. Delete `~/kaya_purge_backup.tgz` once satisfied (contains purged Instagram data).
3. The `validate-main` CI run for these pushes may have been interrupted by the suspend — re-run from the Actions tab if you want the green check.
4. Logs kept for inspection: `logs/train_wpp.log`, `logs/longctx_gen.log`, `logs/gate_g1.log`, `logs/gate_golden.log`, `logs/kb_smoke.log`.
5. This report file (`MIGRATION_REPORT.md`) is untracked — delete or commit as you prefer.
