# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

KayaChatBot is a private AI assistant for the "Kaya" Portuguese friend group. It maintains long-term memory of group facts and events derived from WhatsApp history and answers in **European Portuguese or English**. It is **not** a group member — it is a bot with access to the group's collective memory.

**Core invariant (revised 2026-08-09)**: RAG is always-on **for factual intent**.
It used to be unconditional, and that was the bug: every message retrieved group
context, so `Ahahhha` got a 77-word analysis of group dynamics and `hey` got a
roast aimed at a randomly chosen member. Handed a pile of member profiles and told
to elaborate, the model finds someone to talk about.

`src/chat/router.py` now classifies each message first, and the mode selects
retrieval, prompt and reply length together. `banter` retrieves **nothing** — that
is deliberate, not a missing call. Factual answers are unchanged (verified: golden
excluding greetings moved −0.053, inside the ±0.07 noise band). Any router failure
falls back to `factual`, i.e. the old behaviour.

**The modes are `banter`, `mixed`, `general`, `factual`** (`router.MODES`).
`mixed` is chat that names a person or an event without asking to be informed
("o Rafa outra vez a fazer disso") — it retrieves, but answers short; it is the
reason a reminiscence does not come back as a report.

**`general` was added 2026-08-12.** `factual` used to swallow every question
that was not banter, including the ones with nothing to do with the group, so
"quem é melhor, Ronaldo ou Messi?" retrieved group context plus every member
profile and came back as a sourced report about Kaya. `general` answers the world
with **no group retrieval and no member profiles**. `retrieval_enabled` is
`mode not in (BANTER, GENERAL)`.

**Live model (since 2026-08-08): a STOCK `gemma-4-12b-it` at Q6_K, with no LoRA.**
A 14-config bake-off found the stock 12B beat the fine-tuned E4B on every judged
dimension (xai-judged golden 3.846 vs 3.068, knowledge 2.94 vs 1.84, refusals 0%
vs 15%), and a control run — same base, same quant, with and without the WhatsApp
LoRA — showed the fine-tune contributes nothing (−0.045, inside the ±0.07 noise
band). RAG supplies the group facts; a capable base supplies the voice. Treat "we
must fine-tune" as a claim needing evidence, not a given. Full results:
`reports/benchmarks/bakeoff_20260808T013135Z.json`.

**Privacy invariant**: No group data leaves the box. Knowledge extraction and synthetic data generation run on the LOCAL teacher model (`src/data/local_teacher.py`). Cloud LLMs (Azure/xAI) are for the eval-time LLM judge and the production web-search only (web-search sends member-free user queries, never chat history or profiles).

---

## Environment

Always use the virtualenv at `kaya_chatbot_env/`. Use the Python executable directly:

```bash
source kaya_chatbot_env/bin/activate
# or invoke directly:
kaya_chatbot_env/bin/python <script>
```

Install dependencies inside the venv: `pip install -r requirements.txt`

---

## Common Commands

```bash
# Full pipeline (extract → format → merge → train)
kaya_chatbot_env/bin/python run_full_pipeline.py

# Individual pipeline steps
kaya_chatbot_env/bin/python src/data/extract_all_messages.py
kaya_chatbot_env/bin/python src/data/generate_knowledge_base.py  # --test / --resume-from N / --backend local|cloud — local teacher needs the GPU (stop prod first)
kaya_chatbot_env/bin/python src/data/build_vector_db.py
kaya_chatbot_env/bin/python src/data/format_direct_training.py
kaya_chatbot_env/bin/python src/data/merge_datasets.py
kaya_chatbot_env/bin/python src/finetuning/train.py

# Chat
kaya_chatbot_env/bin/python src/chat/chat.py

# Inference smoke test
kaya_chatbot_env/bin/python src/chat/inference.py
kaya_chatbot_env/bin/python tests/test_inference.py

# Tests
kaya_chatbot_env/bin/python -m pytest tests/ -v
kaya_chatbot_env/bin/python -m pytest tests/rag/ -v
kaya_chatbot_env/bin/python -m pytest tests/pipeline/ -v
kaya_chatbot_env/bin/python scripts/validate_pipeline.py

# Multi-person conversation simulator (real webhook path, mock outbound)
kaya_chatbot_env/bin/python scripts/seed_sim_data.py   # once, builds ./data_sim
docker compose --profile sim up -d kaya-sim
kaya_chatbot_env/bin/python scripts/run_conversation_sim.py --preset smoke      # ~40s, no images
kaya_chatbot_env/bin/python scripts/run_conversation_sim.py --preset standard   # ~10min
kaya_chatbot_env/bin/python scripts/run_conversation_sim.py --preset long_haul  # ~28min
kaya_chatbot_env/bin/python scripts/run_conversation_sim.py --only images,audio

# Model bake-off (candidate models across GPU configurations)
scripts/fetch_bakeoff_models.sh                          # download candidate GGUFs (~262GB)
kaya_chatbot_env/bin/python scripts/run_conversation_probe.py   # routing/brevity/restraint/in-voice/no-dash/compliance
kaya_chatbot_env/bin/python scripts/run_offensive_probe.py      # refusal rate; the group wants 0%
kaya_chatbot_env/bin/python scripts/model_bakeoff.py --list
kaya_chatbot_env/bin/python scripts/model_bakeoff.py --judge azure   # xai is out of credits
kaya_chatbot_env/bin/python scripts/model_bakeoff.py --resume reports/benchmarks/bakeoff_<stamp>.json
kaya_chatbot_env/bin/python scripts/export_gguf.py --profile gemma4-31b-wpp --quant Q4_K_M

# Docker (always rebuild+prune after changes)
docker-compose up --build
docker system prune  # prevent storage overload

# Dev/Test (Docker)
docker compose --profile dev up -d kaya-dev       # dev web UI on :7861, ./src mounted read-write (or use scripts/app_up.sh dev)
docker compose --profile test run --rm kaya-test  # run the pytest suite in-container

# Deployment (see DEPLOYMENT.md)
scripts/deploy_prod.sh [ref]    # make a commit LIVE: updates ~/kaya-prod + restarts prod (CI's Deploy (prod) calls this)
scripts/app_up.sh dev|prod      # manually power up an env + Cloudflare Tunnel (one env at a time — a model may claim both GPUs)
scripts/app_down.sh dev|prod    # stop and free the GPU
scripts/app_status.sh           # running containers + GPU usage
```

---

## Architecture

### Data Flow

```
Raw chat data (data/wpp/)
    → extract_all_messages.py
    → data/all_messages_cleaned.jsonl + data/finetune_chunks.jsonl
    → [optional] generate_knowledge_base.py (local teacher) → data/group_members.json, data/group_knowledge.json
    → build_vector_db.py → data/rag_db/ (ChromaDB: kaya_conversations + kaya_knowledge_base)
    → format_direct_training.py and/or generate_local_synthetic.py (local teacher) → data/synthetic_local.jsonl
    → merge_datasets.py → data/train_synthetic.jsonl, data/val_synthetic.jsonl
    → train.py → models/kaya_<version>/  (LoRA adapter)
    → scripts/export_gguf.py → models/gguf/<name>.gguf  (merge + quantize)
    → chat.py / web_app.py (tokenizer + RAG at runtime; weights in llama.cpp)

NOTE: the live path no longer uses this pipeline. Prod serves a stock GGUF with
no adapter — the training branch above is only exercised when evaluating whether
a fine-tune helps (it currently does not).
```

### RAG System (`src/chat/retriever.py`)

Two knowledge sources are injected at inference time, controlled by `rag.knowledge_approach` in `config.yaml`:

| Approach | What's injected |
|---|---|
| `json_only` | `group_members.json` profiles → system prompt (best benchmark score) |
| `chromadb_only` | Semantic search over `kaya_knowledge_base` ChromaDB collection |
| `both` | Both of the above |
| `none` | Baseline — conversation history only |

`ConversationRetriever` uses BAAI/bge-m3 embeddings against the `kaya_conversations` ChromaDB collection. `extract_query_persons()` detects named group members in the query and post-filters retrieval by `participants`/`mentioned` metadata. `retrieve_all()` enforces `rag.max_context_tokens` (**14000** since 2026-08-08, up from 2500) by truncating lowest-priority context (conversation chunks first, then knowledge, then recent summaries). Token estimation is whitespace-based (`words / 0.60`, tuned for Portuguese subword inflation).

**Date-aware facts (mixed rule).** Knowledge facts carry optional date metadata: `event_date_hint` (an explicit temporal phrase pulled from the source text), `source_date_start`/`source_date_end` (the timestamp range of the source messages), and `last_updated`. These are populated by `generate_knowledge_base.py` and embedded into ChromaDB metadata by `build_vector_db.py`. The retriever only surfaces dates when `_has_temporal_intent(query)` matches a timing question (PT/EN keywords); otherwise normal answers stay date-free. When surfacing, an explicit `event_date_hint` wins over the message timestamps (relative age rendered by `_relative_age`). `chat.py`/`web_app.py` also append `Hoje é <date>.` to the runtime system prompt so the model can reason about recency.

**Follow-up suggestions (web UI only).** After each answer, `src/chat/suggestions.py` prompts the already-loaded local model a second time for 2-3 follow-up questions, shown as clickable chips in the Gradio UI (`web_app.py`). Controlled by `chat.suggestions` in `config.yaml`; degrades to no chips on any failure.

### GPU topology (2× RTX 3090, no NVLink)

The box has **two 24 GB RTX 3090s and no NVLink bridge**. They are two separate
devices, not a 48 GB pool: `can_device_access_peer(0,1)` is False and `nvidia-smi
topo -p2p` reports `CNS`, so there is **no GPU-to-GPU P2P** and all inter-GPU
traffic stages through system RAM.

| | Serving (llama.cpp) | Python (training, hf backend, CI) |
|---|---|---|
| `NVIDIA_VISIBLE_DEVICES` | `all` | `all` |
| `CUDA_VISIBLE_DEVICES` | `0,1` | **`0`** |
| Ceiling | ~45 GB weights+KV, layer-split | 24 GB |

- **Serving can exceed 24 GB** by layer-splitting one model across both cards
  (`-sm layer`). Only the hidden state crosses PCIe at the layer boundary.
  **Never use `-sm row`** here — without P2P it round-trips through host RAM every
  step. There must be **no `deploy.resources.reservations.devices` block** on the
  llama services: a `count:` reservation overrides `NVIDIA_VISIBLE_DEVICES` and
  silently caps serving at one card.
- **`CUDA_VISIBLE_DEVICES=0` on the Python services is load-bearing.** Unsloth
  only installs its `DistributedType.NO` patch when `device_count() == 1`
  (`unsloth/models/_utils.py`); with both cards visible, HF Trainer falls into
  DataParallel — slower and flaky with 4-bit models. Unsloth's own multi-GPU path
  is **DDP** (a full model copy per card, `models/loader_utils.py
  prepare_device_map`), so exposing both cards gives training **no extra
  capacity** anyway. Training above 24 GB would mean leaving Unsloth for HF+peft
  `device_map="auto"`.
- GPU0 drives the desktop and is capped at 300 W. Burn-in once measured its
  sustained clocks ~15% below GPU1's (1238 vs 1448 MHz), which is why two-card
  serving uses `-ts 0.45,0.55` to give it fewer layers. That gap was measured at
  the old 250 W cap; at 300 W it sustains ~1620 MHz. Worth re-measuring before
  trusting the 0.45/0.55 bias, but note the gap is partly real — see below.
- **GPU0 is cooling-limited, not power-limited. The "intake-starved" note in
  `gpu-power-limit.sh` is correct — 300 W is its ceiling.** Measured 2026-08-16 with
  240 s sustained fp16 burns (harsher than a real render), all from a 61 °C start:

  | GPU0 cap | Sustained | Temp | Fan | Throughput | Thermal slowdown |
  |---|---|---|---|---|---|
  | 300 W, alone | 299 W | 82–83 °C | 96% | 59.4 TFLOPS | none |
  | 350 W, alone | 349 W | 85–86 °C | 93–99% | 63.5 TFLOPS | none |
  | **350 W + GPU1 serving** | **falls 349 → 316 W** | **88 °C** | **100%** | **decays to 59.3** | **+112 s** |

  The concurrent row is the real operating case, and it is why 350 W was tried and
  reverted. With both cards working, GPU0 saturates: fans max out, it can no longer
  hold its own cap, and throughput decays back to exactly the 300 W figure while
  running 5 °C hotter and accruing real throttle time. **The extra 50 W buys nothing
  and costs thermal margin.** Do not raise this card without fixing case airflow
  first.
- **`SW Thermal Slowdown` IS the meaningful counter — trust it.** It stays at zero
  through 82–86 °C and only accrues once the card is genuinely saturated (88 °C,
  fans pinned at 100%), which is exactly the state worth catching. It is frozen at
  idle, so a jump between two idle readings means real distress happened in between.
  Do **not** confuse it with `SW Power Capping`, which accrues continuously whenever
  a cap is merely *set* and means nothing at all.
- **PSU headroom is not the constraint.** Peak combined draw measured 672 W (GPU0
  349 W + GPU1 350 W, both working) — comfortable on the HX1200i. Cooling binds long
  before power does on this box.
- Power caps are enforced by `gpu-power-limit.service` **by UUID** (indices are
  not stable across reboots). Raised from 250/280 W to 300/350 W on 2026-08-16,
  once sustained fine-tuning stopped being the workload. GPU1 at 350 W measured
  **+11% generation throughput** (56.3 → 62.6 tok/s on 250-token runs, 1300 →
  1620 MHz) at 70 °C with fans at 65%, and serving was unaffected by a concurrent
  GPU0 render (63.0 tok/s mean across 95 replies). GPU0 was tested at 350 W and
  **reverted to 300 W** for the thermal reason above. `SwPowerCap` in the throttle
  bitmask is expected and healthy.
- **Manual fan control is impossible on this host, and not needed.** The driver is
  `nvidia-driver-595-open` — the *open* kernel module (`/proc/driver/nvidia/version`
  says "NVIDIA UNIX Open Kernel Module"; `modinfo nvidia` reports Dual MIT/GPL).
  `nvidia-smi` has no fan option at all on GeForce, and `nvidia-settings` accepts
  `GPUFanControlState=1` but rejects every `GPUTargetFanSpeed` write with
  "Unknown Error" — with `Option "Coolbits" "4"` confirmed applied in the Xorg log.
  Switching to the proprietary `nvidia-driver-595` would restore it, but the burn
  above shows the automatic curve handles the card fine, so there is no reason to.
  `/etc/X11/xorg.conf.d/20-nvidia-coolbits.conf` is inert; safe to delete.
- **`nvidia-settings -a` exits 0 even when the write was rejected** — it prints
  `ERROR: ... (Unknown Error)` to stderr and still returns 0, the same class of trap
  as `nvidia-smi -pl`. Trusting the exit code would leave a card in *manual* fan
  mode with its fans parked at zero, strictly worse than not touching it. **Always
  read the attribute back and compare.** Note also that no attribute exposes which
  fan belongs to which GPU, and an unverified probe write "succeeds" on all four.
  `gpu0-fan-curve.service` is therefore a **user** unit: it takes over above
  60 °C and hands back to the driver's automatic curve below 55 °C, so idle stays
  in the zero-RPM band and the machine is no louder than before. GDDR6X memory-junction temp is **not readable** on Linux for GeForce —
  do not write monitoring that expects it.

### Inference backends (`src/chat/engine.py`, `src/chat/inference_backend.py`)

`get_engine()` is the process-wide singleton that loads the model + retriever once. Every surface generates through a pluggable `InferenceBackend`: the WhatsApp webhook (`engine.generate_reply`), the Gradio UI token stream (`web_app.py`), and follow-up `suggestions.py`. Two backends:

| Backend | What runs |
|---|---|
| `hf` | Unsloth `FastModel` / PEFT model **in-process** on the GPU (default). |
| `gguf` | Generation is sent to a llama.cpp `llama-server` over HTTP (`LlamaCppBackend`). The app process holds only the tokenizer + RAG retriever (~2 GB); the model lives in the `llama` compose service serving `models/gguf/gemma-4-12b-it-Q6_K.gguf` — **~15× faster** than the bnb-4bit in-process model. This is the only backend that can serve a model larger than one card, and the only one that works with the live profile at all (it has no adapter). |

`KAYA_LLAMA_URL` overrides `inference.gguf.server_url` (env wins), which is how a
benchmark run targets the `llama-bench` candidate server on `127.0.0.1:8081`
while leaving what prod resolves untouched.

Chosen by `resolve_backend()`: the `KAYA_INFERENCE_BACKEND` env var wins, else `inference.backend` in `config.yaml` (**default `gguf` since 2026-08-08**). Both prod and dev run `gguf`; `hf` only works with a profile that owns an adapter directory, and the live profile does not. GGUF files are gitignored — build one from a fine-tuned adapter with `scripts/export_gguf.py` (merge → `convert_hf_to_gguf.py` → `llama-quantize`). `LlamaCppBackend` strips the HF template's leading `<bos>` (llama.cpp adds its own) to avoid a quality-degrading double-BOS. The CLI `chat.py` is hf-only (dev tool).

### Audio (`src/chat/stt.py`, `src/chat/tts.py`)

Incoming voice notes are transcribed with faster-whisper (`large-v3`, int8_float16 on CUDA) and then flow through the ordinary text path, router included — a voice note arrives with **empty text**, so without transcription it is silently dropped. WAHA reports its media at `http://localhost:3000/...`, which is its own container, not ours: `rewrite_media_url()` swaps in the reachable base URL, and removing it breaks every voice note.

**What is spoken is not what is written.** `tts.sanitize_for_speech()` is applied
at the one point a reply becomes audio (`whatsapp_server._tts`) and strips the
sources line, emoji, markdown and bare domains. Without it Piper read
`🌐 Fontes: x.com, play.google.com` aloud, domain by domain, because the citation
was appended in `engine.respond` before delivery ever chose text or speech.
`Reply` now carries `citation` **separately** from `text`: appended for a written
message, sent as a short follow-up after a voice note (`chat.audio.send_citation_as_text`).
The interaction log records `delivered_as` and `spoken_text`, and
`MockWahaClient.send_voice` keeps the spoken text rather than only its byte count
— that missing field is why no test caught this.

Voice replies use Piper on CPU (~28× realtime, so speaking never competes with the GPU). Kokoro, the usual default, only ships Brazilian Portuguese. A Piper voice speaks **one** language, so `synthesize_wav()` splits the reply into sentences, groups consecutive same-language runs (`split_by_language()`, using `language_signal()` from `response_utils.py`), and speaks each with the voice configured under `chat.audio.voices` (`pt` → `pt_PT-tugão`, `en` → `en_GB-alan`) before concatenating the WAV and encoding once to OGG/Opus via PyAV (ffmpeg is not installed on this box). A sentence with no language marker inherits the previous one; a missing voice file falls back to `pt`. Voice replies are sticky per chat (`ChatPreferences`), set through the router's `CMD_AUDIO` / `CMD_TEXT`; `CMD_AUDIO_ONCE` is a one-off delivery hint that does not change the preference.

### Images (`src/chat/vision.py`, `src/chat/imagegen.py`)

**Reading them.** The serving model is multimodal, so `--mmproj` on the `llama` service is all it takes — no second model, ~180MB. An inbound photo is described (`vision.describe_url`) exactly the way a voice note is transcribed, and the description replaces/augments the message text. That reuse is the design: once it is text, the message log, the ingester, the router and retrieval need no changes, which is why "aquela foto do barco" is findable later. A caption is kept alongside the description. Without `--mmproj` the bot answers as though nothing were attached.

**Keeping the face (2026-08-12).** The `0.409` likeness in the bake-off was
measured with an English instruction ending `"Keep the face exactly the same."`
Production was sending the user's **raw Portuguese** with no such clause, at a
fixed seed, through two successive downscales. Four changes, all in
`chat.imagegen`: `identity_clause` is appended to every edit;
`translate_prompt` has the local model rewrite the Portuguese request as a short
English instruction first (Kontext's CLIP-L + T5-XXL are English-trained);
`face_crop` tightens the frame around detected faces when the largest is under
`face_min_ratio` of the width, and `src/chat/face_utils.prepare_source` resizes
**once** straight to a `PREFERRED_KONTEXT_RESOLUTIONS` bucket (`_auto_resize=False`);
`candidates` renders N takes from one model load and ships the one with the
highest ArcFace similarity to the source face. `seed: null` means a fresh seed per
request — the old fixed `1234` made a bad face reproducible.

**The likeness metric was partly measuring the wrong thing.** `image_bakeoff.py`
compared the *largest* face in the output against the largest in the source. Half
the bench photos are group shots by design, so on those it frequently compared two
different people. Re-scoring the 2026-08-09 run against the closest-matching face
moved p05 `0.030 → 0.246` and p06 `0.184 → 0.370` (flux-kontext mean `0.409 → 0.459`).
`medieval-king` stayed at `0.109`: head-covering edits are a real failure, not an
artefact. Face detection is InsightFace `buffalo_l` on CPU, mounted into the
containers from `~/.insightface`; every path degrades to a no-op without it.

**Not everything in a photo is a person (2026-08-16).** The whole edit path
assumed one. `imagegen_worker` chose its identity clause with `_face_count() > 1`,
so **zero** faces took the same branch as one and a photo of four Monster cans on
a shop counter was sent to Kontext with *"Keep the person's face … Do not change
who they are"* appended — by `config.yaml`'s own note, the instruction a model
best satisfies by changing nothing. `_INSTRUCTION_SYSTEM` was no better: five
few-shot examples all of the form *"Dress the person…"*, so an object request was
rewritten as a person edit before it reached the model. Then `pick_best` scored
two candidates against a `None` reference and returned the first.

Three changes. `_face_count` returns `Optional[int]` — `None` (cannot tell) keeps
the singular clause, a real `0` drops it via `select_identity_clause`.
`build_edit_instruction` now returns a fourth value, `SUBJECT: person|object|scene`,
from the same local call; `object`/`scene` sends `--no-identity-clause`,
`--no-face-crop`, `--candidates 1`. And `chat.imagegen.editors` maps subject to
editor — the single-editor choice came from a bake-off whose every photo was
picked for a large frontal face, so it ranked on identity preservation alone.

**The bench could not see any of it.** `pick_bench_photos.py` *selects for* "a
large, confidently-detected, roughly frontal face" and all five `EDITS` transform
a person. `--grid objects` (photos chosen with `--no-faces`, into
`data/bench_objects/`) runs `OBJECT_EDITS` and scores **preservation** — judged
1-5, "did everything the instruction did not mention survive?" — in place of
likeness, which is undefined without a face. The face grid keeps the
`likeness_x_adherence` key so its reports stay comparable with everything before.

**Ran it (2026-08-16, `reports/image_bakeoff/objects_20260816T164851Z`, 5 arms ×
4 photos × 5 edits).** Klein wins the object grid outright and is 2.5× faster:

| arm | presv | adh | presv×adh | usable | secs |
|---|---|---|---|---|---|
| **flux2-klein** | 4.78 | **4.78** | **0.852** | **17/20** | **67** |
| qwen-image-edit-2511 | 4.79 | 4.21 | 0.760 | 14/20 | 453 |
| flux-kontext-objects | 4.58 | 3.90 | 0.716 | 13/20 | 82 |
| flux-kontext-prod | 4.61 | 4.00 | 0.706 | 13/20 | 166 |

Two results worth keeping, both of which contradict what the fix was assumed to
be doing:

- **The identity clause was not the problem.** `flux-kontext-objects` (the fix:
  no clause, no crop, one take) scored 0.716 against `flux-kontext-prod`'s 0.706,
  and 13/20 usable either way. That difference is noise. What the fix actually
  buys is **half the wall time** (82s vs 166s), because best-of-N ranked by
  ArcFace against a photo with no face was two renders to pick the first one.
- **Every arm scores 5.0/5.0 on `swap-object`** — which is precisely the Monster
  request ("transform the cans into X"). So the live failure was almost certainly
  the *prompt rewriter* reframing an object edit as a person edit, not the editor
  and not the clause. This grid does not test the rewriter; its instructions are
  clean English that never mentions a person. That half is covered by unit tests.

Klein's margin comes from `recolour` (3.2→5.0) and `change-setting` (4.2→5.0).
**`remove-object` fails on everything** — Kontext scores adherence 1.0 on all four
photos, Klein 3.0 — so an object *removal* is the request most likely to come back
as "não consegui", which is at least honest: `noop_threshold` catches it rather
than sending the unchanged photo. Qwen matches Klein on quality and costs 453s an
image, 6.8× Klein's, which is not a trade a chat bot can make.

**And a render used to leave no trace at all.** `whatsapp_server` skipped any
result carrying a `command`, and `_handle_image_request` always sets one, so no
image request reached `live_interactions.jsonl`; the translated instruction and
final prompt went to a `logger.info` on a logger nobody configures; the output
lived in a `TemporaryDirectory`. The rule now lives in `metrics.should_log`
(testable — importing `whatsapp_server` loads the model), `imagegen.run` takes an
`on_report` callback that logs a `source="imagegen"` row, and
`chat.imagegen.keep_outputs` keeps the last N renders plus a JSON sidecar in the
gitignored `data/imagegen_log/`.

**Making them.** `imagegen.run()` shells out to `scripts/imagegen_worker.py` — never in-process. Dropping a diffusion pipeline in Python leaves ~20GB allocated with no `nn.Module` alive, it would hold a card between requests, and an OOM would take the bot down. Editing runs **FLUX.1 Kontext** (bake-off winner: likeness 0.409 vs Qwen's 0.138, adherence 4.4/5, ~86s); text-to-image runs **Z-Image Turbo**. The router's `CMD_IMAGE` picks the subject: attached photo, else the last photo seen in that chat, else generate from text. The webhook only acknowledges — the picture is sent from a background thread. `chat.imagegen.allowed_scopes` gates editing to the shared group.

**Kontext keeps its job (2026-08-12).** FLUX.2 Klein 9B ran the same 40-cell
standard grid and lost where it counts:

| arm | likeness | adherence | lik×adh | usable | realism | secs |
|---|---|---|---|---|---|---|
| `flux-kontext-prod` | **0.695** | 4.05 | **0.536** | 21/40 | 2.83 | 177 |
| `flux2-klein-prod` | 0.417 | **4.98** | 0.414 | **23/40** | **4.25** | **145** |

Klein follows instructions almost perfectly and looks more photographic, but a
40% drop in likeness is the wrong trade for a bot whose images are jokes about
specific faces. Two more usable cells do not buy that back.

**The two-stage restore is built but OFF** (`chat.imagegen.restore_face`). The
editor commits to the scene and a LoRA (`Alissonerdx/BFS-Best-Face-Swap` on Klein
9B) puts the face back, gated on the local model answering `FACE: keep` vs
`FACE: change` — restoring the original face onto "faz dele um zombie" would undo
the request. A malformed answer skips the restore, i.e. today's behaviour. The
result is kept **only if it beats the editor's own output** on ArcFace similarity
to the source. It runs (GPU0; it OOMs on GPU1 because the LLM lives there), and
on its one measured sample the guard discarded it: 0.811 restored against 0.881
unrestored. It costs ~90s and a second 17GB load, so it stays off until the
40-cell grid says it earns them.

**JPEG on the wire, PNG on disk.** `send_image` base64-inlines the bytes into the
WAHA JSON body and WhatsApp recompresses to JPEG at the far end regardless, so
`imagegen.run` re-encodes at `jpeg_quality` (92) before returning — a few hundred
KB instead of several MB. The **worker keeps writing PNG**: `image_bakeoff.py`
scores those files with ArcFace and must not be measuring compression artefacts.
A failed re-encode returns the original bytes rather than nothing.

**Quantization rules learned the hard way** (`reports/PHASE5_IMPLEMENTATION.md`): NF4 turns a 20B diffusion transformer's output into a crystalline mosaic — use 8-bit. **Never `enable_model_cpu_offload()` with bitsandbytes weights**: the hooks duplicate rather than move them, which took FLUX from 14.5GB to 22.4GB and OOMed every image. `device_map="balanced"` across both cards is *slower*, not faster (no P2P). Two cards buy parallelism across images, not within one.

### Who is being talked about, and counting (2026-08-16)

**Mentions were numbers.** `_strip_bot_mention` removed only the *bot's* `@` token;
everybody else's stayed as a bare `@lid`. So `@257487651496102 tas fraquinho` said
nothing about Rafa to the model and nothing to `extract_query_persons`, which
matches member *names* — and the roast, handed the usual pile of profiles with
nobody named in the question, went to Manuel, who was not in the conversation.
Both filed reports of the bot "referencing the wrong people" are this.
`_resolve_mentions` rewrites each `@<lid>` via the existing `_name_for_jid`,
applied to the responder text **and** to `message_log.append` (that log is
embedded into ChromaDB; a message stored as a number is unretrievable by a
question about Rafa). An unknown lid is left intact — deleting it would turn
"@X e o @Y" into a sentence about one person.

Two supporting fixes: `_name_for_jid` now tries the `@lid` shape (most of
`whatsapp_contacts.json` is keyed that way, and a body mention arrives with no
suffix at all), and `resolve_speaker` learns a member's *other* ids when it
matches one — four members were mapped by phone, so they never reached the
learning branch and their `@lid` stayed unknown: perfectly identified as
speakers, invisible when someone @-ed them. `_learn_contact(verified=True)`
suppresses the display-name-collision warning there, since two ids for one member
is the normal case.

**"O Gustavo tem de tratar disso", said to Gustavo.** Not misidentification — that
sentence is a verbatim template in the banter/mixed/detailed prompts, and a
technical complaint fired it exactly as written. The clause is now
`{maintainer_clause}`, filled by `engine.apply_speaker_rules` per turn:
`chat.maintainer_self_clause` when the speaker *is* `chat.maintainer`. Both
builders call `fill_prompt_defaults` so surfaces that do not know the speaker
(the web UI does not go through `respond`) never leak the placeholder. The
third-person rule in `data.system_prompt` is deliberate and stays.

**Counting is not retrieval** (`src/chat/tally.py`). Top-k semantic search returns
the chunks nearest a question and cannot answer "how many times". Asked for a
per-member tally the bot wrote a confident table that was out by 8x with the
ranking inverted (the top user, 198, reported third at 3), then agreed when told
it had probably missed some. `CMD_COUNT` routes those; `engine._count_context`
counts the log and hands the model a finished table to phrase. It is scope-bound
(a DM counts only its own file), folds aliases through `SenderResolver`, and
includes the pre-bot export — the group is older than the bot. A term it cannot
identify returns nothing rather than a number. The prompts also stopped accepting
standing jobs the bot has no state for ("Consigo manter o contador atualizado",
then "Aí está, Frederico" with no list).

### An opinion may vary, a fact may not (2026-08-16)

Peter asked to be roasted four times over three days and got the same four beats
every time: Rotterdam and Queijas, editing other people's videos, Five Guys,
posting concert videos like a music critic. Romano got *"analista político por
ler tweets"* five times, Rafa *"ginásio próprio"* and *"o Iñaki no sparring"*
five times. The model was not at fault — it was handed identical material every
turn, by two mechanisms:

- **`whatsapp_server` builds the system prompt once, at import.** So the member
  profiles, *including the `shuffle=True` meant to vary them*, were byte-identical
  for the whole uptime. `engine.system_prompt_factory` now rebuilds it per turn
  (~0.6 ms) — but only for open-ended turns.
- **`key_facts[:max_facts]` truncates.** With `max_facts_per_member: 4`, Peter's
  first four of five facts went out in the same order forever, and the fifth
  (*he owns a dog called Kobe*, *he hosts the group and organises the football*)
  had **never been shown to the model at all**. `sample_facts=True` draws a random
  handful instead; `rag.max_facts_open_ended` (3) makes that ten different
  triples for Peter rather than one.

`src/chat/variety.py` adds the third piece: what the bot has **already said**
about this person. `previous_bot_replies` only ever covered the last few turns of
one chat, so it could not see the same roast repeated three days later in a
different thread. The interaction log already records `reply_members`, so the
material was on disk and simply never read back. The subjects of a turn are the
members named in the message **plus the speaker** — which is how "roast me"
resolves to the person asking, the exact case that repeated. Noise is filtered
out: only open-ended rows count (a count table names everyone and is nobody's
joke), replies under 8 words carry no angle, and a reply naming more than 5
members is about none of them.

**`variety.is_open_ended` is the gate, and `factual` is deliberately outside it.**
Sampling facts for a factual answer would make *"o que faz o Gil?"* depend on
whether his job survived the draw, and `CMD_COUNT` borrows the factual mode
config — it must not borrow this. Variety is for roasts, insults and opinions;
a count must come out the same every time it is asked.

### Slash commands, and why they must not be remembered (2026-08-13)

`/clear` (`/limpar`), `/bug` (`/erro`) and `/feedback` (`/sugestao`) are matched
literally in `whatsapp_adapter.handle_event` and never reach the model. `/bug`
and `/feedback` take the rest of the message as the body; sent bare they reply
with usage and store **nothing** — deliberately no pending-capture state, so an
unrelated next message can never be swallowed into someone's report. They reuse
`feedback.log_bug_report` and the new `feedback.log_note` (the latter exists
because `log_comment` joins to an earlier 👍/👎 by `feedback_id`, and a
standalone `/feedback` has no rating to attach to).

**The trap: `message_log.append` runs BEFORE the reply gate.** Everything the bot
sees is logged first — that is the whole point, group chatter it was not
addressed in is the memory worth keeping — and `src/data/ingest.py` folds that
log into ChromaDB. So a command left unfiltered becomes a searchable thing "the
group said", and a week of bug reports would come back out of retrieval.
`_is_command()` guards the append, **mention-stripped first** because in a group
the text arrives as `@Kaya /bug ...`. This fixed `/clear` at the same time; it
had been leaking since it was written, unnoticed only because nobody had used it.

New reports are announced by DM to `KAYA_REPORT_JID` (env, not `config.yaml` — a
real number), and a report filed *in the group* also DMs its author a private
copy; from a DM that would be the same message twice, so it is not sent. Sending
is done in the adapter, not in `feedback._notify_bug_report`: that seam has no
WAHA client and stays reserved for email. Any send failure is swallowed — the
report is already on disk.

### Conversational memory (`src/chat/summary.py`, `src/data/ingest.py`)

**The model was never the constraint — the prompt was.** Only the last 6 turns
reached the model verbatim while the served context is 32768 tokens and needle
recall is 60/60 out to 27,411. `whatsapp.history_turns` is **60** and
`inference.history_max_words` (**40**) truncates each line, so the recent thread
is carried instead of re-retrieved.

Past that window a per-chat **rolling summary** (`ChatSummaryStore` +
`SummaryWriter`) is refreshed on a background thread and prepended to the user
turn. Two rules matter: the writer takes `gpu_section()` and **skips on
`GpuBusyError`** rather than queueing, so summarising never delays a reply; and
**banter never receives the summary**, for the same reason banter retrieves
nothing — hand the model a digest of the group and it will find someone to talk
about.

Ingestion is incremental and watermarked. `build_chunks` returns
`(chunks, consumed_through)` and a chunk within `settle_minutes` (**10**) of now
is left for the next pass, because a chunk closed mid-conversation is a chunk
that can never be extended. The watermark is clamped to `consumed_through`, so an
unsettled tail is not marked as read.

### Conversation simulator (`src/testing/persona_sim.py`, `scripts/run_conversation_sim.py`)

The unit suite proves the wiring and `preflight_e2e.py` proves each capability in
isolation. Neither reproduces **an evening in the group**: several people talking
over each other, a photo arriving mid-argument, an edit requested while the last
one is still rendering, and a thread long enough to overflow the 14000-token
retrieval budget. That is what this is for.

**It drives the real webhook.** Every message is a synthetic WAHA event POSTed to
the `kaya-sim` service, which runs the production `whatsapp_server` under
`KAYA_WHATSAPP_MOCK=1`: parsing, routing, the GPU lock, scoping, media and the
async image path are all the real ones — only the outbound WhatsApp client is a
mock. In mock mode the webhook **awaits** generation and returns the result dict,
so a beat asserts on the routing decision instead of guessing it from the reply.

**Deterministic spine, improvised filler.** Free-form LLM chatter cannot be
asserted on, so a scenario is a list of beats: `say` beats carry exact text and
expectations, `improv` beats ask the Grok personas for natural conversation so
the context the bot sees is real rather than a list of probes. Assertions live on
the scripted beats only.

Three things are load-bearing:

- **`kaya-sim` deliberately does not `extends: kaya-base`.** The base hard-mounts
  `./data`, and the simulator invents conversations that would then be logged as
  group memory and ingested into the real vector store. It gets `./data_sim`,
  seeded by `scripts/seed_sim_data.py`.
- **Its GPU split mirrors prod** — app on the LLM's card, image worker on the
  other. Arranging it the other way left FLUX ~19GB instead of 23.5GB and every
  *edit* OOMed while generation still worked, which reads as "edits are broken"
  when it is really "the rig was arranged differently from production".
- **Message ids are unique per run** (`uuid4` prefix, not a counter). The sim
  container outlives a run, so a restarting counter made the adapter's replay
  guard — which is correct — treat the second run as the first run's backlog and
  ignore every message.

Presets: `smoke` (3 people, no image rendering, ~40s), `standard` (4 people, the
full feature surface, ~10 min), `long_haul` (5 people, overflows the retrieval
budget and the session window, ~28 min). Reports land in `reports/sim/<stamp>/`
with an `index.html` contact sheet; the run exits non-zero on any failed
assertion, so it can gate a deploy. Personas cost a few cents to ~€1 per run.

### Config System (`src/config_loader.py`)

Single entry point: `load_config(path, profile_override=None)`. Profiles (defined under `model_profiles` in `config.yaml`) deep-merge into the top-level `model:` and `training:` sections. The active profile is set by `active_model_profile` in `config.yaml` or passed via `--profile` CLI flag. **All code paths must use `load_config()` — never read `config.yaml` directly.**

### Web search (`src/chat/web_search.py`)

Grok's web-search answer is **context, not the reply** (`web_search.synthesize_locally`,
default true). Returning it verbatim bypassed the persona, the uncensored preamble
and `clean_response`, and Grok answers only the half of a message it considers
web-answerable — the logged result was a Ronaldo/Messi comparison followed by
*"A primeira parte da pergunta não se enquadra em resposta factual baseada na web."*
The local model now writes every reply, so one voice answers the whole message.
Set `synthesize_locally: false` to get the old behaviour back. The privacy guard is
unchanged: a query naming a group member never leaves the box.

### LLM Providers (`src/llm_providers/`)

Unified `LLMProvider` interface with `_retry_with_backoff()` for rate-limit resilience. Azure OpenAI (`azure_provider.py`) and xAI Grok (`xai_provider.py`); switch via `generation.provider` in `config.yaml`. **Eval-judge + web-search only** — never send group data to these providers; knowledge extraction and synthetic generation use the local teacher (`src/data/local_teacher.py`).

### Fine-tuning (`src/finetuning/train.py`)

Uses Unsloth (`FastModel` / `FastLanguageModel`) for Gemma4 and Qwen3. Training calls `SFTTrainer` directly (no wrapper class — a previous `KayaTrainer` wrapper caused 20+ GB RAM spikes). LoRA adapters are saved to `training.output_dir`. Inference expects `adapter_config.json` in the model directory.

### Deployment (`DEPLOYMENT.md`)

`kaya-prod` is the **always-on** production web app. The box is **serving-only** (fine-tuning is done separately). Access is via a **Cloudflare Tunnel** (`cloudflared` compose service, `tunnel` profile). The UI header shows the running env + commit (`KAYA_ENV`/`KAYA_VERSION`).

**Two routes, one process** (`whatsapp_server.py`):

| Path | Who gets in | What it is |
|---|---|---|
| `/` | **anyone** | the public explainer (`src/chat/static/landing.html`) — simple / in detail, EN / PT, with a login button |
| `/app` | `KAYA_WEB_USER` + `KAYA_WEB_PASS` | the Gradio chat |
| `/whatsapp/*` | WAHA | the webhook — must stay open, it is how messages arrive |

**The `auth=` on `mount_gradio_app` is load-bearing and was missing until 2026-08-13.** Prod runs `whatsapp_server`, *not* `web_app.__main__`, so the credentials were set in the deployed environment and silently ignored — `GET /config` served the whole app, member profiles included, to anyone. Cloudflare Access was the only thing in front of it, which is not what this file used to claim. `tests/test_landing_page.py` parses the mount and fails if the argument disappears again (it cannot import the module: `engine = get_engine(config)` runs at import and loads the model).

**Prod runs from its own checkout** at `~/kaya-prod` (separate from this dev copy), with `models/` and `data/` symlinked to the shared originals — so you can develop here without touching the live site. All four live services (`kaya-prod`, `kaya-waha`, `kaya-llama`, `cloudflared`) have `restart: unless-stopped`, and Docker here is the **snap** build — the unit is `snap.docker.dockerd.service`, so `systemctl enable docker` returns `not-found` and enables nothing. With `snap.docker.dockerd` enabled the stack **auto-recovers after a reboot** (measured: daemon +7s, `kaya-llama` +13s from boot). The one way it breaks is an *explicit* `docker stop` / `app_down.sh` before shutdown — that survives the reboot as "stopped". A daemon-initiated stop during `systemctl poweroff` does not.

Prod serves generation from the `llama` compose service (`gguf` profile) with `KAYA_INFERENCE_BACKEND=gguf` set on `kaya-prod`; `deploy_prod.sh` starts the `llama` server automatically. **Roll back to the in-process model** with `KAYA_INFERENCE_BACKEND=hf scripts/deploy_prod.sh`. Note: `~/kaya-prod/data/` must contain the gitignored runtime files (`rag_db/`, `group_members.json`, `whatsapp_whitelist.json`, `whatsapp_contacts.json`) — if `data/` is a real dir instead of the intended symlink to the dev copy, copy them in or RAG/whitelist gating silently fail.

**Push to prod:** `scripts/deploy_prod.sh [ref]` checks out the ref in `~/kaya-prod`, rebuilds, and restarts the live container — that is what makes a commit live. CI/CD on a **self-hosted GPU runner**: `ci.yml` tests every PR; `validate-main.yml` rebuilds + tests on merge to `main` (no container start); `deploy-prod.yml` (manual, `prod` Environment requires reviewer approval) calls `deploy_prod.sh` to update the live site. `kaya-dev` (port 7861) is for occasional manual dev runs only; run one env at a time, since a served model may claim both GPUs. Full runbook in `DEPLOYMENT.md`.

---

## Gemma 4 Specifics

These are easy to break — treat them as hard rules:

- Use `FastModel` (not `FastLanguageModel`) with `unsloth>=2026.4.5`
- Chat template: `get_chat_template(tokenizer, "gemma-4")` → produces `<|turn>user\n...<turn|>\n` format
- **Thinking mode must be disabled during SFT** — do not enable `<|think|>` tokens in training
- Inference must use `Gemma4ForConditionalGeneration.from_pretrained()` or Unsloth's `FastModel` — it is **not** registered with `AutoModelForCausalLM`
- Unsloth returns a `Gemma4Processor`, not a plain tokenizer. Always use `tokenizer(text=input_text, ...)` with the `text=` keyword — positional args are interpreted as `images` and will crash
- Set `autocast_adapter_dtype=False` for PEFT compatibility

---

## Training Memory Rules

Training is capped at **one 24 GB card** — see the GPU topology section above for
why the second 3090 does not raise this ceiling. To avoid OOM:

- `skip_memory_metrics=True` — avoids the HF `TrainerMemoryTracker` busy-loop
- `dataset_num_proc: 1` — prevents fork-based memory duplication
- `dataloader_pin_memory: False`, `dataloader_num_workers: 0`
- OOM fallback: lower `lora_r` to 8 and/or reduce `max_seq_length` to 2048
- VRAM budget: gemma4-e4b ~11 GB, qwen3-14b ~15 GB. Always leave ~2 GB headroom.

---

## PEFT `float8_e8m0fnu` Patch

PEFT 0.19.0 checks for `torch.float8_e8m0fnu` which doesn't exist in PyTorch 2.6. Two files in the venv are manually patched with `hasattr` guards:
- `kaya_chatbot_env/lib/python3.12/site-packages/peft/tuners/tuners_utils.py`
- `kaya_chatbot_env/lib/python3.12/site-packages/peft/tuners/lora/layer.py`

**Reapply these patches if PEFT is reinstalled or upgraded.** The fix wraps `torch.float8_e8m0fnu` references in `hasattr(torch, "float8_e8m0fnu")` guards.

---

## Package Version Pins

- `trl<=0.24.0` — newer versions break `SFTConfig` API
- `unsloth>=2026.4.5` — required for Gemma 4 via `FastModel`
- `transformers>=5.5.0` — required for `Gemma4ForConditionalGeneration`

---

## Coding Conventions

- No backup or temporary files when rewriting — replace in place or create new and delete old
- No inline comments unless requested; no license headers
- No one-letter variable names
- Fix root causes, not surface patches; keep changes minimal and consistent with existing style
- `political_preference` is stored in `group_members.json` but **never** embedded into ChromaDB vectors
- After any change, test in Docker to verify containerized behavior
