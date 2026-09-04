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

**A follow-up is not classified on its own words (2026-08-17).** The router saw
`recent_lines[-2:]`, so mid-thread about the Bernardo, "Muda a tua opinião, agora
que entendeste que é o bana?" came back **GENERAL** — the one mode forbidden from
naming a member — and answered about him with retrieval off. Eleven GENERAL turns
in the live log named somebody. Three changes, all in the one call the router
already makes:

- `chat.router.context_lines` (**6**) instead of two, plus explicit continuation
  rules. A reaction stays BANTER however much the thread is about a person —
  without that clause, six lines of context turned a third of banter into `mixed`
  ("Conheço uns quantos ya" became a question about Gil).
- the router also emits `Q: <standalone question>` on a second line, carried on
  `Route.query` and used for **retrieval only** — the model still reads what the
  person wrote. "E de bater na mãe?" becomes "quem do grupo tinha maior
  probabilidade de bater na mãe?"; "Do que é que se trata a minha start up?"
  becomes "a startup do Pedro". No `Q:` line means the raw message, i.e. the old
  behaviour. `_parse` reads the label from the **first line only**, or a rewrite
  restating the question would outvote it.
- `router.reconcile` downgrades GENERAL→MIXED when the message names a member who
  is **already in the recent lines**. Naming one is not enough on its own: "o Gil
  também acha que o Ronaldo é melhor, e tu?" is a question about Ronaldo.

A **correction about a member** routes FACTUAL even though it is not a question:
checking it needs the profiles, and only the retrieving modes carry them.
Verify with `scripts/replay_routing.py`, which re-routes a logged session against
the interleaved history it actually had.

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
kaya_chatbot_env/bin/python scripts/run_conversation_probe.py   # routing/brevity/restraint/in-voice/no-dash/compliance (cases may carry `history` and `accept`)
kaya_chatbot_env/bin/python scripts/replay_routing.py --date YYYY-MM-DD  # re-route a logged session against the history it really had
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

**The whole bot runs on ONE card (since 2026-09-04).** Prod is `NVIDIA_VISIBLE_DEVICES=1` +
`CUDA_VISIBLE_DEVICES=0` — llama-server ~12.9 GB (weights 9.1 + mmproj 0.17 + KV)
and the app process ~4.3 GB (Whisper large-v3 + the bge-m3 embedder + CUDA
context), **17.2 GB of 24.6 GB, with ~7 GB spare** at the configured 32768
context. GPU0 holds the desktop and nothing of Kaya's.

This is not a downgrade — it is what was already happening. The prod llama
command has been `-sm none` since the 12B landed, and the *only* thing that ever
used the second card was the image worker, which is gone. What changed is that
prod no longer *reserves* GPU0: `NVIDIA_VISIBLE_DEVICES` was `all` purely so a
20 GB diffusion pipeline had somewhere to run that was not on top of the LLM.

`NVIDIA_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are **not redundant**: the
first picks which physical card the container is handed, the second indexes what
the container can already see. Expose one card and it is index **0** inside,
whatever its host index. Getting that pair wrong is how a service silently lands
on the desktop card.

The box still has **two 24 GB RTX 3090s and no NVLink bridge**, and everything
below still applies to anything that tries to use both. They are two separate
devices, not a 48 GB pool: `can_device_access_peer(0,1)` is False and `nvidia-smi
topo -p2p` reports `CNS`, so there is **no GPU-to-GPU P2P** and all inter-GPU
traffic stages through system RAM.

| | Serving today | Serving, two-card (available, unused) | Python (training, hf backend, CI) |
|---|---|---|---|
| `NVIDIA_VISIBLE_DEVICES` | `1` | `all` | `0` |
| `CUDA_VISIBLE_DEVICES` | `0` | `0,1` | **`0`** |
| `-sm` | `none` | `layer` | — |
| Ceiling | 24 GB | ~45 GB weights+KV | 24 GB |

- **kaya-dev still owns GPU0 and can run alongside prod.** That got *cleaner*,
  not worse: prod used to reserve the dev card for renders, so "starting
  alongside it" meant sharing after all the moment somebody asked for a picture.
- **Serving can still exceed 24 GB** by layer-splitting across both cards — set
  `KAYA_GPU_PROD=` (empty → `all`) and `KAYA_PROD_SM=layer`. `llama-bench`
  (profile `bench`) is the one service that routinely does this, since it exists
  to score models bigger than one card; it takes `KAYA_BENCH_CVD=0,1` alongside
  an empty `KAYA_GPU_BENCH`.
  Only the hidden state crosses PCIe at the layer boundary.
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
  serving used `-ts 0.45,0.55` to give it fewer layers. That gap was measured at
  the old 250 W cap; at 300 W it sustains ~1620 MHz. **Nothing sets `-ts` today**
  — prod is single-card — so re-measure before trusting 0.45/0.55 if you ever go
  back to a split. The gap is partly real; see below.
- **GPU0 is cooling-limited, not power-limited. The "intake-starved" note in
  `gpu-power-limit.sh` is correct — 300 W is its ceiling.** This was measured
  against FLUX renders, which no longer happen, so GPU0 now sits idle unless
  kaya-dev is up. The finding stands and is why the cap must not be raised.
  Measured 2026-08-16 with
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
  `gpu0-fan-curve.service` is a **user** unit: it takes over above 60 °C and hands
  back to the driver's automatic curve below 55 °C, so idle stays in the zero-RPM
  band. It exists for FLUX renders heating GPU0 and is **disabled and inactive**;
  with generation gone there is nothing left to cool. `/usr/local/bin/gpu0-fan-curve.sh`
  and `/etc/X11/xorg.conf.d/20-nvidia-coolbits.conf` are both inert; safe to delete. GDDR6X memory-junction temp is **not readable** on Linux for GeForce —
  do not write monitoring that expects it.

### Inference backends (`src/chat/engine.py`, `src/chat/inference_backend.py`)

`get_engine()` is the process-wide singleton that loads the model + retriever once. Every surface generates through a pluggable `InferenceBackend`: the WhatsApp webhook (`engine.generate_reply`), the Gradio UI token stream (`web_app.py`), and follow-up `suggestions.py`. Two backends:

| Backend | What runs |
|---|---|
| `hf` | Unsloth `FastModel` / PEFT model **in-process** on the GPU (default). |
| `gguf` | Generation is sent to a llama.cpp `llama-server` over HTTP (`LlamaCppBackend`). The app process holds only the tokenizer + RAG retriever (~2 GB); the model lives in the `llama` compose service serving `models/gguf/gemma-4-12b-it-Q6_K.gguf` — **~15× faster** than the bnb-4bit in-process model. This is the only backend that can serve a model larger than one card, and the only one that works with the live profile at all (it has no adapter). |

**`cache_prompt` is ON (2026-09-04), and was off for no reason.** It arrived
`False` with the original GGUF backend commit carrying no rationale — a copied
default — so llama.cpp re-prefilled the whole prompt on every call, twice per
turn. A turn is two calls: `router.classify` and the reply. The router's system
prompt is **~2,350 tokens and byte-identical on every message**, including "😂",
to produce a one-word label.

Measured against the live server, medians over 10 turns:

| | ideal (one repeated 3,015-token prefix) | realistic (router/reply interleaved) |
|---|---|---|
| `cache_prompt: False` | 3.01 s/call | 11.49 s/turn |
| `cache_prompt: True` | **1.45 s/call** | **9.26 s/turn** |

The interleaved number is the one that matters and is smaller for a structural
reason: `--parallel 1` means **one slot and therefore one cached prefix**, so the
router and reply prompts partly evict each other. Giving each its own slot would
need `--parallel 2`, which splits the KV budget — 32768/2 = 16384 per slot would
halve the context the needle-recall work depends on, so it would also need
`-c 65536` and about 3.4 GB more KV (17.2 → ~20.6 GB of 24.6). That fits, but it
has not been measured; do not change `--parallel` without checking recall first.

Note also that the reply call's prefix is *designed* to vary on an open-ended
turn — `sample_facts=True` reshuffles the member profiles every time, which is
the anti-repetition fix and is worth more than a cache hit.

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

### Images (`src/chat/vision.py`) — read only

**Reading them.** The serving model is multimodal, so `--mmproj` on the `llama` service is all it takes — no second model, ~180MB. An inbound photo is described (`vision.describe_url`) exactly the way a voice note is transcribed, and the description replaces/augments the message text. That reuse is the design: once it is text, the message log, the ingester, the router and retrieval need no changes, which is why "aquela foto do barco" is findable later. A caption is kept alongside the description. Without `--mmproj` the bot answers as though nothing were attached.

`vision.py` depends on nothing but PIL (`flatten_animation`, for animated-WebP
stickers), `stt.rewrite_media_url` and an HTTP POST to the same llama-server.
There is no second model and no GPU of its own.

**Making them was removed on 2026-09-04.** Generation and editing are gone —
`imagegen.py`, `face_utils.py`, `imagegen_worker.py`, the bake-off harness, the
GPU0 lease and ~264GB of diffusion weights. Two weeks of live logs recorded
**one** image request, and it was wrong: *"mete maquilhagem de palhaço nesta
cara"* routed as `generate` rather than `edit`, so with no source photo it
invented a stranger's Joker face, took 62 seconds, and was logged `ok: true`.
That is the whole production record for the feature. It cost a reserved GPU,
~1,500 lines, three test files and 150 lines of `config.yaml`.

**`CMD_IMAGE` stays in the router on purpose.** Dropping the intent would let
"faz uma imagem de um gato astronauta" fall through to GENERAL, where the model
answers conversationally — in practice by describing the picture it is not
making, or by promising to send one later. The command now returns a fixed
`image_unsupported` line: no LLM call, no GPU, nothing it can promise. The two
router examples that route a question *about* a photo to FACTUAL ("quem está
nesta foto?", "manda a foto do jantar") are what keep photo questions working
and must stay.

Image turns are still written to the interaction log (`metrics.should_log` keeps
`image` out of `BOOKKEEPING_COMMANDS`): how often the group keeps asking anyway
is the only evidence there will be about whether removing it hurt.

The historical bake-off reports stay in `reports/image_bakeoff/` — they contain
real photos of real people, so the `.gitignore` guard stays with them.
`reports/PHASE5_IMPLEMENTATION.md` is kept for the same reason: its hardware
findings (NF4 destroys a 20B diffusion transformer, never
`enable_model_cpu_offload()` with bitsandbytes weights, `device_map="balanced"`
is *slower* here because there is no P2P) cost real time to measure and would
have to be re-learned by anyone trying a diffusion model on this box again.

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

**And the same sentence, not just the same material (2026-08-17).** All of the
above guards WHO a joke is about. 198 of 339 routed turns were banter and they
recycled the shape: *"Estás só a tentar X mas Y"*, *"É só mais um exemplo de
alguém X"*, *"Pelo menos eu não Z"*. `variety.recent_openers` reads the first
three words of the bot's recent replies **in that mode** out of the interaction
log — cross-chat and cross-session, which `previous_bot_replies` (one chat, last
four lines) structurally cannot be — and tells it not to open that way again.
Banter and mixed only, `inference.variety_recent_openers` (**6**), and it runs
even when nobody is named: a banter reply is usually about nothing, and it is
banter that repeats itself.

### A roast is about one person (2026-09-04)

Two of the three roasts in the fortnight to 2026-09-03 answered the request and
then appended a whole paragraph about somebody who was not in the conversation.
*"convence o Gil a ficar até mais tarde"* roasted Gil, then Frederico — Gustavo:
*"Ninguém te perguntou nada do Fred"*. *"Say gugu's mom is a hot momma milf"*
answered, then went after Gil — Gustavo: *"O gajo a alucinar"*.

Roast is the only mode that combines the full detailed system prompt
(`system_prompt: null`), **all 15 member profiles** reshuffled per turn
(`variety.OPEN_ENDED` includes roast, so `sample_facts=True`), full-depth RAG at
`top_k: 10`, and a budget 4× banter's. Banter and mixed each carry a scoping
clause; roast carried none. Worse, its `mode_hint` ended with *"se o pedido não
disser em quem, escolhe alguém que não tenha sido gozado nas últimas mensagens, e
varia"* — appended to **every** roast including the aimed ones, i.e. a standing
order to go and find a fresh victim after answering.

`engine._roast_hint` already computed the only condition under which that clause
is safe (nobody named) and correctly returned `""` for an aimed roast — it just
could not suppress the standing hint. The varying clause moved into it, and it
now emits the opposite instruction when a target *is* named: *"O roast é sobre X.
Fala só dessa pessoa e de mais ninguém."* The `mode_hint` keeps only what is true
of every roast, plus the scoping clause banter and mixed always had.
`max_new_tokens` 200 → **120**: the three logged roasts ran 69, 78 and 104 words
against banter's 6–16, and 200 is room for a second paragraph the model will fill.

`ROAST` stays in `_CAN_ELABORATE`. `mode_hint` is deliberately **not** gated on
`wants_long_answer` — unlike `brevity_hint` it says how to answer, not how long to
be — so the scoping clause survives *"justifica com tudo o que tens"*. That
separation is what makes elaboration safe here.

### Agreeing is not answering (2026-08-17)

`data.system_prompt` said *"Se alguém te corrigir, reconhece o erro"* — with
nothing about checking first — and the banter prompt said *"aceita a correção"*.
So told *"este bernardo é o bana já agora, não é o benny pereira burro"*, the bot
answered *"Tens razão, Gustavo, enganei-me completamente… foi uma burrice minha"*
— while `bana` and `benny pereira` were both listed in its own prompt as aliases
of the same member. It apologised for a correct answer, to a wrong correction.
Forty seconds apart it also said Romano worked at Glovo, then flipped to Gil and
accused Romano of *"desinformação"*; Romano has no `occupation` at all and Gil
*"works remotely from WeWork in the Glovo building"*. Neither was right and it
never said "não sei". Pedro got invented startup detail until he wrote *"Bruv is
hallucinating hard"*.

The clause is now conditional in all three prompts that carry one (detailed,
banter, mixed): verify against the profiles, accept in one sentence if it holds,
hold the position and say why if it does not, say you do not know if you have
nothing — and **never open with "tens razão", "peço desculpa", "confundi" or
"enganei-me"**. That last line is what finally moved it: given the check, the
model found the right fact and then apologised anyway. `build_member_prompt_suffix`
states that the names in *"também lhe chamam …"* are one person, and the detailed
prompt adds a no-silent-flip rule and a no-embroidering rule for a member with only
a generic line. `scripts/audit_interactions.py` wanted *"tens TODA a razão"* and so
scored this morning clean on the very failure it exists to catch; its regex and its
remediation pointer (which recommended the clause that caused this) are fixed.

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

**An unknown `/command` used to be answered by the model (2026-09-04).**
`_parse_command` was an exact-token lookup over seven tokens with no
`startswith("/")` branch and no `/help`, so `/feature have better update of
facts` fell through, routed GENERAL, and came back *"Understood. I will
prioritize and integrate new information more aggressively… Expect more relevant
updates in our future interactions"* — a promise the bot has no state to keep.
And because `seen` is `(… and not _command)`, the whole line went into
`message_log` and from there into ChromaDB: a feature request, permanently
retrievable as something the group said. Exactly the leak `_is_command` exists to
stop; it simply did not know `/feature` was a command.

A **leading** `/word` now returns family `"unknown"` and gets a fixed usage line
from code. Leading only: mid-message is where the *known* commands are found, but
treating any stray slash as a command would swallow "vamos dia 12/09" and
"sim/não". Nothing is stored, and there is deliberately still no pending-capture
state, so the next message cannot be swallowed.

The promise itself had a second cause: **`general` was the only mode prompt
without `{maintainer_clause}`**, the clause that says the bot has no state and
must not promise to keep counters or lists updated. All four prompts that carry
one now do, pinned by a test.

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

**Those lines are the whole room now (2026-08-17).** `session_store.append` ran
only on turns the bot ANSWERED, and in a group it answers on a mention or a reply,
so its history was a thread of its own mentions stitched to its own replies. One
live morning: 46 messages in the room, 29 in the prompt. It never saw "Bruh nunca
vi programador tão fraco" or "Bruv is hallucinating hard", which is why a "toma
aí" between them came back as an unrelated stock insult. The durable `MessageLog`
had all of it; it just never reached the model. Every message the bot sees is now
appended **before** the reply gate, mention-resolved, with the same
`_is_command()` guard the log uses — this window feeds the rolling summary, and a
week of `/bug` reports must not become things "the group said".

Three consequences. The answered message is appended once, before the gate, and
dropped from the `recent` it is handed back (it would otherwise arrive as both the
question and something already said). `KeyedSessionMemory.max_lines` went 2x→3x
`history_turns`, since a busy group writes several inbound lines per reply. And
the retrieval-exclusion window is no longer a fraction: `_inbound_window` is gone,
`_note_message_time` fires for every message, and `_session_window_start` takes the
count of non-`Kaya Bot:` lines actually being sent. Getting that wrong opens a
**hole**, not a duplicate — retrieval would drop chunks nothing carries verbatim.
It is close to token-neutral: the window is capped in lines, so denser lines cover
less time rather than costing more prefill.

Past that window a per-chat **rolling summary** (`ChatSummaryStore` +
`SummaryWriter`) is refreshed on a background thread and prepended to the user
turn. Two rules matter: the writer takes `gpu_section()` and **skips on
`GpuBusyError`** rather than queueing, so summarising never delays a reply; and
**banter never receives the summary**, for the same reason banter retrieves
nothing — hand the model a digest of the group and it will find someone to talk
about.

**It was dead for three weeks, and a counter is why (2026-09-04).** The live
group's last summary was written 2026-08-13; ~1,500 messages later nothing had
changed. Not a `GpuBusyError` skip — that path retries correctly and was never
reached. `SessionMemory.save` capped history at an unconditional
`MAX_SAVED_MESSAGES = 100`, *below* the `max_lines` (180) `KeyedSessionMemory`
believed it had, so `len(history)` could never exceed 100. `maybe_update` fired
on `len(history) - lines_seen >= every_lines` and then ratcheted
`lines_seen = len(history)`. At `lines_seen=90` against a pinned 100 that is
`10 >= 30`, evaluated on every message, forever. Simulated against the same
300-message slide, the old trigger fires **once** and never again.

Two fixes, because either alone leaves it fragile. `SessionMemory` now takes
`max_messages` and `KeyedSessionMemory` passes its own `max_lines`, so the two
caps cannot disagree again. And the trigger no longer counts: `summary.new_lines_since`
locates the last **three summarised lines** by content (one line collides in a
chat full of "Fds"; three do not) and returns everything after them. A window
that has moved on entirely yields all of it — which is correct, and is also what
heals a state file written by the old counter, so no manual repair was needed.
`lines_seen` is still written, for whoever opens the file, and is read by nothing.

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
- **Its GPU pinning mirrors prod** — one card, the prod one, exposed as index 0.
  The lesson outlived the two-card era: a rig arranged differently from
  production tests something nobody ships. Arranging it the other way once left
  FLUX ~19GB instead of 23.5GB and every *edit* OOMed while generation still
  worked, which reads as "edits are broken" when it is really "the rig is wrong".
- **Message ids are unique per run** (`uuid4` prefix, not a counter). The sim
  container outlives a run, so a restarting counter made the adapter's replay
  guard — which is correct — treat the second run as the first run's backlog and
  ignore every message.

Presets: `smoke` (3 people, no image rendering, ~40s), `standard` (4 people, the
full feature surface, ~10 min), `long_haul` (5 people, overflows the retrieval
budget and the session window, ~28 min). Reports land in `reports/sim/<stamp>/`
with an `index.html` contact sheet; the run exits non-zero on any failed
assertion, so it can gate a deploy. Personas cost a few cents to ~€1 per run.

`KAYA_SIM_LLAMA_URL` picks which llama-server answers. Unset (the default) it
falls through to `inference.gguf.server_url`, i.e. **the one prod is using** —
right for a run meant to mirror production, wrong when the point is to leave the
live model alone. `http://llama-bench:8080` drives the bench server instead; that
one has no `--mmproj`, so any preset involving photos needs the prod one.

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
