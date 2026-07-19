# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

KayaChatBot is a private AI assistant for the "Kaya" Portuguese friend group. It maintains long-term memory of group facts and events derived from WhatsApp history and answers in **European Portuguese or English**. It is **not** a group member — it is a bot with access to the group's collective memory.

**Core invariant**: RAG is always-on. The model must never answer from fine-tuned weights alone — every message retrieves context first.

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

# Docker (always rebuild+prune after changes)
docker-compose up --build
docker system prune  # prevent storage overload

# Dev/Test (Docker)
docker compose --profile dev up -d kaya-dev       # dev web UI on :7861, ./src mounted read-write (or use scripts/app_up.sh dev)
docker compose --profile test run --rm kaya-test  # run the pytest suite in-container

# Deployment (see DEPLOYMENT.md)
scripts/deploy_prod.sh [ref]    # make a commit LIVE: updates ~/kaya-prod + restarts prod (CI's Deploy (prod) calls this)
scripts/app_up.sh dev|prod      # manually power up an env + Cloudflare Tunnel (one GPU → one env at a time)
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
    → chat.py (loads adapter + RAG at runtime)
```

### RAG System (`src/chat/retriever.py`)

Two knowledge sources are injected at inference time, controlled by `rag.knowledge_approach` in `config.yaml`:

| Approach | What's injected |
|---|---|
| `json_only` | `group_members.json` profiles → system prompt (best benchmark score) |
| `chromadb_only` | Semantic search over `kaya_knowledge_base` ChromaDB collection |
| `both` | Both of the above |
| `none` | Baseline — conversation history only |

`ConversationRetriever` uses BAAI/bge-m3 embeddings against the `kaya_conversations` ChromaDB collection. `extract_query_persons()` detects named group members in the query and post-filters retrieval by `participants`/`mentioned` metadata. `retrieve_all()` enforces `rag.max_context_tokens` by truncating lowest-priority context (conversation chunks first, then knowledge, then recent summaries). Token estimation is whitespace-based (`words / 0.60`, tuned for Portuguese subword inflation).

**Date-aware facts (mixed rule).** Knowledge facts carry optional date metadata: `event_date_hint` (an explicit temporal phrase pulled from the source text), `source_date_start`/`source_date_end` (the timestamp range of the source messages), and `last_updated`. These are populated by `generate_knowledge_base.py` and embedded into ChromaDB metadata by `build_vector_db.py`. The retriever only surfaces dates when `_has_temporal_intent(query)` matches a timing question (PT/EN keywords); otherwise normal answers stay date-free. When surfacing, an explicit `event_date_hint` wins over the message timestamps (relative age rendered by `_relative_age`). `chat.py`/`web_app.py` also append `Hoje é <date>.` to the runtime system prompt so the model can reason about recency.

**Follow-up suggestions (web UI only).** After each answer, `src/chat/suggestions.py` prompts the already-loaded local model a second time for 2-3 follow-up questions, shown as clickable chips in the Gradio UI (`web_app.py`). Controlled by `chat.suggestions` in `config.yaml`; degrades to no chips on any failure.

### Config System (`src/config_loader.py`)

Single entry point: `load_config(path, profile_override=None)`. Profiles (defined under `model_profiles` in `config.yaml`) deep-merge into the top-level `model:` and `training:` sections. The active profile is set by `active_model_profile` in `config.yaml` or passed via `--profile` CLI flag. **All code paths must use `load_config()` — never read `config.yaml` directly.**

### LLM Providers (`src/llm_providers/`)

Unified `LLMProvider` interface with `_retry_with_backoff()` for rate-limit resilience. Azure OpenAI (`azure_provider.py`) and xAI Grok (`xai_provider.py`); switch via `generation.provider` in `config.yaml`. **Eval-judge + web-search only** — never send group data to these providers; knowledge extraction and synthetic generation use the local teacher (`src/data/local_teacher.py`).

### Fine-tuning (`src/finetuning/train.py`)

Uses Unsloth (`FastModel` / `FastLanguageModel`) for Gemma4 and Qwen3. Training calls `SFTTrainer` directly (no wrapper class — a previous `KayaTrainer` wrapper caused 20+ GB RAM spikes). LoRA adapters are saved to `training.output_dir`. Inference expects `adapter_config.json` in the model directory.

### Deployment (`DEPLOYMENT.md`)

`kaya-prod` is the **always-on** production web app. The box is **serving-only** (fine-tuning is done separately). Access is via a **Cloudflare Tunnel** (`cloudflared` compose service, `tunnel` profile) with two protection layers: Cloudflare Access (network login) and the Gradio username/password (`KAYA_WEB_USER`/`KAYA_WEB_PASS`, read from env in `web_app.py`, overriding `chat.web_auth`). The UI header shows the running env + commit (`KAYA_ENV`/`KAYA_VERSION`).

**Prod runs from its own checkout** at `~/kaya-prod` (separate from this dev copy), with `models/` and `data/` symlinked to the shared originals — so you can develop here without touching the live site. `kaya-prod` has `restart: unless-stopped`, so with Docker enabled on boot (`sudo systemctl enable docker`) the site **auto-recovers after a reboot**.

**Push to prod:** `scripts/deploy_prod.sh [ref]` checks out the ref in `~/kaya-prod`, rebuilds, and restarts the live container — that is what makes a commit live. CI/CD on a **self-hosted GPU runner**: `ci.yml` tests every PR; `validate-main.yml` rebuilds + tests on merge to `main` (no container start); `deploy-prod.yml` (manual, `prod` Environment requires reviewer approval) calls `deploy_prod.sh` to update the live site. `kaya-dev` (port 7861) is for occasional manual dev runs only and shares the single GPU with prod (run one at a time). Full runbook in `DEPLOYMENT.md`.

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

To avoid OOM on RTX 3090 (24 GB VRAM):

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
