# GitHub Copilot Instructions

## Project Overview
KayaChatBot is an AI assistant bot for a Portuguese friend group chat called **Kaya**. The bot is NOT a group member — it is an assistant with access to the group's collective memory. It has long-term memory of facts, events, and people learned from real WhatsApp and Instagram conversation history (via RAG + fine-tuning). It communicates in **European Portuguese or English**; it does NOT need to use the group's specific slang or lingo. The focus is on natural language ability and factual memory, not mimicking any particular speech style.

**Key architecture decisions:**
- RAG is **always on** — every message (casual or Q&A) retrieves context from conversation history and the curated knowledge base. The model never answers from fine-tune memory alone.
- Group member knowledge is stored in `data/group_members.json` (injected into system prompt) and `data/group_knowledge.json` (embedded into ChromaDB `kaya_knowledge_base` collection).
- The `rag.knowledge_approach` config toggle (`both` / `json_only` / `chromadb_only` / `none`) enables benchmarking different knowledge injection strategies.

## Environment Setup
- Always run code using the virtual environment named 'kaya_chatbot' located in the `kaya_chatbot_env/` directory
- Ensure the virtual environment is activated before executing any Python scripts or commands
- Always install Python packages within the 'kaya_chatbot' virtual environment
- **Prefer using Python executable directly** (e.g., `python script.py`) always inside virtual environment

## Coding Preferences
- Avoid creating backup and temporary code files when rewriting existing ones; either replace the existing file or create a new one and delete the old one

 - Branching & PRs in Plans: Whenever you're asked to create a plan, include explicit steps to:
	 - create a new Git branch for the work,
	 - open a pull request (PR) for that branch,
	 - run tests and verify the change (including running the project in Docker where applicable),
	 - iterate until the implementation is well tested and well implemented,
	 - merge the PR after tests pass and approvals are obtained.

## Build, Test & Validation Commands
- **Install dependencies**: `pip install -r requirements.txt`
- **Run all tests**: `python -m pytest tests/ -v`
- **Run RAG tests only**: `python -m pytest tests/rag/ -v`
- **Run pipeline tests only**: `python -m pytest tests/pipeline/ -v`
- **Validate pipeline outputs**: `python tests/pipeline/validate_pipeline.py`
- **Build & run in Docker**: `docker-compose up --build`
- **Run bot locally**: `python src/chat/chat.py`
- **Run full pipeline**: `python run_full_pipeline.py`
- **Build vector DB**: `python src/data/build_vector_db.py`

## Key File Locations
- **Central config**: `config.yaml` (local), `config.docker.yaml` (Docker overrides)
- **Config loader**: `src/config_loader.py` — `load_config(path, profile_override=None)` resolves the active model profile; all config consumers must use this
- **Model profiles**: `configs/models/*.yaml` — per-model preset files; also declared in `config.yaml` under `model_profiles`
- **RAG retriever**: `src/chat/retriever.py` — semantic search over ChromaDB
- **Inference engine**: `src/chat/inference.py` — model loading and generation
- **Chat loop**: `src/chat/chat.py` — interactive chat with always-on RAG
- **Data pipeline**: `src/data/` — extraction, synthetic generation, knowledge base, vector DB
- **LLM providers**: `src/llm_providers/` — Azure OpenAI, xAI Grok abstractions (follow `base.py` interface)
- **Pydantic models**: `src/models.py` — type-safe data structures
- **Member profiles**: `data/group_members.json` (injected into system prompt)
- **Curated knowledge**: `data/group_knowledge.json` (embedded into ChromaDB `kaya_knowledge_base`)
- **ChromaDB storage**: `data/rag_db/` — persistent vector DB with `kaya_conversations` and `kaya_knowledge_base` collections
- **Tests**: `tests/rag/`, `tests/pipeline/`, `tests/test_inference.py`

## Docker Usage
- When requested and building images, make sure to erase previously built images, containers, volumes, or builds to prevent storage overload (e.g. using `docker system prune` or similar)
- After completing any change, always test it inside Docker to verify it works correctly in the containerized environment

## Custom Agents & Automation
- **Agent profiles**: `.github/agents/` — specialized agent configurations:
  - `bug-fixer` — root cause analysis, minimal fixes, regression tests; also handles auto-created GPU pipeline failure issues
  - `feature-dev` — new features following existing patterns
  - `test-specialist` — test coverage improvements, never modifies production code
  - `model-trainer` — fine-tuning config, LoRA settings, data pipeline improvements
- **Task intake**: `tasks.json` — JSON file for submitting bugs/features; automatically creates GitHub Issues via `.github/workflows/create-issues-from-tasks.yml`
- When working on a task, always check which custom agent profile applies based on the issue labels

## Self-Healing GPU Pipeline Cycle

The project has an automated failure→fix→retry loop:

```
GPU Pipeline fails
      ↓
gpu-failure-handler.yml triggers automatically
      ↓
Downloads log artifact, extracts error details
      ↓
Creates GitHub Issue: "[Auto] GPU Pipeline failed: {mode}"
  Labels: bug, priority:high, agent:bug-fixer
  Assignee: copilot
      ↓
Copilot bug-fixer agent picks up the issue
  • Reads log excerpt from issue body
  • Applies minimal fix (Dockerfile, requirements.txt, config, or source)
  • Runs pytest (no GPU needed for unit tests)
  • Opens PR: "fix/gpu-pipeline-{issue_number}"
  • PR description includes "Fixes #{issue_number}"
      ↓
PR triggers GPU Pipeline automatically (touches src/**, Dockerfile, etc.)
      ↓
GPU Pipeline succeeds → results posted to PR comment
      ↓
Human reviews and approves → merge → issue auto-closes
```

**Loop guard**: The failure handler refuses to create new auto-issues when 3+ are already open, preventing infinite loops.

**All agents**: When opening fix PRs, always include `Fixes #N` in the description, never self-merge, keep a clean commit history, and ensure tests pass before requesting review.

## Hardware Constraints
- **GPU**: NVIDIA RTX 3090 — 24 GB VRAM (self-hosted runner)
- **System RAM**: 32 GB
- **CUDA**: 12.4
- **VRAM budget per model profile** (training, 4-bit + LoRA):
  - `qwen3-14b`: ~15 GB (safe)
  - `gemma4-e4b`: ~11 GB (comfortable headroom)
- Always leave at least 2 GB VRAM headroom to avoid OOM during training.
- If a training job OOMs: reduce `lora_r` to 8 and/or `max_seq_length` to 2048 as fallback.

## Model Profiles
- Model profiles live in `configs/models/*.yaml` and are also declared in `config.yaml` under `model_profiles`.
- Switch the active model by changing `active_model_profile` in `config.yaml` (or passing `--profile <name>` on the CLI).
- Config is resolved via `src/config_loader.py` — `load_config(path, profile_override=None)` deep-merges the active profile into `model:` and `training:` sections. All config consumers must use this loader.
- **Available profiles**:
  - `qwen3-14b` — `unsloth/Qwen3-14B-bnb-4bit`, seq 4096, lora_r 32, output `models/kaya_qwen3_14b`
  - `gemma4-e4b` — `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` (8B), seq 4096, lora_r 16, output `models/kaya_gemma4_e4b`
- **Gemma 4 notes**: uses `<|turn>` chat template tokens and supports `<|think|>` thinking mode. Thinking mode must be **disabled** for SFT (do not add `<|think|>` to the system prompt during training). Unsloth handles the chat template via `tokenizer.apply_chat_template()` automatically when using `FastLanguageModel`.
- **Excluded**: Gemma 4 26B-A4B (MoE) — Unsloth explicitly warns against QLoRA for MoE models; bf16 LoRA would require ~28 GB VRAM, exceeding the RTX 3090 budget.

## GPU Constraints
- The Copilot coding agent runs on GitHub-hosted runners — **no GPU available**.
- For model-related tasks (fine-tuning, inference testing, evaluation), the agent modifies code/config **only**. It must never attempt to run training commands (`python src/finetuning/train.py`, `docker-compose up`, etc.).
- GPU execution is handled automatically by `.github/workflows/gpu-pipeline.yml`, which runs on the **self-hosted runner** (RTX 3090, runner ID #2) when a PR touches training-related files.
- **Copilot cloud mode**: VS Code cloud agents run on GitHub-hosted runners (no GPU), but they create PRs that automatically trigger `gpu-pipeline.yml` on the self-hosted RTX 3090 runner. This enables a fully autonomous issue→PR→train→benchmark loop without VS Code being open.
- Training modes: `finetune` (default for PRs, 240 min timeout), `full-pipeline` (240 min), `evaluate` (10 min), `inference-test` (10 min), `benchmark` (60 min).
- Training results (loss, duration, steps) are posted back to the PR as a comment by the workflow.
- **PR trigger paths**: Any PR touching `src/**`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `config.docker.yaml`, `data/*.jsonl`, `run_full_pipeline.py`, or `.github/scripts/**` automatically triggers the GPU Pipeline on the self-hosted runner.
