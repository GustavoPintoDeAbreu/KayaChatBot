# GitHub Copilot Instructions

## Project Overview
KayaChatBot is an AI assistant bot for a Portuguese friend group chat called **Kaya**. The bot is NOT a group member — it is an assistant with access to the group's collective memory. It has long-term memory of facts, events, and people learned from real WhatsApp and Instagram conversation history (via RAG + fine-tuning). It communicates in **European Portuguese only** (never Brazilian Portuguese, never emojis). Users can prefix any message with `/en` to receive a one-off English reply. The focus is on natural language ability and factual memory, not mimicking any particular speech style.

**Key architecture decisions:**
- RAG is **always on** — every message (casual or Q&A) retrieves context from conversation history and the curated knowledge base. The model never answers from fine-tune memory alone.
- Group member knowledge is stored in `data/group_members.json` (injected into system prompt) and `data/group_knowledge.json` (embedded into ChromaDB `kaya_knowledge_base` collection).
- The `rag.knowledge_approach` config toggle (`both` / `json_only` / `chromadb_only` / `none`) enables benchmarking different knowledge injection strategies.
- **Language policy**: Always European Portuguese. No emojis. No Brazilian Portuguese. `/en <message>` prefix triggers English for that turn only.
- **Knowledge generation** uses xAI Grok (configured via `generation.provider: "xai"` in config.yaml).

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
- **RAG retriever**: `src/chat/retriever.py` — semantic search over ChromaDB
- **Inference engine**: `src/chat/inference.py` — model loading and generation
- **Chat loop**: `src/chat/chat.py` — interactive chat with always-on RAG
- **Data pipeline**: `src/data/` — extraction, synthetic generation, knowledge base, vector DB
- **LLM providers**: `src/llm_providers/` — Azure OpenAI, xAI Grok abstractions (follow `base.py` interface)
- **Pydantic models**: `src/models.py` — type-safe data structures
- **Member profiles**: `data/group_members.json` (injected into system prompt)
- **Curated knowledge**: `data/group_knowledge.json` (embedded into ChromaDB `kaya_knowledge_base`)
- **ChromaDB storage**: `data/rag_db/` — persistent vector DB with `kaya_conversations` and `kaya_knowledge_base` collections
- **Language filters**: `src/data/language_filters.py` — BR→PT-EU substitution, emoji removal
- **Language validator**: `src/testing/language_validator.py` — response validation for PT-EU compliance
- **Tests**: `tests/rag/`, `tests/pipeline/`, `tests/test_inference.py`

## Docker Usage
- When requested and building images, make sure to erase previously built images, containers, volumes, or builds to prevent storage overload (e.g. using `docker system prune` or similar)
- After completing any change, always test it inside Docker to verify it works correctly in the containerized environment

## Custom Agents & Automation
- **Agent profiles**: `.github/agents/` — specialized agent configurations:
  - `brainstormer` — architecture planning, system design, trade-off analysis (Claude Opus 4.6; produces plans only, never writes code)
  - `bug-fixer` — root cause analysis, minimal fixes, regression tests (Claude Sonnet 4.6)
  - `feature-dev` — new features following existing patterns (Claude Sonnet 4.6)
  - `test-specialist` — test coverage improvements, never modifies production code (Claude Haiku 4.5)
  - `model-trainer` — fine-tuning config, LoRA settings, data pipeline improvements (Claude Sonnet 4.6)
- **Task intake**: `tasks.json` — JSON file for submitting bugs/features; automatically creates GitHub Issues via `.github/workflows/create-issues-from-tasks.yml`
- When working on a task, always check which custom agent profile applies based on the issue labels
- **Brainstormer workflow**: For complex multi-phase changes, invoke the `brainstormer` agent first to produce a plan, then assign implementation phases to the appropriate specialized agents.

## GPU Constraints
- The Copilot coding agent runs on GitHub-hosted runners — **no GPU available**.
- For model-related tasks (fine-tuning, inference testing, evaluation), the agent modifies code/config **only**. It must never attempt to run training commands (`python src/finetuning/train.py`, `docker-compose up`, etc.).
- GPU execution is handled automatically by `.github/workflows/gpu-pipeline.yml`, which runs on the **self-hosted runner** (user's local GPU machine) when a PR touches training-related files.
- Training modes: `finetune` (default for PRs, 240 min timeout), `full-pipeline` (240 min), `evaluate` (10 min), `inference-test` (10 min).
- Training results (loss, duration, steps) are posted back to the PR as a comment by the workflow.
