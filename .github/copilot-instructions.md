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

## GPU & Docker Execution

The project uses a **self-hosted GitHub Actions runner** on the user's local GPU machine. How you access it depends on context:

### Coding agent (GitHub Actions runner)
Never run `docker-compose`, `python train.py`, or any heavy command directly — the coding agent runs on a GitHub-hosted runner with **no GPU**. Use the dispatch helper script to send jobs to the self-hosted runner:

```bash
bash .github/scripts/trigger-gpu-pipeline.sh <mode> [--wait]
```

Available modes:

| Mode | Timeout | Use `--wait`? | Purpose |
|------|---------|---------------|---------|
| `finetune` | 240 min | No | LoRA fine-tuning on Qwen3-14B |
| `full-pipeline` | 240 min | No | Full data + training pipeline |
| `evaluate` | 10 min | Yes | Full pytest suite in Docker |
| `inference-test` | 10 min | Yes | Test model inference |
| `generate-knowledge` | 30 min | Yes | Regenerate group_knowledge.json via xAI Grok |
| `build-vectordb` | 15 min | Yes | Rebuild ChromaDB vector database |
| `benchmark` | 60 min | No | Conversation benchmark |

- **Short modes** (`evaluate`, `inference-test`, `generate-knowledge`, `build-vectordb`): use `--wait` to get results inline.
- **Long modes** (`finetune`, `full-pipeline`, `benchmark`): dispatch without `--wait` — the script prints the Actions run URL for tracking.
- PRs that touch `src/finetuning/**`, `config.docker.yaml`, or training data files **auto-trigger** the GPU pipeline (`finetune` mode) and post results as a PR comment.

### VS Code Copilot Chat (local)
When working interactively in VS Code on the local machine, run `docker-compose` commands directly:

```bash
docker system prune -f                              # clean up before building
docker-compose build                               # build image
docker-compose run --rm kaya-chatbot python -m pytest tests/ -v  # run tests
docker-compose run --rm kaya-chatbot python src/chat/chat.py     # interactive chat
```

Always clean up after use to prevent storage overload (`docker system prune` or `docker-compose down --rmi local --volumes`).
