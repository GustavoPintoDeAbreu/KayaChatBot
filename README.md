# KayaChatBot

An AI assistant bot for the **Kaya** Portuguese friend group chat, trained on real WhatsApp and Instagram conversations using Qwen3-14B with LoRA.

## 🎯 Overview

KayaChatBot is the AI memory of the Kaya group. It is **not** a group member — it is an assistant with access to the group's collective memory. It learns facts, events, and relationships from the group's conversation history so it can answer questions like "what did we talk about at the beach trip?" or just have a casual chat. It communicates naturally in **European Portuguese or English**.

**Key Features:**
- Extracts and cleans messages from WhatsApp exports and Instagram JSON
- Generates synthetic multi-turn training conversations using xAI Grok or Azure OpenAI GPT-4.1-mini
- **Always-on RAG**: Retrieves relevant context for every message (not just detected questions)
- **Dual knowledge system**: JSON member profiles injected into the system prompt + curated ChromaDB knowledge base
- **Automated knowledge generation**: Uses Azure GPT-4.1-mini to extract biographical facts from chat history
- **Benchmarking toggle**: Switch between `both` / `json_only` / `chromadb_only` / `none` knowledge approaches
- Fine-tunes Qwen3-14B using LoRA (Low-Rank Adaptation) with 4-bit quantization
- Efficient training on consumer GPUs (requires ~16GB VRAM)

## 🤖 RAG & Knowledge System

### Always-On Retrieval
RAG is enabled for every message. The bot never answers from fine-tune memory alone — it always retrieves context first.

### Dual Knowledge Sources
| Source | File | How it's used |
|---|---|---|
| Member profiles | `data/group_members.json` | Injected directly into the system prompt |
| Curated facts | `data/group_knowledge.json` | Embedded into ChromaDB `kaya_knowledge_base` collection |
| Conversation history | `data/rag_db/` (ChromaDB) | Semantic search over `kaya_conversations` collection |

### Knowledge Approach Toggle
Control which knowledge sources are active in `config.yaml`:

```yaml
rag:
  knowledge_approach: "both"   # best coverage
  # Options:
  #   "both"          — JSON members in system prompt AND ChromaDB KB retrieval
  #   "json_only"     — JSON injection only, no KB retrieval
  #   "chromadb_only" — ChromaDB KB only, no JSON injection
  #   "none"          — Baseline: conversation RAG + fine-tune only
```

### Smart Context Retrieval
- Uses BAAI/bge-m3 multilingual embeddings
- Person-aware filtering: queries mentioning "Peter" retrieve Peter's messages
- Semantic search across conversation chunks
- Real-time retrieval stats during chat

### Example Usage
```
User: What do you know about Peter?
📚 Retrieved 3 conversation chunks + 1 knowledge fact
Kaya Bot: Peter is a member of the Kaya group. He enjoys music and...

User: olá pessoal
📚 Retrieved 3 conversation chunks
Kaya Bot: oi! tudo bem? 😊
```

## 📁 Project Structure

```
KayaChatBot/
├── src/
│   ├── data/                         # Data processing & generation
│   │   ├── extract_all_messages.py   # WhatsApp + Instagram parser
│   │   ├── generate_synthetic_data.py # LLM synthetic conversation generation
│   │   ├── generate_knowledge_base.py # LLM biographical fact extraction (Azure)
│   │   ├── build_vector_db.py        # Build ChromaDB collections
│   │   ├── prepare_portuguese_data.py
│   │   ├── merge_datasets.py
│   │   ├── format_direct_training.py
│   │   └── readers.py
│   ├── finetuning/                   # Model training
│   │   ├── train.py
│   │   └── trainer.py
│   ├── chat/                         # Inference & interaction
│   │   ├── chat.py                   # Interactive chat loop (always-on RAG)
│   │   ├── inference.py
│   │   └── retriever.py              # RAG retrieval (conversations + KB)
│   ├── llm_providers/                # LLM provider abstractions
│   │   ├── azure_provider.py
│   │   ├── xai_provider.py
│   │   └── base.py
│   └── models.py
├── data/
│   ├── group_members.json            # Member profiles (system prompt injection)
│   ├── group_knowledge.json          # Curated facts (ChromaDB KB source)
│   ├── all_messages_cleaned.jsonl    # Cleaned message history
│   ├── rag_db/                       # ChromaDB persistent storage
│   └── wpp/                          # Raw WhatsApp exports
├── models/                           # Trained LoRA adapters (gitignored)
├── config.yaml                       # Central configuration
├── config.docker.yaml                # Docker-specific config overrides
├── run_full_pipeline.py              # Pipeline orchestrator
├── Dockerfile
├── docker-compose.yml
└── .env                              # API keys (gitignored)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU with 16GB+ VRAM (for training; less for inference)
- Azure OpenAI API access (for knowledge generation and optional synthetic generation)
- xAI API access (for synthetic data generation with Grok models)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd KayaChatBot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv kaya_chatbot_env
   source kaya_chatbot_env/bin/activate   # Linux/Mac
   # kaya_chatbot_env\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up credentials**
   ```bash
   # Create .env with your API keys:
   echo 'AZURE_OPENAI_API_KEY_gpt_41_mini=your_key_here' >> .env
   echo 'XAI_API_KEY=your_key_here' >> .env
   ```

### Data Preparation

1. **Add your chat data**
   - WhatsApp: Export chat as TXT → `data/wpp/`
   - Instagram: Download JSON messages → `data/insta/`

2. **Extract and clean messages**
   ```bash
   kaya_chatbot_env/bin/python src/data/extract_all_messages.py
   ```

3. **Generate knowledge base from chat history (recommended)**
   ```bash
   # Test with 3 chunks first
   kaya_chatbot_env/bin/python src/data/generate_knowledge_base.py --test

   # Full run (processes all ~7 chunks of 2000 tokens each)
   kaya_chatbot_env/bin/python src/data/generate_knowledge_base.py

   # Resume after interruption
   kaya_chatbot_env/bin/python src/data/generate_knowledge_base.py --resume-from 5
   ```
   This populates `data/group_members.json` (notes field) and `data/group_knowledge.json` (text field).

4. **Build the ChromaDB vector database**
   ```bash
   kaya_chatbot_env/bin/python src/data/build_vector_db.py
   ```
   This builds two collections: `kaya_conversations` (chat history) and `kaya_knowledge_base` (curated facts).

### Training

```bash
# Skip synthetic generation, train directly from messages
# (ensure pipeline.skip_synthetic: true in config.yaml)
kaya_chatbot_env/bin/python run_full_pipeline.py

# Or train step by step:
kaya_chatbot_env/bin/python src/data/format_direct_training.py
kaya_chatbot_env/bin/python src/data/merge_datasets.py
kaya_chatbot_env/bin/python src/finetuning/train.py
```

### Chat with Your Bot

```bash
kaya_chatbot_env/bin/python src/chat/chat.py
```

The `knowledge_approach` in `config.yaml` controls what knowledge is injected:
- `"both"` — JSON profiles + ChromaDB KB (recommended)
- `"json_only"` — JSON profiles only
- `"chromadb_only"` — ChromaDB KB only
- `"none"` — baseline (conversation history only)

## 📊 Pipeline Stages

### 1. **Message Extraction** (`extract_all_messages.py`)
- Reads WhatsApp TXT and Instagram JSON files
- Cleans and standardizes messages (removes URLs, media, system messages)
- Merges consecutive messages from the same sender

**Output:** 
- `data/all_messages_cleaned.jsonl` — all cleaned messages
- `data/finetune_chunks.jsonl` — chunked messages for generation

### 1b. **Knowledge Base Generation** (`generate_knowledge_base.py`) *(optional, recommended)*
- Iterates over the cleaned message history in ~2000-token chunks
- Calls Azure GPT-4.1-mini to extract biographical facts per member
- Merges facts into member profiles and curated knowledge entries
- Checkpoints every 5 chunks; resumable with `--resume-from N`

**Output (updated):** `data/group_members.json`, `data/group_knowledge.json`

### 2. **Synthetic Data Generation** (`generate_synthetic_data.py`) *(skip_synthetic: false)*
- Uses xAI Grok or Azure OpenAI GPT-4.1-mini to generate diverse Q&A conversations
- Requires `pipeline.skip_synthetic: false` in `config.yaml`

**Output:** `data/synthetic_kaya.jsonl`

### 2 (direct). **Direct Training Format** (`format_direct_training.py`) *(skip_synthetic: true)*
- Formats raw messages into training pairs without any API calls
- Context blocks use `=== Conversas relevantes do grupo ===` markers (matches inference format)

**Output:** `data/direct_training.jsonl`

### 3. **Build Vector Database** (`build_vector_db.py`)
- Builds `kaya_conversations` ChromaDB collection from message chunks
- Builds `kaya_knowledge_base` ChromaDB collection from `group_knowledge.json`
- Uses BAAI/bge-m3 embeddings

**Output:** `data/rag_db/` (ChromaDB)

### 4. **Dataset Merging** (`merge_datasets.py`)
- Combines datasets and applies Qwen3 ChatML template
- 90/10 train/val split

**Output:** `data/train_synthetic.jsonl`, `data/val_synthetic.jsonl`

### 5. **Fine-Tuning** (`train.py`)
- Loads Qwen3-14B with 4-bit quantization (unsloth)
- LoRA adapters: rank=32, alpha=32
- 1500 steps with linear learning rate schedule

**Output:** `models/kaya_v2_synthetic/`

## ⚙️ Configuration

All settings live in [config.yaml](config.yaml). Key sections:

### Pipeline Mode

```yaml
pipeline:
  skip_synthetic: true   # true = direct format (no API), false = synthetic generation
  generate_knowledge: false  # true = run knowledge extraction via Azure (before training)
```

### Knowledge Approach (Benchmarking)

```yaml
rag:
  knowledge_approach: "both"
  # "both"          — JSON members in system prompt + ChromaDB KB retrieval
  # "json_only"     — JSON injection only
  # "chromadb_only" — ChromaDB KB retrieval only
  # "none"          — Baseline (conversation history only)
```

### Test Mode

```yaml
test_mode:
  enabled: true   # Fast validation run (few steps, small data)
```

### Model & Training

```yaml
model:
  model_id: "unsloth/Qwen3-14B-bnb-4bit"
  max_seq_length: 4096

training:
  output_dir: "./models/kaya_v2_synthetic"
  max_steps: 1500
  lora_r: 32
  lora_alpha: 32
  learning_rate: 0.0001
```

## 🧪 Testing

### Test Knowledge Generation
```bash
kaya_chatbot_env/bin/python src/data/generate_knowledge_base.py --test
```
Processes 3 message chunks and shows extracted bios without running the full set.

### Test RAG Retrieval
```bash
# After building the vector DB:
kaya_chatbot_env/bin/python src/data/build_vector_db.py
# Then start chat:
kaya_chatbot_env/bin/python src/chat/chat.py
```

### Run Test Pipeline
```bash
# Set test_mode.enabled: true in config.yaml first
kaya_chatbot_env/bin/python run_full_pipeline.py
```

### Validate Pipeline Outputs
```bash
kaya_chatbot_env/bin/python tests/pipeline/validate_pipeline.py
```

### Test Azure Connection
```bash
kaya_chatbot_env/bin/python src/testing/test_azure.py
```

## 💡 Tips & Best Practices

### Knowledge Base Quality
- Run `generate_knowledge_base.py` after any significant addition of new messages
- Review `data/group_members.json` bios manually and edit them for accuracy
- The more diverse the chat data, the richer the extracted biographies

### Rate Limiting
- Azure GPT-4.1-mini has rate limits — the knowledge generator has a 2-second delay between calls
- Use `--resume-from N` to resume if the script is interrupted

### RAG Benchmarking
- Set `knowledge_approach: "json_only"` for simplest setup (no vector KB needed)
- Set `knowledge_approach: "both"` for best coverage
- Set `knowledge_approach: "none"` to measure baseline performance without any knowledge injection

### Training
- Monitor GPU with `nvidia-smi`
- Reduce `per_device_train_batch_size` if OOM errors occur
- Training loss should decrease steadily

### Inference
- First load takes ~1 minute (model initialization)
- Subsequent responses: ~2-3 seconds
- Adjust `inference.temperature` in `config.yaml` for response creativity

## 📦 Pydantic Models

The codebase uses Pydantic models for type safety (see [src/models.py](src/models.py)):

- `WhatsAppMessage` — Raw WhatsApp TXT message
- `InstagramMessage` — Raw Instagram JSON message
- `CleanedMessage` — Standardized message format
- `FinetuneChunk` — Chunked messages for generation
- `SyntheticConversation` — Generated Q&A pairs
- `TrainingExample` — Formatted training instance

## 🐳 Docker Support

Docker configuration is available (`Dockerfile`, `docker-compose.yml`). See `config.docker.yaml` for Docker-specific settings.

```bash
docker-compose up --build
```

## 🔒 Security

- Never commit `.env` or `credentials.txt` (they're in `.gitignore`)
- Regenerate API keys if accidentally exposed
- Keep chat data private (data/ folder is gitignored)

## 🛠️ Troubleshooting

### Import Errors
```bash
# Ensure virtual environment is activated
# Add project root to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:/path/to/KayaChatBot"
```

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` in config.yaml
- Reduce `max_seq_length` (but this affects context)
- Use gradient checkpointing (already enabled)

### Azure Rate Limit Errors
- Wait 60 seconds between generation runs
- Check Azure portal quota limits
- Consider upgrading to higher TPM tier

### Model Not Loading
- Ensure models are downloaded to `kaya_chatbot_env/`
- Check disk space (models are ~5GB each)
- Verify HuggingFace access token if using gated models

## 📈 Expected Results

With ~20K messages and 2000+ synthetic conversations:
- Training converges in ~3 epochs
- Model learns facts, events, and relationships from shared conversation history
- Responses feel grounded in real group memories
- Communicates naturally in European Portuguese and English

## � Copilot Coding Agent Team

This project uses **GitHub Copilot coding agent** with custom agent profiles and a JSON-based task intake system. You write bugs and features in a JSON file, push it, and Copilot autonomously creates PRs.

### How It Works

```
tasks.json → GitHub Action → GitHub Issues → Copilot Coding Agent → Pull Requests
```

1. Add tasks to `tasks.json` (see templates below)
2. Push to `main`
3. A GitHub Action creates labeled issues and assigns them to Copilot
4. Copilot picks up each issue, works in its own environment, runs tests, and opens a PR
5. You review and merge the PR

### Custom Agent Profiles

| Agent | File | Specialization |
|-------|------|----------------|
| `bug-fixer` | `.github/agents/bug-fixer.agent.md` | Root cause analysis, minimal fixes, regression tests |
| `feature-dev` | `.github/agents/feature-dev.agent.md` | New features following existing patterns, with tests |
| `test-specialist` | `.github/agents/test-specialist.agent.md` | Test coverage improvements, never modifies production code |

### Task Templates for `tasks.json`

The `tasks.json` file at the repo root accepts an array of task objects. After pushing, the GitHub Action processes them into issues and clears the file.

#### Bug Report Template

```json
[
  {
    "title": "Fix: <short description of the bug>",
    "type": "bug",
    "priority": "high",
    "description": "**What happens:** <describe the incorrect behavior>\n\n**Expected:** <describe what should happen>\n\n**Steps to reproduce:**\n1. <step 1>\n2. <step 2>\n\n**Error message (if any):**\n```\n<paste error here>\n```",
    "files_hint": ["src/chat/retriever.py"],
    "agent": "bug-fixer"
  }
]
```

#### Feature Request Template

```json
[
  {
    "title": "Feature: <short description>",
    "type": "feature",
    "priority": "medium",
    "description": "**Goal:** <what should the feature do?>\n\n**Details:**\n- <requirement 1>\n- <requirement 2>\n\n**Acceptance criteria:**\n- [ ] <criterion 1>\n- [ ] <criterion 2>",
    "files_hint": ["src/chat/chat.py", "config.yaml"],
    "agent": "feature-dev"
  }
]
```

#### Improvement Template

```json
[
  {
    "title": "Improvement: <short description>",
    "type": "improvement",
    "priority": "low",
    "description": "**Current behavior:** <what exists today>\n\n**Proposed improvement:** <what should change>\n\n**Motivation:** <why this matters>",
    "files_hint": [],
    "agent": "feature-dev"
  }
]
```

#### Test Coverage Template

```json
[
  {
    "title": "Test: <what needs test coverage>",
    "type": "test",
    "priority": "medium",
    "description": "**Module to test:** `src/chat/retriever.py`\n\n**What to cover:**\n- <scenario 1>\n- <scenario 2>\n- Edge cases: <describe>",
    "files_hint": ["src/chat/retriever.py"],
    "agent": "test-specialist"
  }
]
```

#### Multiple Tasks Example

```json
[
  {
    "title": "Fix: RAG retriever returns empty results for short queries",
    "type": "bug",
    "priority": "high",
    "description": "When a user sends a message shorter than 3 words, the retriever returns no context chunks. Expected: still retrieve relevant chunks based on semantic similarity.",
    "files_hint": ["src/chat/retriever.py"],
    "agent": "bug-fixer"
  },
  {
    "title": "Feature: Add health check endpoint",
    "type": "feature",
    "priority": "medium",
    "description": "Add a simple HTTP health check endpoint that returns the bot's status, loaded model info, and ChromaDB collection stats.",
    "agent": "feature-dev"
  },
  {
    "title": "Test: Cover knowledge base generation edge cases",
    "type": "test",
    "priority": "low",
    "description": "Add tests for `src/data/generate_knowledge_base.py` covering: empty input, malformed JSON chunks, API timeout handling, and resume-from checkpoint logic.",
    "files_hint": ["src/data/generate_knowledge_base.py"],
    "agent": "test-specialist"
  }
]
```

### Task JSON Schema Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | Yes | Issue title. Prefix with `Fix:`, `Feature:`, `Improvement:`, or `Test:` |
| `type` | string | Yes | One of: `bug`, `feature`, `improvement`, `test` |
| `priority` | string | Yes | One of: `high`, `medium`, `low` |
| `description` | string | Yes | Detailed description. Supports Markdown. |
| `files_hint` | string[] | No | Relevant file paths to help the agent find context faster |
| `agent` | string | No | Agent profile to use: `bug-fixer`, `feature-dev`, or `test-specialist`. Defaults to standard Copilot if omitted. |

### Setup (One-Time)

1. **Enable Copilot coding agent**: Repo Settings → Code & automation → Copilot → Coding agent ✅
2. **Create labels**: Run the label setup script:
   ```bash
   # Requires GitHub CLI (gh) authenticated
   bash .github/scripts/setup-labels.sh YOUR_USERNAME/KayaChatBot
   ```
3. **Push this branch to main**: The `copilot-setup-steps.yml` workflow must be on the default branch for Copilot to use it.

### MCP Servers

The GitHub MCP server is configured in `.vscode/mcp.json` for local VS Code Copilot Chat, giving the IDE access to issues, PRs, and repo context. The coding agent on GitHub.com has the built-in GitHub MCP server enabled by default.

## 🤝 Contributing

This is a personal project, but suggestions welcome! Open issues for bugs or feature requests — or add them to `tasks.json` and let Copilot handle it.

## 📄 License

Private project - All rights reserved.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient LLM fine-tuning
- [HuggingFace](https://huggingface.co/) for model hosting and transformers
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) for synthetic data generation

---

**Note:** This bot is trained on personal group chat data. Its "memory" of past events and people comes entirely from that history. The bot is NOT a group member — it is an AI assistant with access to the group's collective memory.