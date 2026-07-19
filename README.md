# KayaChatBot

An AI assistant bot for the **Kaya** Portuguese friend group chat, trained on real WhatsApp conversations using LoRA fine-tuning. Supports multiple model profiles; production currently runs Gemma 4 E4B (heretic base) trained at 4096 context, with Qwen3-14B available as an alternative profile. All of the app runs on a single RTX 3090 GPU.

## 🎯 Overview

KayaChatBot is the AI memory of the Kaya group. It is **not** a group member — it is an assistant with access to the group's collective memory. It learns facts, events, and relationships from the group's conversation history so it can answer questions like "what did we talk about at the beach trip?" or just have a casual chat. It communicates naturally in **European Portuguese or English**.

**Key Features:**
- Extracts and cleans messages from WhatsApp exports
- Generates synthetic training conversations fully on-prem with a local teacher model (no group data leaves the box)
- **Always-on RAG**: Retrieves relevant context for every message (not just detected questions)
- **Dual knowledge system**: JSON member profiles injected into the system prompt + curated ChromaDB knowledge base
- **Automated knowledge generation**: A local on-prem teacher model (Qwen3.5-27B, 4-bit) extracts biographical facts from chat history — no data leaves the machine
- **Benchmarking toggle**: Switch between `both` / `json_only` / `chromadb_only` / `none` knowledge approaches
- Fine-tunes the active model profile (Gemma 4 E4B by default) using LoRA with 4-bit quantization
- Efficient training on consumer GPUs (RTX 3090 24 GB; ~11 GB VRAM for gemma4-e4b, ~15 GB for qwen3-14b)

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
  knowledge_approach: "json_only"   # best benchmark score
  # Options:
  #   "both"          — JSON members in system prompt AND ChromaDB KB retrieval (can overflow the token budget)
  #   "json_only"     — JSON injection only, no KB retrieval (best benchmark score)
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
│   │   ├── extract_all_messages.py   # WhatsApp export parser
│   │   ├── generate_synthetic_data.py # LLM synthetic conversation generation
│   │   ├── generate_knowledge_base.py # Biographical fact extraction (local teacher)
│   │   ├── generate_local_synthetic.py # On-prem synthetic training data (local teacher)
│   │   ├── local_teacher.py          # Shared 4-bit local teacher model
│   │   ├── build_vector_db.py        # Build ChromaDB collections
│   │   ├── prepare_portuguese_data.py
│   │   ├── merge_datasets.py
│   │   ├── format_direct_training.py
│   │   └── readers.py
│   ├── finetuning/                   # Model training
│   │   └── train.py
│   ├── chat/                         # Inference & interaction
│   │   ├── chat.py                   # Interactive chat loop (always-on RAG)
│   │   ├── web_app.py                # Gradio web UI (suggestions, feedback, metrics)
│   │   ├── whatsapp_server.py        # WhatsApp bridge (WAHA webhook server)
│   │   ├── engine.py                 # Shared generation engine (web + WhatsApp)
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
├── run_full_pipeline.py              # Pipeline orchestrator
├── Dockerfile
├── docker-compose.yml
└── .env                              # API keys (gitignored)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU with 16GB+ VRAM (for training; less for inference)
- xAI API access *(optional)* — used only for the production web-search feature and the eval-time LLM judge; no group data is ever sent, except member-free web queries

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

4. **Set up credentials** *(optional — web-search / eval judge only)*
   ```bash
   echo 'XAI_API_KEY=your_key_here' >> .env
   ```

### Data Preparation

1. **Add your chat data**
   - WhatsApp: Export chat as TXT → `data/wpp/`

2. **Extract and clean messages**
   ```bash
   kaya_chatbot_env/bin/python src/data/extract_all_messages.py
   ```

3. **Generate knowledge base from chat history (recommended)**

   Runs on the local on-prem teacher model — stop the serving container first (the teacher needs the GPU).
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
# Full pipeline: extract → format → merge → train (fully on-prem)
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
- Reads WhatsApp TXT export files
- Cleans and standardizes messages (removes URLs, media, system messages)
- Merges consecutive messages from the same sender

**Output:** 
- `data/all_messages_cleaned.jsonl` — all cleaned messages
- `data/finetune_chunks.jsonl` — chunked messages for generation

### 1b. **Knowledge Base Generation** (`generate_knowledge_base.py`) *(optional, recommended)*
- Iterates over the cleaned message history in ~2000-token chunks
- The local on-prem teacher model extracts biographical facts per member (no data leaves the box)
- Merges facts into member profiles and curated knowledge entries
- Checkpoints every 5 chunks; resumable with `--resume-from N`

**Output (updated):** `data/group_members.json`, `data/group_knowledge.json`

### 2. **Local Synthetic Data Generation** (`generate_local_synthetic.py`) *(manual, GPU)*
- A local teacher model (Qwen3.5-27B, 4-bit) generates behavior-targeted Q&A grounded in local RAG context
- Fully on-prem — no API calls, no data leaves the machine

**Output:** `data/synthetic_local.jsonl`

### 2b. **Direct Training Format** (`format_direct_training.py`)
- Formats raw messages into training pairs without any API calls
- Context blocks use `=== Conversas relevantes do grupo ===` markers (matches inference format)

**Output:** `data/direct_training.jsonl`

### 3. **Build Vector Database** (`build_vector_db.py`)
- Builds `kaya_conversations` ChromaDB collection from message chunks
- Builds `kaya_knowledge_base` ChromaDB collection from `group_knowledge.json`
- Uses BAAI/bge-m3 embeddings

**Output:** `data/rag_db/` (ChromaDB)

### 4. **Dataset Merging** (`merge_datasets.py`)
- Combines datasets and applies the active profile's chat template (gemma-4 or ChatML)
- 90/10 train/val split (`data.train_test_split`)

**Output:** `data/train_synthetic.jsonl`, `data/val_synthetic.jsonl`

### 5. **Fine-Tuning** (`train.py`)
- Loads the active profile's model with 4-bit quantization (Unsloth)
- LoRA adapters per profile (gemma4: r=16/alpha=32; qwen3: r=32/alpha=32)
- Steps/LR set by the active profile in `config.yaml`

**Output:** the profile's `training.output_dir` (currently `models/kaya_gemma4_heretic_seq4096/`)

## ⚙️ Configuration

All settings live in [config.yaml](config.yaml). Key sections:

### Pipeline Mode

```yaml
pipeline:
  generate_knowledge: false  # true = run knowledge extraction on the local teacher (before training)
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

Model and training settings come from the **active model profile** (`active_model_profile` in `config.yaml`, currently `gemma4-e4b-seq4096`), deep-merged over the top-level `model:`/`training:` sections by `src/config_loader.py`:

```yaml
active_model_profile: "gemma4-e4b-seq4096"

model_profiles:
  gemma4-e4b-seq4096:
    model:
      model_id: "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
      max_seq_length: 4096
      lora_r: 16
      lora_alpha: 32
    training:
      output_dir: "./models/kaya_gemma4_heretic_seq4096"
      learning_rate: 0.00005
      max_steps: 450
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
kaya_chatbot_env/bin/python scripts/validate_pipeline.py
```

### Run the Test Suite
```bash
kaya_chatbot_env/bin/python -m pytest tests/ -v
```

## 💡 Tips & Best Practices

### Knowledge Base Quality
- Run `generate_knowledge_base.py` after any significant addition of new messages
- Review `data/group_members.json` bios manually and edit them for accuracy
- The more diverse the chat data, the richer the extracted biographies

### Knowledge Generation Runs
- The local teacher needs the GPU — stop the serving container first (`scripts/app_down.sh prod`)
- Use `--resume-from N` to resume if the script is interrupted (checkpoints every 5 chunks)

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
- `CleanedMessage` — Standardized message format
- `FinetuneChunk` — Chunked messages for generation
- `SyntheticConversation` — Generated Q&A pairs
- `TrainingExample` — Formatted training instance

## 🐳 Docker Support

Docker configuration is available (`Dockerfile`, `docker-compose.yml`) with profiles for prod, dev, test, and the Cloudflare Tunnel. See `DEPLOYMENT.md` for the full runbook and `WHATSAPP.md` for the WhatsApp bridge.

```bash
docker-compose up --build                          # prod web app
docker compose --profile dev up -d kaya-dev        # dev UI on :7861
docker compose --profile test run --rm kaya-test   # test suite in-container
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

## ⚠️ Things to Be Aware Of

### PEFT `float8_e8m0fnu` Patch
Current PEFT (0.19.0) has a bug where it checks for `torch.float8_e8m0fnu` dtype which doesn't exist in PyTorch 2.6. Two files in the venv are manually patched with `hasattr` guards:
- `kaya_chatbot_env/lib/python3.12/site-packages/peft/tuners/tuners_utils.py`
- `kaya_chatbot_env/lib/python3.12/site-packages/peft/tuners/lora/layer.py`

**If you reinstall or upgrade PEFT, these patches need to be reapplied.** The patch wraps `torch.float8_e8m0fnu` references in `hasattr(torch, "float8_e8m0fnu")` checks.

### Training Memory (OOM)
The training script (`src/finetuning/train.py`) uses a flat code path — it calls `SFTTrainer` directly without a wrapper class. This was done because a previous `KayaTrainer` wrapper caused unexplained memory spikes (1.4 GB → 20+ GB RSS) during `SFTTrainer.__init__`. Key settings that prevent OOM:
- `skip_memory_metrics=True` — avoids the HF `TrainerMemoryTracker` busy-loop
- `dataset_num_proc=1` — prevents fork-based memory duplication
- `dataloader_pin_memory=False` and `dataloader_num_workers=0` — reduces memory overhead
- Do **not** set `builtins.psutil` or run background memory monitor threads during training

### Gemma 4 Specifics
- Uses `FastModel` (not `FastLanguageModel`) with Unsloth ≥2026.4.5
- Chat template: `get_chat_template(tokenizer, "gemma-4")` — produces `<|turn>user\n...<turn|>\n` format
- Thinking mode (`<|think|>`) must be **disabled** during SFT training
- `autocast_adapter_dtype=False` is required for PEFT compatibility
- **Model class**: `Gemma4ForConditionalGeneration` — NOT registered with `AutoModelForCausalLM`. Inference code must use `Gemma4ForConditionalGeneration.from_pretrained()` or Unsloth's `FastModel` instead.
- **Processor vs tokenizer**: Unsloth returns a `Gemma4Processor` (not a plain tokenizer). When tokenizing text, always pass `text=` as a keyword argument: `tokenizer(text=input_text, ...)`. Positional args are interpreted as `images` and will crash.

### Training Checkpoints
After training, only the best-eval-loss checkpoint is kept (e.g., `checkpoint-1200`). The final adapter is saved at the output directory root. If you need to roll back, load from the checkpoint subdirectory.

### Package Version Pins
- `trl<=0.24.0` — newer versions have breaking API changes with `SFTConfig`
- `unsloth>=2026.4.5` — required for Gemma 4 support via `FastModel`
- `transformers>=5.5.0` — needed for `Gemma4ForConditionalGeneration`

## 🤝 Contributing

This is a personal project, but suggestions welcome! Open issues for bugs or feature requests.

## 📄 License

Private project - All rights reserved.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient LLM fine-tuning
- [HuggingFace](https://huggingface.co/) for model hosting and transformers

---

**Note:** This bot is trained on personal group chat data. Its "memory" of past events and people comes entirely from that history. The bot is NOT a group member — it is an AI assistant with access to the group's collective memory.
