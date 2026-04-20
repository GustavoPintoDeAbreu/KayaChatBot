# KayaChatBot

An AI assistant bot for the **Kaya** Portuguese friend group chat, trained on real WhatsApp and Instagram conversations using LoRA fine-tuning. Supports multiple model profiles including Qwen3-14B and Gemma 4 E4B (8B).

## 🎯 Overview

KayaChatBot is the AI memory of the Kaya group. It is **not** a group member — it is an assistant with access to the group's collective memory. It learns facts, events, and relationships from the group's conversation history so it can answer questions like "what did we talk about at the beach trip?" or just have a casual chat. It communicates naturally in **European Portuguese or English**.

**Key Features:**
- Extracts and cleans messages from WhatsApp exports and Instagram JSON
- Generates synthetic multi-turn training conversations using xAI Grok or Azure OpenAI GPT-4.1-mini
- **Always-on RAG**: Retrieves relevant context for every message (not just detected questions)
- **Dual knowledge system**: JSON member profiles injected into the system prompt + curated ChromaDB knowledge base
- **Automated knowledge generation**: Uses Azure GPT-4.1-mini to extract biographical facts from chat history
- **Benchmarking toggle**: Switch between `both` / `json_only` / `chromadb_only` / `none` knowledge approaches
- Fine-tunes Qwen3-14B and Gemma 4 E4B using LoRA (Low-Rank Adaptation) with 4-bit quantization
- **Unrestricted mode**: Uncensored system prompt preamble injected at runtime — dark humour, roasts, banter, and profanity all enabled; no guardrails for this private group
- Offline biography comparison tool generates member profiles with 5 model backends
- Efficient training on a single consumer GPU (RTX 3090 24 GB)

## 🧠 Model History & VRAM

The project started with Qwen3-14B and later moved to Gemma 4 E4B as the primary model.

### Trained Models

| Profile | Base Model | LoRA Rank/Alpha | VRAM Training | VRAM Inference (4-bit) | Output Dir |
|---|---|---|---|---|---|
| `qwen3-14b` | Qwen3-14B (14B params) | r=32, α=32 | ~15 GB | ~10–12 GB | `models/kaya_v2_synthetic/` |
| `gemma4-e4b` | Gemma 4 E4B (~8B active) | r=16, α=16 | ~11 GB alloc / ~12.6 GB reserved | ~8–9 GB | `models/kaya_gemma4_e4b/` |

All numbers on NVIDIA RTX 3090 24 GB, CUDA 12.4. Always leave ≥2 GB VRAM headroom.

**Training speed (Gemma 4 E4B):** ~2.5 s/step, 1200+ steps to convergence.

### Restriction Level Benchmarks

Both models were evaluated for how well they preserve group banter and edgy content versus self-censoring:

| Capability | Gemma 4 E4B (finetuned) | Qwen3-14B (finetuned) | Grok (cloud) | Azure GPT-4.1-mini (cloud) |
|---|---|---|---|---|
| Roasts & banter | ✅ Full commitment | ✅ Strong | ✅ Good | ⚠ May soften |
| Dark humour | ✅ Yes | ✅ Yes | ✅ Yes | ⚠ Occasional refusal |
| Profanity / slang | ✅ Natural | ✅ Natural | ✅ Yes | ⚠ Sometimes hedges |
| Third-person factual recall | ✅ | ✅ | ✅ | ✅ |
| Portuguese banter style | ✅ (finetuned on real data) | ✅ (finetuned on real data) | Decent | Generic |

Cloud LLM self-censorship: Azure content filters are managed at the Azure Portal resource level (no API toggle). Grok is less restricted by design. Local finetuned models bypass all filters entirely.

Use the bio generator (`src/data/generate_bios_offline.py`) to run your own comparison — see [Model Comparison Tool](#-model-comparison-tool) below.

### Switch Model Profile

```yaml
# config.yaml
active_model_profile: gemma4-e4b   # or: qwen3-14b
```

Or pass `--profile gemma4-e4b` / `--profile qwen3-14b` on any CLI command.

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

### Example Interaction

```
User: quem é o peter?
📚 Retrieved 4 conversation chunks + 2 knowledge facts (0.31s)

Kaya Bot: O Peter é um dos membros mais activos do grupo — está sempre a organizar jantares,
póqueres, e eventos. Pelas conversas, vive em Paço de Arcos, tem um cão chamado Kaya, e
trabalhava na DAZN como editor. É conhecido pelo bom humor, pela paixão pelo futebol
(frustrações com o Benfica incluídas) e por sempre ter um sítio para receber o grupo.

---
User: and what about gil?
📚 Retrieved 5 conversation chunks + 1 knowledge fact (0.28s)

Kaya Bot: Gil is the group's resident audio nerd — constantly talking about 8D audio, Dolby
Atmos, and music technology. He has a daughter and a dog called Cuca, often brings his
partner Luana to group events, and was planning to do a plumbing course starting September
2026. He plays padel, enjoys techno music, and recovered from a broken nose after a football
match. Sharp sense of humour; never misses a poker night.

---
User: oi!
📚 Retrieved 3 conversation chunks (0.19s)

Kaya Bot: oi! tudo bem contigo? 😄
```

> Responses always reference retrieved context — the bot never fabricates group history.

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
- CUDA-capable GPU: ≥16 GB VRAM for Qwen3-14B, ≥12 GB for Gemma 4 E4B (RTX 3090 24 GB recommended)
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
Model-specific settings are in `configs/models/*.yaml`; select a profile in `config.yaml`.

| Profile | Loader | LoRA | Steps | Output |
|---|---|---|---|---|
| `qwen3-14b` | `FastLanguageModel` | r=32, α=32 | ~1500 | `models/kaya_v2_synthetic/` |
| `gemma4-e4b` | `FastModel` | r=16, α=16 | ~1200 | `models/kaya_gemma4_e4b/` |

**Output:** adapter files under the configured `output_dir`.

## 🔬 Model Comparison Tool

Generate biographical profiles for every group member using all available model backends, then compare them side-by-side:

```bash
# All 5 backends (requires API keys for cloud models)
kaya_chatbot_env/bin/python src/data/generate_bios_offline.py \
    --models gemma4,qwen3,grok,azure,azure_gpt53

# Local models only (fully offline)
kaya_chatbot_env/bin/python src/data/generate_bios_offline.py --models gemma4,qwen3

# Rebuild comparison report from existing JSONs (no generation)
kaya_chatbot_env/bin/python src/data/generate_bios_offline.py --compare-only

# Disable uncensored preamble (investigate self-censorship baseline)
kaya_chatbot_env/bin/python src/data/generate_bios_offline.py --models grok --no-uncensored
```

Outputs:
- `reports/bios/bios_{model}.json` — per-member structured profiles for each backend
- `reports/bios/bio_comparison.md` — side-by-side markdown comparison (age, occupation, interests, narrative bio)

This is useful for benchmarking **information extraction quality** and **restriction level** across finetuned vs. cloud models. Cloud self-censorship notes: Azure content filters are controlled at the Azure Portal resource level (no API toggle); Grok is less restricted by design.

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
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) for synthetic data generation

---

**Note:** This bot is trained on personal group chat data. Its "memory" of past events and people comes entirely from that history. The bot is NOT a group member — it is an AI assistant with access to the group's collective memory.