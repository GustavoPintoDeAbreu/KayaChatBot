# KayaChatBot

An AI "extra member" for a Portuguese friend group chat, trained on real WhatsApp and Instagram conversations using Llama-3.1-8B with LoRA.

## 🎯 Overview

KayaChatBot is designed to feel like an extra member of the group chat — someone who was always there and remembers everything. It learns facts, events, and relationships from the group's conversation history so it can answer questions like "what did we talk about at the beach trip?" or just have a casual chat. It communicates in **European Portuguese or English** (not necessarily using the group's own slang or lingo).

**Key Features:**
- Extracts and cleans messages from WhatsApp exports and Instagram JSON
- Generates synthetic multi-turn conversations using xAI Grok or Azure OpenAI GPT-4
- **RAG System**: Retrieves relevant conversation history for factual questions
- Merges with general Portuguese instruction data for better language understanding
- Fine-tunes Llama-3.1-8B using LoRA (Low-Rank Adaptation) with 4-bit quantization
- Dual-mode chat: Q&A with context vs casual conversation
- Efficient training on consumer GPUs (requires ~12GB VRAM)

## 🤖 RAG Features

KayaChatBot includes a Retrieval-Augmented Generation (RAG) system for enhanced conversational capabilities:

### Dual-Mode Chat
- **Q&A Mode**: When asked questions, retrieves relevant conversation history and provides context-aware answers
- **Casual Mode**: For general conversation, responds naturally as a group member

### Smart Context Retrieval
- Uses Alibaba-NLP GTE multilingual embeddings (optimized for Portuguese)
- Person-aware filtering: Queries about "Peter" retrieve Peter's messages
- Semantic search across 1750+ conversation chunks
- Real-time retrieval stats during chat

### Example Usage
```
User: What did Peter say about music?
📚 Retrieved 3 relevant chunks
Kaya: Peter said he loves this music, chemistry is top ahah

User: olá pessoal
Kaya: oi tudo bem? 😊
```

## 📁 Project Structure

```
KayaChatBot/
├── src/
│   ├── data/                    # Data processing & generation
│   │   ├── extract_all_messages.py
│   │   ├── generate_synthetic_data.py
│   │   ├── prepare_portuguese_data.py
│   │   ├── merge_datasets.py
│   │   └── readers.py            # Data readers and formatters
│   ├── finetuning/              # Model training
│   │   ├── train.py
│   │   └── trainer.py            # Training utilities
│   ├── chat/                    # Inference & interaction
│   │   ├── chat.py
│   │   ├── inference.py
│   │   └── retriever.py          # RAG retrieval system
│   ├── testing/                 # Test scripts
│   │   ├── test_azure.py
│   │   └── test_azure.ipynb
│   ├── llm_providers/           # LLM provider abstractions
│   │   ├── azure_provider.py
│   │   ├── xai_provider.py
│   │   └── base.py
│   └── models.py                # Pydantic data models
├── data/                        # Generated data (gitignored)
├── models/                      # Trained models (gitignored)
├── config.yaml                  # Training configuration
├── run_full_pipeline.py         # Main pipeline orchestrator
├── test_pipeline.py             # Test mode runner
├── validate_pipeline.py         # Data validation
└── .env.template                # Credentials template

```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU with 12GB+ VRAM (for training)
- Azure OpenAI API access (for synthetic generation)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd KayaChatBot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv kaya_chatbot_env
   
   # Windows
   kaya_chatbot_env\Scripts\activate
   
   # Linux/Mac
   source kaya_chatbot_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up credentials**
   ```bash
   cp .env.template .env
   # Edit .env and add your Azure OpenAI credentials
   ```

### Data Preparation

1. **Add your chat data**
   - WhatsApp: Export chat as TXT → `data/wpp/`
   - Instagram: Download JSON messages → `data/insta/`

2. **Run the pipeline**
   ```bash
   # Interactive mode (recommended for first run)
   python run_full_pipeline.py
   
   # Or step by step:
   python src/data/extract_all_messages.py
   python src/data/generate_synthetic_data.py
   python src/data/prepare_portuguese_data.py
   python src/data/merge_datasets.py
   ```

### Training

```bash
python src/finetuning/train.py
```

Training takes ~2-4 hours on RTX 3090 depending on dataset size.

### Chat with Your Model

```bash
# Interactive chat
python src/chat/chat.py

# Quick inference test
python src/chat/inference.py
```

## 📊 Pipeline Stages

### 1. **Message Extraction** (`extract_all_messages.py`)
- Reads WhatsApp TXT and Instagram JSON files
- Cleans and standardizes messages (removes URLs, media, system messages)
- Merges consecutive messages from the same sender
- Creates finetune chunks (~50K tokens each for GPT-4 context window)

**Output:** 
- `data/all_messages_cleaned.jsonl` - All cleaned messages
- `data/finetune_chunks.jsonl` - Chunked messages for generation

### 2. **Synthetic Data Generation** (`generate_synthetic_data.py`)
- Uses xAI Grok or Azure OpenAI GPT-4.1-mini to generate diverse Q&A conversations
- Creates 2-5 turn conversations based on your chat history
- Varies question types: personality, opinions, events, relationships
- Rate limiting: ~4 chunks/minute (200K TPM limit for Azure, higher for xAI)

**Output:** `data/synthetic_kaya.jsonl`

**Usage:**
```bash
# Batch mode (default - processes all chunks)
python src/data/generate_synthetic_data.py

# Single conversation (for rate limit workarounds)
python src/data/generate_synthetic_data.py --mode single --depth 4

# Generate specific number of conversations
python src/data/generate_synthetic_data.py --mode count --count 50
```

### 3. **Portuguese Dataset** (`prepare_portuguese_data.py`)
- Downloads alpaca-portuguese instruction dataset from HuggingFace
- Filters for quality (>20 chars, Portuguese text)
- Converts to ShareGPT format

**Output:** `data/synthetic_portuguese.jsonl`

### 4. **Dataset Merging** (`merge_datasets.py`)
- Combines Kaya-specific and general Portuguese data
- Applies Llama-3.1 chat template formatting
- Shuffles and splits into train/val (90/10)

**Output:** 
- `data/train_synthetic.jsonl`
- `data/val_synthetic.jsonl`

### 5. **Fine-Tuning** (`train.py`)
- Loads Llama-3.1-8B-Instruct with 4-bit quantization
- Applies LoRA adapters (rank=16, alpha=16)
- Trains for 3 epochs with cosine learning rate schedule
- Saves checkpoints every 100 steps

**Output:** `models/kaya_v1/`

## ⚙️ Configuration

### Test Mode

Toggle between quick testing and full production runs by editing [config.yaml](config.yaml):

```yaml
test_mode:
  enabled: true  # Set to true for fast testing, false for production
```

**Test Mode (enabled: true):**
- Processes only 1 Instagram file
- Generates from only 2 finetune chunks
- Creates 2 conversations per chunk
- Uses 100 Portuguese examples
- Trains for only 50 steps (quick validation)

**Production Mode (enabled: false):**
- Processes all data files
- Generates from all finetune chunks
- Creates 5 conversations per chunk
- Uses 800 Portuguese examples
- Trains with full parameters (6000 steps)

### Model & Training

Edit `config.yaml` to customize:

```yaml
model:
  model_id: "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
  max_seq_length: 2048

training:
  output_dir: "./models/kaya_v1"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 0.0002
  lora_r: 16
  lora_alpha: 16
```

## 🧪 Testing

### Test Mode
Set `TEST_MODE = True` in scripts to process only small samples:
- Extract: First Instagram file only
- Generate: First 2 finetune chunks
- Portuguese: 2000 examples

Run test pipeline:
```bash
python test_pipeline.py
```

### Validate Outputs
Check pipeline outputs without regenerating:
```bash
python validate_pipeline.py
```

### Test Azure Connection
```bash
python src/testing/test_azure.py
```

## 💡 Tips & Best Practices

### Rate Limiting
- Azure OpenAI has strict rate limits (200K TPM for gpt-4.1-mini)
- Use `generate_many.bat` for automated generation with delays (no longer available, use direct Python script with delays)
- Monitor Azure portal for quota usage

### Data Quality
- More chat data = better results (aim for 10K+ messages)
- Diverse conversation topics improve generalization
- Review synthetic_kaya.jsonl to ensure quality before training

### Training
- Monitor GPU memory with `nvidia-smi`
- Reduce batch size if OOM errors occur
- Training loss should decrease steadily (check wandb logs if enabled)

### Inference
- First load is slow (~1 minute) due to model initialization
- Subsequent responses are fast (~2-3 seconds)
- Adjust temperature in config for more/less creative responses

## 📦 Pydantic Models

The codebase uses Pydantic models for type safety (see `src/models.py`):

- `WhatsAppMessage` - Raw WhatsApp TXT message
- `InstagramMessage` - Raw Instagram JSON message  
- `CleanedMessage` - Standardized message format
- `FinetuneChunk` - Chunked messages for generation
- `SyntheticConversation` - Generated Q&A pairs
- `TrainingExample` - Formatted training instance

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

## 🤝 Contributing

This is a personal project, but suggestions welcome! Open issues for bugs or feature requests.

## 📄 License

Private project - All rights reserved.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient LLM fine-tuning
- [HuggingFace](https://huggingface.co/) for model hosting and transformers
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) for synthetic data generation

---

**Note:** This bot is trained on personal group chat data. Its "memory" of past events and people comes entirely from that history.