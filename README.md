# KayaChatBot

An AI assistant bot for the **Kaya** Portuguese friend group chat. It answers from
the group's collective memory — RAG over real WhatsApp history, retrieved when
the question calls for it and deliberately skipped when it does not.

**Production runs a stock `gemma-4-12b-it` at Q6_K with no fine-tune**, served by
llama.cpp with a 32K context window (RAG budget 14,000 tokens). An August 2026
bake-off across 14 serving configurations found the stock 12B beat the previously
fine-tuned Gemma 4 E4B on every judged dimension, and that the WhatsApp LoRA added
nothing on an identical base — so the adapter was retired from the live path, and on
2026-10-26 the fine-tuning pipeline was removed from the repo (it is in git history).

The box has **2× RTX 3090 (24 GB each, no NVLink)**: prod owns one card, and the
other stays free for dev, training and other projects. They are two independent
devices, not a 48 GB pool — see `CLAUDE.md` for what that does and does not allow.

## 🎯 Overview

KayaChatBot is the AI memory of the Kaya group. It is **not** a group member — it is an assistant with access to the group's collective memory. It learns facts, events, and relationships from the group's conversation history so it can answer questions like "what did we talk about at the beach trip?" or just have a casual chat. It communicates naturally in **European Portuguese or English**.

**Key Features:**
- **Lives in WhatsApp**: a WAHA bridge puts the bot in the real group chat (`WHATSAPP.md`), plus a Gradio web UI
- **Intent-routed retrieval**: every message is classified first, and the mode picks retrieval, prompt and reply length together — banter gets no group context at all
- **Dual knowledge system**: JSON member profiles injected into the system prompt + curated ChromaDB knowledge base
- **Remembers the thread**: 60 lines verbatim — about thirty exchanges — plus a per-chat rolling summary, so a long exchange survives what semantic search alone cannot return
- **Voice in and out**: voice notes are transcribed with faster-whisper; replies can be spoken with Piper, per-language and sticky per chat
- **Sees pictures**: inbound photos are described into text by the serving model's own vision projector, and so become searchable memory later. It does not make or edit them — that was removed on 2026-09-04, after two weeks of logs recorded a single request that misfired
- **Reads documents**: a shared PDF is extracted, summarised by the local model and indexed **page by page**, so it can be quoted with a page number later. Nothing is announced in the chat
- **Argues, when asked**: it will take a side or judge who is right in a running argument, decides for itself whether the case needs facts it does not have, and may only cite sources it actually retrieved — an invented page number is stripped by code, not merely discouraged by a prompt
- **Does not tell the same joke twice**: an open-ended answer (banter, a roast, an
  opinion) is built from a random handful of each member's facts and is shown what
  it already said about that person, so "roast me" stops returning the same four
  beats. A factual answer keeps the full set and stays identical every time it is asked
- **Tells you when it is wrong**: `/bug` and `/feedback` on WhatsApp (plus a web form) collect reports into the Feedback dashboard — and are deliberately never stored as group memory
- **Automated knowledge generation**: A local on-prem teacher model (Qwen3.5-27B, 4-bit) extracts biographical facts from chat history — no data leaves the machine
- **Benchmarking toggle**: Switch between `both` / `json_only` / `chromadb_only` / `none` knowledge approaches

## 🤖 RAG & Knowledge System

### Retrieval Follows Intent

Retrieval used to run on *every* message, and that was a bug: `Ahahhha` came back
with a 77-word analysis of group dynamics, and `hey` with a roast aimed at a
randomly chosen member. Handed a pile of member profiles and told to elaborate,
the model finds someone to talk about.

`src/chat/router.py` now classifies each message first, and the mode selects
retrieval, prompt and reply length together:

| Mode | What it is | Group retrieval |
|---|---|---|
| `banter` | Social noise — laughter, emoji, greetings, insults, "roast me" | **None** |
| `mixed` | Names a person or event without asking to be informed | Yes, answers short |
| `factual` | A question about the group, its members or its history | Yes, full context |
| `roast` | Asking for a verdict *aimed at* a member — "quem é o mais burro?" | Yes, and steered off whoever it just hit |
| `general` | A question about the world — football, cooking, code, advice | **None**, and no member profiles |
| `debate` | Asked to argue a position, judge who is right, or fact-check a claim | Yes, plus shared documents and — when it decides it needs them — web lookups |

`debate` is the newest and the only one that goes looking for evidence. A `roast`
is a verdict about a **person**; a `debate` is a verdict about a **claim**. It
fires only when somebody asks for it — the bot never volunteers itself into an
argument — and it is the one mode that may cite. See
[Arguing, with sources](#-arguing-with-sources) below.

One request is not a mode at all: a **counting** question ("quantas vezes é que
o Rafa disse isso?") bypasses retrieval entirely. Top-k search returns the
chunks nearest a question and cannot answer "how many times", so `src/chat/tally.py`
scans the message log and hands the model a finished table to phrase.

Any router failure falls back to `factual`, i.e. the old always-on behaviour, so
a bad classification degrades to a correct-but-verbose answer rather than a wrong
one.

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
- Person-aware filtering: a query naming a member post-filters retrieval by the `participants`/`mentioned` metadata, so it gets that member's messages rather than whatever is semantically nearest
- Semantic search across conversation chunks
- A 14,000-token context budget, enforced by truncating the lowest-priority context first (conversation chunks, then knowledge, then recent summaries)
- Dates are only surfaced when the question is about timing (`_has_temporal_intent`); ordinary answers stay date-free
- Real-time retrieval stats during chat

### Example Usage
```
User: o que sabes sobre o <member>?          → factual
📚 Retrieved 3 conversation chunks + 1 knowledge fact
Kaya Bot: <grounded answer, with what it actually found>

User: olá pessoal                             → banter
📚 Retrieved nothing — by design
Kaya Bot: oi! tudo bem? 😊

User: quem é melhor, Ronaldo ou Messi?        → general
📚 Retrieved nothing, and no member profiles injected
Kaya Bot: <answers the question, not a report about the group>
```

## ⚖️ Arguing, with sources

Asked to, the bot will argue a position or judge a running argument. Asked to do
neither, it stays out of it — this fires on an explicit request only, because a
bot that volunteers itself into every disagreement is a bot people mute.

Two shapes, picked automatically from how the request is phrased:

* **arbitrate** — "quem tem razão nisto?", "vê lá a conversa e diz quem tem razão,
  sê analítico". It goes claim by claim and may side with different people on
  different points, or with nobody.
* **advocate** — "defende o contrário", "I'll defend communism u capitalism". It
  takes the side it is given and commits to it, instead of both-sidesing.

### It decides for itself whether it needs facts

Before answering, a debate turn writes a private plan and ends it with either
`NEED: none` or up to three search queries. What decides this is not a keyword but
whether the case rests on a number — "defende que a habitação pública é a solução"
is an argument from principle, while "o salário mínimo subiu mais que a inflação?"
is not. Measured against the live model:

| turn | lookups | latency |
|---|---|---|
| cost of food, with a shared document in hand | 0 | ~13 s |
| an argument from principle | 0 | ~15 s |
| "procura os números" | 2 | ~35 s |

Documents are searched on every debate regardless; the web is only touched for an
explicit `NEED`, and **a query naming a group member never leaves the box**.

### Cite or concede

The bot may only cite sources it was actually handed this turn, and that is
enforced twice — in the prompt, and by `src/chat/sources.py`, which deletes any
citation that was not. A `[D3]` when two documents were retrieved is removed; a
page number no retrieved chunk covers is redacted to "essa parte do documento".

The claim itself is left standing. Dropping the marker turns it into the bot's own
assertion, which is honest; deleting the sentence would change what it said. And
the check runs on **every** mode that retrieved a document, not only debates,
because "dá-me uma página" is an ordinary question.

If it has nothing, it says so. "Não tenho fonte para isso" is a better answer than
a plausible one, and this group will check.

## 📄 Documents shared in the chat

A PDF sent to the bot is read, summarised by the **local** model, and indexed page
by page. Nothing is announced — the message quietly becomes
`[Documento: nome, N páginas — sinopse]`, exactly the way a photo becomes
`[Imagem: …]`, so it is searchable from then on:

```
Rafa:  @Kaya que documentos é que o Bernardo partilhou aqui?
Kaya:  O Bernardo partilhou dois documentos: um sobre a desregulação do mercado
       de trabalho e o declínio sindical desde 1980, e outro do FMI sobre a
       acessibilidade à habitação.
```

**Page numbers are the point.** A citation that cannot name a page is worth
nothing to a group that asks "página X capítulo X", so every chunk carries the
pages it came from and answers cite them.

| | |
|---|---|
| Formats | PDF, plain text, Markdown |
| Extraction | `pypdf` — pure Python, no system dependencies |
| Synopsis | The local serving model. Group data does not leave the box. |
| Storage | `data/documents/<scope>/<sha256>.pdf` plus a JSON sidecar |
| Index | ChromaDB `kaya_documents`, separate from `kaya_conversations` because a full pipeline rebuild drops that one |
| Scope | A document shared in a DM is **never** visible from the group |
| Size | ~26–32 KB per chunk. 99 pages across 3 real PDFs → 105 chunks → 3.3 MB |

Re-sharing a document the group already has costs nothing: it is recognised by
content hash, so it is not re-extracted, re-summarised or re-embedded.

Documents already shared before the bot could read them need a WhatsApp export —
WAHA cannot replay them:

```bash
kaya_chatbot_env/bin/python scripts/ingest_documents.py --export chat.txt --media ./chat
kaya_chatbot_env/bin/python scripts/ingest_documents.py --dir ~/papers --stats   # size a corpus, writing nothing
```

## 📁 Project Structure

```
KayaChatBot/
├── src/
│   ├── data/                         # Data processing & generation
│   │   ├── extract_all_messages.py   # WhatsApp export parser
│   │   ├── generate_synthetic_data.py # LLM synthetic conversation generation
│   │   ├── generate_knowledge_base.py # Biographical fact extraction (local teacher)
│   │   ├── local_teacher.py          # Shared 4-bit local teacher model
│   │   ├── build_vector_db.py        # Build ChromaDB collections
│   │   ├── ingest.py                 # Incremental, watermarked ingest of live messages
│   │   ├── message_log.py            # Append-only per-scope log of everything the bot sees
│   │   ├── identity_resolver.py      # Display name → canonical member
│   │   └── prepare_portuguese_data.py
│   ├── chat/                         # Inference & interaction
│   │   ├── web_app.py                # Gradio web UI (suggestions, feedback, metrics)
│   │   ├── whatsapp_server.py        # WhatsApp bridge (WAHA webhook server)
│   │   ├── whatsapp_adapter.py       # Event parsing, mention/whitelist gating, speaker identity
│   │   ├── engine.py                 # Shared generation engine (web + WhatsApp)
│   │   ├── router.py                 # Intent classification: banter/mixed/factual/general/roast/debate + commands
│   │   ├── retriever.py              # RAG retrieval (conversations + KB + documents)
│   │   ├── documents.py              # Shared PDFs → page-numbered searchable memory
│   │   ├── sources.py                # What may be cited, and stripping what may not
│   │   ├── summary.py                # Per-chat rolling summary, written off the reply path
│   │   ├── tally.py                  # Counting questions, answered by scanning the log not by search
│   │   ├── variety.py                # Keeps an open-ended answer off what it already said
│   │   ├── metrics.py                # The one interaction log every surface writes to
│   │   ├── feedback.py               # /bug, /feedback and 👍/👎 ratings
│   │   ├── memory.py                 # Per-chat session history and voice/text preferences
│   │   ├── waha_client.py            # Outbound WhatsApp calls (real + mock)
│   │   ├── response_utils.py         # Member profiles, cleanup, language detection
│   │   ├── face_utils.py             # Face detection, framing and identity scoring for edits
│   │   ├── scope.py                  # What is group-wide memory vs private to one chat
│   │   ├── stt.py / tts.py           # faster-whisper transcription, Piper speech
│   │   ├── vision.py                 # Describing inbound photos
│   │   ├── web_search.py             # Grok web results, synthesized locally into the reply
│   │   ├── suggestions.py            # The follow-up question chips in the web UI
│   │   ├── inference_backend.py      # HTTP backends: gguf (llama.cpp server) | ollama
│   │   └── gpu_lock.py               # One generation at a time; summaries skip rather than queue
│   ├── testing/                      # Benchmarks and the multi-person conversation simulator
│   │   ├── persona_sim.py            # Drives the real webhook with invented people
│   │   ├── scenarios.py              # Beat lists + smoke/standard/long_haul presets
│   │   └── sim_world.py
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
├── models/                           # Adapters, tokenizers + GGUFs (gitignored)
├── config.yaml                       # Central configuration
├── run_full_pipeline.py              # Pipeline orchestrator
├── Dockerfile
├── docker-compose.yml
└── .env                              # API keys (gitignored)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (the live setup uses one 24 GB card)
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

### Chat with Your Bot

```bash
docker compose --profile dev up -d kaya-dev   # dev web UI on :7861
```

The `knowledge_approach` in `config.yaml` controls what knowledge is injected:
- `"both"` — JSON profiles + ChromaDB KB
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

### 3. **Build Vector Database** (`build_vector_db.py`)
- Builds `kaya_conversations` ChromaDB collection from message chunks
- Builds `kaya_knowledge_base` ChromaDB collection from `group_knowledge.json`
- Uses BAAI/bge-m3 embeddings

**Output:** `data/rag_db/` (ChromaDB)

## ⚙️ Configuration

All settings live in [config.yaml](config.yaml). Key sections:

### Pipeline Mode

```yaml
pipeline:
  generate_knowledge: false  # true = run knowledge extraction on the local teacher
```

### Knowledge Approach (Benchmarking)

```yaml
rag:
  knowledge_approach: "json_only"   # what the live config uses
  # "both"          — JSON members in system prompt + ChromaDB KB retrieval
  # "json_only"     — JSON injection only
  # "chromadb_only" — ChromaDB KB retrieval only
  # "none"          — Baseline (conversation history only)
```

### Model & Training

Model and training settings come from the **active model profile** (`active_model_profile` in `config.yaml`, currently `gemma4-12b-stock`), deep-merged over the top-level `model:`/`training:` sections by `src/config_loader.py`. The live profile has **no adapter** — it is a stock model served from a GGUF:

```yaml
active_model_profile: "gemma4-12b-stock"
```

### Inference Backend

```yaml
inference:
  backend: "gguf"   # llama.cpp server; prod sets KAYA_INFERENCE_BACKEND=ollama
                    # (Ollama behind the GPU broker). There is no in-process backend.
```

`KAYA_INFERENCE_BACKEND` overrides this; `KAYA_LLAMA_URL` overrides the server address.

### Who maintains the bot

The bot cannot change its own code, so a technical complaint has to be pointed at
a person — and that person is in the group, which the rule naming him did not
allow for. Asked by its own maintainer why a feature was bad, the bot answered
that the maintainer had to deal with it.

```yaml
chat:
  maintainer: "Gustavo"
  maintainer_clause: "...diz que o {maintainer} tem de tratar disso..."
  maintainer_self_clause: "...quem está a escrever agora é o {maintainer}..."
```

`{maintainer_clause}` is a placeholder in three prompts (`data.system_prompt`,
`chat.modes.banter`, `chat.modes.mixed`), filled per turn by
`engine.apply_speaker_rules` with whichever form matches the speaker.

### Variety, and where it must not apply

```yaml
rag:
  max_facts_per_member: 4    # a factual answer: the same facts every time
  max_facts_open_ended: 3    # a roast or an opinion: a different draw every time
inference:
  variety_scan_interactions: 400   # how far back to look for what was already said
  variety_lines_per_member: 3      # how many of those lines to show the model
```

An open-ended turn (`banter`, `mixed`, `roast`) rebuilds the system prompt with a
random handful of each member's facts and is shown what it already said about
whoever it is about. `factual` does neither: sampling would make "o que faz o
Gil?" depend on whether his job survived the draw, and a counting question must
return the same number every time it is asked.

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
```

### Run the Test Suite
```bash
kaya_chatbot_env/bin/python -m pytest tests/ -v
```

### End-to-End: one capability at a time
```bash
kaya_chatbot_env/bin/python scripts/preflight_e2e.py
```

### End-to-End: a whole evening in the group

The unit suite proves the wiring; this reproduces several people talking over
each other, a photo arriving mid-argument, and a thread long enough to overflow
the retrieval budget. It POSTs synthetic WAHA events at a `kaya-sim` container
running the **real** webhook path in mock mode — only the outbound WhatsApp
client is faked — so routing, the GPU lock, scoping and media handling are the
production ones.

```bash
kaya_chatbot_env/bin/python scripts/seed_sim_data.py     # once — builds ./data_sim
docker compose --profile sim up -d kaya-sim

kaya_chatbot_env/bin/python scripts/run_conversation_sim.py --preset smoke      # ~40s
kaya_chatbot_env/bin/python scripts/run_conversation_sim.py --preset standard   # ~10min, full surface
kaya_chatbot_env/bin/python scripts/run_conversation_sim.py --preset long_haul  # ~28min, overflows the budget
```

`kaya-sim` reads `./data_sim`, never `./data` — invented conversations must not
end up in the real vector store. Reports land in `reports/sim/<stamp>/index.html`
and the run exits non-zero on a failed assertion, so it can gate a deploy.

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
- Set `knowledge_approach: "both"` for both sources at once (live config uses `json_only`, which scored best)
- Set `knowledge_approach: "none"` to measure baseline performance without any knowledge injection

### Inference
- On the `gguf` backend the app process holds only the tokenizer and the retriever (~2 GB); the weights live in the `llama` container
- Typical latencies: banter ~5 s, a question that searches memory 7–13 s, describing a photo ~4 s
- Generation is serialized by a GPU lock, so four people asking at once queue (8–35 s) rather than drop
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
docker compose --profile prod up -d kaya-prod      # prod web app (scripts/deploy_prod.sh wraps this)
docker-compose up --build                          # the data pipeline, NOT the web app
docker compose --profile dev up -d kaya-dev        # dev UI on :7861
docker compose --profile test run --rm kaya-test   # test suite in-container
docker compose --profile sim up -d kaya-sim        # conversation simulator target
```

Run one environment at a time — a served model may claim both GPUs.

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

### Model Not Loading
- Ensure models are downloaded to `kaya_chatbot_env/`
- Check disk space (models are ~5GB each)
- Verify HuggingFace access token if using gated models

## 🤝 Contributing

This is a personal project, but suggestions welcome! Open issues for bugs or feature requests.

## 📄 License

Private project - All rights reserved.

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for model hosting and transformers

---

**Note:** This bot is trained on personal group chat data. Its "memory" of past events and people comes entirely from that history. The bot is NOT a group member — it is an AI assistant with access to the group's collective memory.
