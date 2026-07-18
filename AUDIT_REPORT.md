# KayaChatBot — Architectural Audit Report

**Auditor role:** Senior Staff AI Engineer (read-only review)
**Date:** 2026-05-28
**Scope:** Full repository — config system, RAG pipeline, ChromaDB interactions, local inference/memory, fine-tuning pipeline, data extraction, LLM providers, and infrastructure (Docker, dependencies).
**Method:** Static review of source only. No files were modified and nothing was executed.

> **Headline:** The application logic is generally well-structured and thoughtfully commented, but there are two **critical** problems that undermine the project's own stated guarantees: (1) the Docker image installs a dependency set that **contradicts the project's hard version constraints** and the local `requirements.txt`, and (2) **real people's sensitive personal data (including political preference) is committed to git**, contradicting the careful sensitivity handling elsewhere in the code. There is also a cluster of RAG-quality and model-agnosticism bugs that silently degrade output.

---

## 1. Executive Summary

| Area | Health | Notes |
|---|---|---|
| Code quality | 🟡 Fair | Clear structure, good docstrings; undermined by duplicated logic, dead code, and 3 competing token-estimation heuristics. |
| Security / Privacy | 🔴 Poor | PII (incl. political preference) committed to git; uncensored bot served on `0.0.0.0` with no auth. |
| Edge-case handling | 🟡 Fair | Many guarded paths, but several latent crashes (`UnboundLocalError` in providers) and silent-failure bare `except:` blocks. |
| Performance (RAG/Chroma/memory) | 🟡 Fair | Wrong distance metric for the embedding model; duplicate query embeddings; per-query `count()`; O(n²) budget loop. |
| Technical debt | 🟠 Elevated | Docker/req divergence, dead `config.docker.yaml` duplicate, unused validation models, "multi-profile" config that only works for Gemma 4. |

### Issue count by priority
- **P0 (Critical):** 2
- **P1 (High):** 4
- **P2 (Medium):** 6
- **P3 (Low):** 7

---

## 2. Priority-Ranked Findings

Each finding is tagged with the relevant audit category: `[SECURITY]` `[QUALITY]` `[EDGE]` `[PERF]` `[DEBT]`.

---

### 🔴 P0-1 — Docker image dependencies contradict `requirements.txt` *and* the project's own hard constraints `[DEBT][QUALITY]`

**Evidence:**
- `Dockerfile:52-84` installs `transformers==4.57.6`, `trl==0.29.1`, `peft==0.15.2`, `unsloth==2026.4.2`, `unsloth-zoo==2026.4.2`, on **Python 3.10**.
- `requirements.txt:14-24` pins `transformers==5.5.0`, `trl==0.24.0`, `peft==0.19.0`, `unsloth==2026.4.5`, `unsloth-zoo==2026.4.5`.
- `CLAUDE.md` declares hard rules: `trl<=0.24.0` ("newer versions break `SFTConfig` API"), `unsloth>=2026.4.5` ("required for Gemma 4 via `FastModel`"), `transformers>=5.5.0` ("required for `Gemma4ForConditionalGeneration`").
- `Dockerfile:48` copies `requirements.txt` but **never runs `pip install -r requirements.txt`** — the inline `pip install` lines are the real source of truth and they diverge.
- `gradio` (required by `src/chat/web_app.py`, and pinned in `requirements.txt:50`) is **not installed in the Dockerfile at all**.
- The venv is Python **3.12** (`CLAUDE.md` PEFT-patch paths reference `python3.12`); the Docker image is Python **3.10**.

**Impact:**
- The Docker image violates every documented version constraint. By the project's own notes, `trl==0.29.1` breaks the `SFTConfig` API and `transformers==4.57.6` lacks `Gemma4ForConditionalGeneration` — so **containerized training/inference of the active Gemma-4 profile is expected to fail or behave incorrectly**.
- The web UI cannot run in Docker (no `gradio`).
- "Works on my machine vs. in Docker" divergence is guaranteed; the documented PEFT `float8_e8m0fnu` patch targets the 3.12 venv path and won't apply in the 3.10 image.

**Proposed fix:**
1. Make the Dockerfile the single consumer of `requirements.txt`: replace the inline `pip install transformers==... trl==...` blocks with `RUN pip install --no-cache-dir -r requirements.txt` (keep the separate torch/CUDA index-url install for the pinned wheels).
2. Align the Python base image to 3.12 to match the venv and the documented PEFT patch paths.
3. Add `gradio==6.15.1` to the install set and add a `kaya-web` compose service (see P1-4).
4. Add a CI smoke step (`python -c "import torch, transformers, trl, peft, unsloth"` + a 1-step train) so divergence is caught automatically.

---

### 🔴 P0-2 — Sensitive personal data of real individuals is committed to git `[SECURITY]`

**Evidence:**
- `.gitignore:35-36` force-tracks the data files: `!data/group_members.json` and `!data/group_knowledge.json`.
- `git ls-files` confirms both are tracked.
- `src/models.py:244-285` (`MemberProfile`) and `config.yaml:128-140` show these files contain `political_preference`, `state_of_mind`, `marital_status`, `age`, `living_place`, `occupation`, and `biography_summary` for named real people.
- The architecture goes to great lengths to keep `political_preference` *out of ChromaDB* (`models.py:274-285` `to_public_dict()`, `CLAUDE.md` coding conventions) — yet the **raw JSON containing that exact field is committed to version control**, which is a far larger exposure surface (clones, forks, history, GitHub indexing).

**Impact:**
- Real, identifiable individuals' political affiliation, mental-state notes, and relationship status are stored in git history. Even if the repo is private today, this is irreversible once pushed/forked and is a serious privacy/GDPR-style concern for a "private friend group" tool.
- The protection invariant ("never embed `political_preference`") is rendered moot by the git exposure.

**Proposed fix:**
1. Stop tracking the raw profile files: remove the `!data/group_members.json` / `!data/group_knowledge.json` exceptions and instead commit **schema-only** sanitized templates (e.g. `data/group_members.example.json` with empty/placeholder values).
2. Purge the sensitive files from history (`git filter-repo` / BFG) and rotate the repo if it was ever pushed publicly.
3. If the files must be tracked for reproducibility, split sensitive fields (`political_preference`, `state_of_mind`, any `*health*`) into a separate **git-ignored** sidecar file loaded at runtime, mirroring the existing `to_public_dict()` split.
4. Audit `data/golden_test_conversations.json` (also force-tracked, `.gitignore:33`) for verbatim private quotes.

---

### 🟠 P1-3 — RAG retrieval uses the wrong distance metric for the embedding model `[PERF][QUALITY]`

**Evidence:**
- Embedding model is `BAAI/bge-m3` (`config.yaml:217`), which is trained for **cosine** similarity and produces **1024-dim** vectors.
- Stored embeddings are **not normalized**: `build_vector_db.py:303` calls `self.encoder.encode(batch_docs, ...)` with no `normalize_embeddings=True`. Query embeddings are likewise un-normalized: `retriever.py:133` and `retriever.py:221`.
- The collection is created **without** specifying a space: `build_vector_db.py:269-276` passes only descriptive metadata, so ChromaDB defaults to **L2 (squared Euclidean)**.
- `retriever.py:171` computes `similarity_score = 1 - distance`. With un-normalized vectors under L2, `distance` is unbounded, so `similarity_score` can be **negative and meaningless**.
- Inconsistency tell: the dimension *probe* on `build_vector_db.py:228` uses `normalize_embeddings=True`, but the real ingestion path does not.
- The hardcoded log `build_vector_db.py:244` prints `Dimension check passed (768)` — wrong for bge-m3 (1024), indicating the check is stale.

**Impact:**
- Retrieval ranking is computed in a geometry the embeddings were not designed for, degrading the relevance of every retrieved chunk — directly weakening the project's core "always-on RAG" guarantee.
- Downstream consumers that reason about `similarity_score` get values outside `[0, 1]`.

**Proposed fix:**
1. Create both collections with cosine space: `create_collection(..., metadata={"hnsw:space": "cosine"})` in `build_vector_db.py` for `kaya_conversations` and `kaya_knowledge_base`.
2. Normalize on both write and read paths (`normalize_embeddings=True` in all `encoder.encode(...)` calls) so `1 - distance` is a valid cosine similarity.
3. Add a minimum-similarity threshold in `retriever.retrieve()` so off-topic queries (always-on RAG) don't inject the top-k *least-irrelevant* chunks regardless of quality.
4. Fix the hardcoded "768" log line to print the actual `embedding_dim`.

---

### 🟠 P1-4 — Chat responses are truncated to a single line `[QUALITY][EDGE]`

**Evidence:**
- `chat.py:263`: `response_text = response_text.split('\n')[0].replace(f"{user_name}:", "")`.
- The `TextStreamer` (`chat.py:242`) prints the **full** multi-line generation to the terminal, but only the **first line** is saved to history (`chat.py:266`) and written to the interaction log (`chat.py:275-276`).
- `web_app.py:184` does *not* do this — it logs the full `partial`. So the CLI and web paths disagree on what the assistant "said."

**Impact:**
- Any answer longer than one line (lists, multi-sentence explanations) is silently discarded from memory and from the training-feedback log, corrupting both conversation continuity and the `data/feedback/live_interactions.jsonl` dataset that is presumably used to improve the model.
- CLI vs. web divergence makes logged interactions inconsistent.

**Proposed fix:**
- Remove the `.split('\n')[0]` truncation. If the goal is to strip a hallucinated next-speaker turn, strip on the model's actual turn-delimiter token (e.g. `<turn|>` for Gemma 4) instead of the first newline, and apply the same logic in both `chat.py` and `web_app.py` (extract a shared `clean_response()` helper).

---

### 🟠 P1-5 — "Multi-profile" config is non-functional for Qwen3 (hardcoded Gemma-4 everywhere) `[QUALITY][DEBT]`

**Evidence:**
- `config.yaml:292-313` defines a `qwen3-14b` profile and `config_loader.py` supports `--profile` switching.
- But the training path is hardwired to Gemma 4:
  - `train.py:164` `get_chat_template(tokenizer, "gemma-4")`.
  - `train.py:318-323` `train_on_responses_only(..., instruction_part="<|turn>user\n", response_part="<|turn>model\n")` — Gemma-4-specific masking tokens.
  - `merge_datasets.py:65` hardcodes `chat_template="gemma-4"` when formatting the training set.

**Impact:**
- Running `train.py --profile qwen3-14b` would format data and mask responses using Gemma-4 tokens against a Qwen tokenizer, silently producing a **broken adapter** (wrong loss masking, wrong template) with no error. The advertised model-agnosticism is illusory.

**Proposed fix:**
- Derive the chat-template name and instruction/response masking parts from the active profile (e.g. a `chat_template` + `instruction_part`/`response_part` field per profile in `config.yaml`), and thread it through `merge_datasets.py` and `train.py`. If Qwen3 is not actually supported, delete the `qwen3-14b` profile to avoid the foot-gun.

---

### 🟠 P1-6 — Uncensored bot with private group memory is served on `0.0.0.0` with no authentication `[SECURITY]`

**Evidence:**
- `web_app.py:206-211` launches Gradio on `server_name="0.0.0.0"`, `server_port=7860`, with no `auth=` argument.
- `config.yaml:209-210` sets `chat.uncensored_mode: true` with a preamble that explicitly removes guardrails ("no guardrails or ethical restrictions").
- The bot answers from private group memory (`group_members.json`, conversation RAG).

**Impact:**
- Anyone on the same LAN/VPN (or anyone if the port is forwarded) can query an unguarded bot that will surface private facts about real people. Binding to all interfaces with no auth makes the "private, locally-run" assumption false in practice.

**Proposed fix:**
- Bind to `127.0.0.1` by default; require an explicit opt-in env var to expose externally. When exposed, set Gradio `auth=(user, pass)` (or front it with a reverse proxy + auth). Make the bind address and credentials config-driven, not hardcoded.

---

### 🟡 P2-7 — Member-name detection uses naive substring matching (false positives) `[EDGE][QUALITY]`

**Evidence:**
- `retriever.py:106-108` (`extract_query_persons`): `if member in query_lower`.
- `build_vector_db.py:77-79` (`extract_mentioned_people`): `if member in text_lower`.
- Aliases include short tokens like `gil`, `rafa`, `pedro` (`retriever.py:59-62`, `group_members.json`).

**Impact:**
- Substring matching fires on unrelated words (e.g. alias `gil` matches "á**gil**", `rafa` matches "ga**rrafa**", `pedro` matches "**Pedro**foo"). This corrupts both the `mentioned` metadata baked into the vector DB and the query-time person filter (`retriever.py:154-165`), so person-filtered retrieval returns wrong chunks or wrongly excludes good ones.

**Proposed fix:**
- Match on word boundaries: tokenize and compare whole tokens (reuse the token approach already proven in `identity_resolver.py:117-121`), or use `re.search(rf"\b{re.escape(member)}\b", text_lower)`.

---

### 🟡 P2-8 — Training-time and inference-time RAG context formats diverge (train/serve skew) `[QUALITY]`

**Evidence:**
- Training context lines: `format_direct_training.py:151` → `[{sender}]: {text}` inside `=== Conversas relevantes do grupo ===`, with a `"Com base nestas conversas passadas, responde:"` preamble (`:197-201`).
- Inference context lines: `retriever.py:205-206` → `{sender}: {text}` (no brackets) grouped under `--- Conversa N [date] ---` headers; assembled in `chat.py:209-219` with a different `Conversa recente:` block and no "responde" preamble.

**Impact:**
- The model is trained on one context shape but served another, weakening grounding — the model has to generalize across a formatting gap that didn't need to exist. This likely contributes to the low benchmark scores noted in `config.yaml:225-229`.

**Proposed fix:**
- Define one canonical context-formatting function and import it in both `format_direct_training.py` and `retriever.py`/`chat.py` so training and inference are byte-identical in structure.

---

### 🟡 P2-9 — `config.docker.yaml` is a hand-maintained duplicate that has already drifted `[DEBT][EDGE]`

**Evidence:**
- `docker-compose.yml:29` mounts `./config.docker.yaml` as `/app/config.yaml`; `Dockerfile:99` bakes it in too.
- Comparing the two: `config.yaml:37` defines `azure.api_key_env` and `azure.extraction_temperature:41`, but `config.docker.yaml`'s azure block (`:35-41`) **omits both**.

**Impact:**
- Docker silently uses different generation settings than local (e.g. falls back to the generic `AZURE_OPENAI_API_KEY` and a default extraction temperature). Two sources of truth guarantee ongoing drift; the most security-relevant key (`uncensored_mode`) could also diverge unnoticed.

**Proposed fix:**
- Eliminate the duplicate. Keep a single `config.yaml` and override only the handful of Docker-specific paths via environment variables or a small `config.docker.overlay.yaml` that is deep-merged by the existing `config_loader._deep_merge`. The merge machinery already exists — reuse it instead of copy-pasting the whole file.

---

### 🟡 P2-10 — Dead code and three competing token-estimation heuristics `[DEBT][QUALITY]`

**Evidence:**
- `src/chat/context_injection.py` (`estimate_tokens`, `build_recent_summaries`, `truncate_to_budget`) is imported **only by its own test** (`tests/rag/test_context_injection.py`); the live retriever reimplements all of it inline (`retriever.py:263-272`, `:274-291`, `:333-342`).
- Three different token estimators coexist: `context_injection.py:14` uses **chars/4**; `retriever.py:272` uses **words/0.60**; `config.yaml:231` and `CLAUDE.md` document **words/0.75**.
- `src/models.py:330-336` defines `ConfigModel` with the docstring "Use: `ConfigModel(**config)` at startup" — but `grep` shows it is **never instantiated**; `config_loader.load_config` performs no validation.

**Impact:**
- Maintenance confusion and silent inconsistency: the token budget (`max_context_tokens: 3000`) means different things depending on which estimator is consulted, and the documented value (0.75) matches neither live path.
- Config typos (missing `training.output_dir`, malformed `rag`) surface as deep `KeyError`s at runtime instead of a clear validation error, despite validation models already being written.

**Proposed fix:**
1. Delete `context_injection.py` (and its test) or make the retriever import from it — pick one implementation and standardize the divisor; update `config.yaml`/`CLAUDE.md` to match.
2. Wire `ConfigModel(**config)` into `config_loader.load_config()` so config is validated once at load.

---

### 🟡 P2-11 — Import-time side effects + non-thread-safe retriever singleton `[QUALITY][PERF][EDGE]`

**Evidence:**
- `retriever.py:14-27` loads `config.yaml` and derives module-level constants (`EMBEDDING_MODEL`, `TOP_K`, `FILTER_BY_PERSON`) **at import time**; `build_vector_db.py:16-50` also loads config and calls `DB_DIR.mkdir(...)` at import.
- The `ConversationRetriever` instance receives a `config` argument but still reads some values from the import-time module globals (`retriever.py:130`, `:133`, `:90`), so passing a different config doesn't fully take effect.
- `get_retriever()` (`retriever.py:374-380`) is a process-global singleton with no lock; if the first caller's config differs from a later caller's, the later config is ignored. `web_app.py` serves concurrent Gradio requests against this shared, lock-free singleton and a shared `model`.

**Impact:**
- Hard to test (importing the module requires a valid `config.yaml` on disk).
- Concurrent web requests share one model + one retriever with no synchronization; overlapping `model.generate` calls can serialize unpredictably or contend on the GPU.

**Proposed fix:**
- Move config loading and `mkdir` into functions / class `__init__`, not module top-level. Read all tunables from `self.rag_config` consistently. Guard the singleton with a lock, and serialize generation (a queue or lock) in `web_app.py`, or document max-concurrency=1 via `demo.queue(default_concurrency_limit=1)`.

---

### 🟡 P2-12 — `UnboundLocalError` in both LLM providers when a conversation dict lacks expected keys `[EDGE]`

**Evidence:**
- `azure_provider.py:194-205` and `xai_provider.py:143-156`: `turns` is only assigned inside the `if 'turns' in conv / elif 'conversation' / elif 'messages'` branches. If `conv` is a dict with none of those keys (or neither dict nor list), `turns` is referenced (`if not turns`) **before assignment** → `UnboundLocalError`.
- The two `_parse_response` implementations are near-identical copies (DRY violation).

**Impact:**
- A malformed LLM JSON response crashes the parser with a confusing `UnboundLocalError` instead of being skipped, aborting a long/expensive generation run.

**Proposed fix:**
- Initialize `turns = None` before the branches and `continue` when it stays `None`. Hoist the shared `_parse_response` into `base.LLMProvider` to remove the duplication.

---

### 🟢 P3-13 — Bare `except:` blocks swallow all exceptions `[QUALITY][EDGE]`

**Evidence:** `xai_provider.py:128`, `azure_provider.py:181`, `retriever.py:202`, `build_vector_db.py:265`, `extract_all_messages.py:321`, `readers.py:216`, `readers.py:337`.

**Impact:** Catches `KeyboardInterrupt`/`SystemExit` and masks real bugs (e.g. the timestamp parse in `retriever.py:202` hides any formatting error).

**Proposed fix:** Replace with specific exceptions (`except (ValueError, KeyError):`, `except json.JSONDecodeError:`, etc.).

---

### 🟢 P3-14 — Redundant embedding + DB calls per query `[PERF]`

**Evidence:**
- `retrieve_all()` embeds the query twice: once in `retrieve()` (`retriever.py:133`) and again in `retrieve_knowledge()` (`retriever.py:221`).
- `collection.count()` is called on every query (`retriever.py:142`, `:225`) — an extra DB round-trip per call.
- The token-budget loop in `retrieve_all()` (`retriever.py:333-336`) re-formats and re-counts **all** chunks on every `pop()` → O(n²) string building.

**Impact:** Avoidable per-message latency on the local inference path (embedding a query with bge-m3 is the dominant cost).

**Proposed fix:** Encode the query once and pass the vector into both retrieval calls; cache `count()` after `initialize()`; compute per-chunk token counts once and subtract on pop.

---

### 🟢 P3-15 — Timezone-naive timestamps and fragile lexicographic time comparison `[EDGE]`

**Evidence:**
- `extract_all_messages.py:286` uses `datetime.fromtimestamp(timestamp_ms/1000)` (machine-local tz, naive) for Instagram; WhatsApp uses `strptime` (also naive) and **falls back to the raw string** on parse failure (`:163-165`).
- `incremental_update.py:165` compares timestamps with `<` on **strings**: `msg["timestamp"] < last_timestamp`.

**Impact:** Local-timezone dependence makes extraction non-deterministic across machines; if a WhatsApp date ever falls back to the raw `"M/D/YY, HH:MM"` string, lexicographic comparison and sorting silently break the incremental dedupe/date filter.

**Proposed fix:** Parse to timezone-aware UTC datetimes everywhere; compare `datetime` objects, not strings; drop (or hard-fail) messages whose timestamp can't be parsed instead of storing the raw string.

---

### 🟢 P3-16 — Non-atomic session-history write `[EDGE]`

**Evidence:** `memory.py:67-70` writes `history_file` via `write_text(...)` directly.

**Impact:** A crash/interrupt mid-write leaves a truncated/corrupt JSON file; next load fails the `json.loads` and silently discards the whole session history (`memory.py:52-54`).

**Proposed fix:** Write to a temp file in the same directory and `os.replace()` into place (atomic on POSIX).

---

### 🟢 P3-17 — Duplicated member-prompt building + config files read without `load_config()` `[DEBT]`

**Evidence:**
- The member-profile system-prompt block is copy-pasted in `chat.py:59-83` and `web_app.py:70-87`.
- `extract_all_messages.py:18-19` and `incremental_update.py:47-48` read `config.yaml` via `yaml.safe_load` directly, violating `CLAUDE.md`'s rule: "All code paths must use `load_config()` — never read `config.yaml` directly."

**Impact:** Two copies of prompt logic drift apart; pipeline scripts bypass profile merging (works today only because they read profile-independent sections).

**Proposed fix:** Extract a shared `build_member_prompt(config)` helper used by both chat entry points; route the pipeline scripts through `load_config()`.

---

### 🟢 P3-18 — Training script ergonomics `[QUALITY]`

**Evidence:**
- `train.py:79` calls `input("Continue anyway? (y/n): ")` when no GPU is detected — **blocks forever in non-interactive/Docker** runs (the default compose `command` is `run_full_pipeline.py`).
- `train.py:282-283` passes both `dataset_text_field="formatted_text"` *and* a `formatting_func` to `SFTConfig`/`SFTTrainer` — redundant/ambiguous.

**Impact:** A headless training run on a misconfigured GPU host hangs instead of failing fast.

**Proposed fix:** Gate the prompt on `sys.stdin.isatty()` and otherwise exit non-zero; pick either `dataset_text_field` or `formatting_func`, not both.

---

### 🟢 P3-19 — Two virtualenv conventions in the tree `[DEBT]`

**Evidence:** `CLAUDE.md` and `run_full_pipeline.py:30` / `incremental_update.py:56` reference `kaya_chatbot_env/`, but the repo also contains a `.venv/` (only `pip` installed, per the file listing). `.gitignore` ignores both.

**Impact:** Confusion about which interpreter is authoritative; a stray `.venv` invites accidental use of an empty environment.

**Proposed fix:** Standardize on one venv name, delete the stray `.venv/`, and make the doc + scripts agree.

---

## 3. Findings by Requested Category

### 3.1 Code quality, security, and edge-case handling
- **Security/Privacy:** P0-2 (PII in git), P1-6 (open uncensored endpoint). These are the most important to address.
- **Edge cases / latent crashes:** P2-12 (`UnboundLocalError` in providers), P3-15 (timestamp parsing/comparison), P3-16 (non-atomic history write), P1-4 (response truncation drops data).
- **Quality:** P2-7 (substring matching), P2-8 (train/serve skew), P2-10 (dead code, 3 token heuristics), P3-13 (bare `except:`), P3-17/18.

### 3.2 Performance (RAG pipeline, ChromaDB, local-inference memory)
- **Retrieval correctness/speed:** P1-3 (L2 vs cosine + unnormalized vectors is the biggest quality lever), P3-14 (double embedding, per-query `count()`, O(n²) budget loop).
- **ChromaDB:** the dimension-probe path adds and deletes a throwaway document on every build (`build_vector_db.py:232-243`) and the collection space is never set (P1-3). Consider building the conversation collection once and reusing the loaded encoder for the KB build (`build_vector_db.py:339` reloads bge-m3 a second time — ~minutes and extra RAM).
- **Inference memory:** the `MemoryMonitorCallback` (`train.py:193-218`) and the documented RAM rules are sensible. Main runtime concern is the lock-free shared model under Gradio concurrency (P2-11).

### 3.3 Technical debt
- P0-1 (Docker/req divergence) is both a correctness and a debt issue and should be fixed first.
- P1-5 (Gemma-4-hardcoded "multi-profile"), P2-9 (`config.docker.yaml` duplicate), P2-10 (unused `ConfigModel`, dead `context_injection.py`), P3-17/19.

---

## 4. Suggested Remediation Order

1. **P0-2** — Stop committing PII; purge history. *(privacy, irreversible)*
2. **P0-1** — Make Docker install from `requirements.txt`; align Python to 3.12; add gradio. *(unblocks reproducible runs)*
3. **P1-3** — Switch ChromaDB to cosine + normalize embeddings + add similarity threshold. *(biggest RAG-quality win, low effort)*
4. **P1-4** — Remove first-line response truncation. *(one-line fix, restores data integrity)*
5. **P1-6** — Bind web UI to localhost + add auth.
6. **P1-5** — Make chat template profile-driven (or delete the Qwen3 profile).
7. **P2 cluster** — substring→word-boundary matching, unify train/serve context format, collapse the `config.docker.yaml` duplicate, delete dead code + wire `ConfigModel`, fix singleton/import side effects, fix provider `UnboundLocalError`.
8. **P3 cluster** — bare excepts, perf micro-fixes, UTC timestamps, atomic writes, dedupe prompt builders, venv cleanup.

---

## 5. What's Working Well (Keep)

- **Clear layering** — config / data / RAG / finetuning / providers / chat are cleanly separated; the `LLMProvider` abstraction with `_retry_with_backoff` (`base.py:37-62`) is a good pattern.
- **Pydantic models** (`models.py`) give strong typed contracts for each pipeline stage; the sensitive-field split in `to_public_dict()` is the right idea (it just needs to extend to git — P0-2).
- **Thoughtful sender resolution** — `identity_resolver.py` handles the genuinely hard multi-source name-matching problem with a sensible priority chain and ambiguity guard.
- **Reproducibility intent** — versions are pinned and rationale is documented in `CLAUDE.md`; the gap is enforcement (P0-1), not intent.
- **Good test surface** — `tests/rag`, `tests/pipeline`, `tests/testing` exist with unit + integration coverage; the memory-monitoring training callbacks show real operational awareness of the 24 GB VRAM constraint.

---

*End of report. No source files were modified during this audit; only `AUDIT_REPORT.md` was created.*

---

## 6. Remediation Status (updated 2026-07-18)

| Finding | Status | Notes |
|---|---|---|
| P0-1 Docker deps contradict requirements.txt | ✅ Fixed | Dockerfile now installs from `requirements.txt` (single source of truth); torch/CUDA wheels kept separate. |
| P0-2 PII committed to git | 🟠 Partial | Real `group_members.json` / `group_knowledge.json` / `golden_test_conversations.json` untracked; only `.example` templates tracked. **Git-history purge (filter-repo/BFG) still REQUIRED — pending owner approval.** |
| P1-3 Wrong distance metric | ✅ Fixed | Collections rebuilt with `hnsw:space=cosine`; embeddings normalized. |
| P1-4 Single-line response truncation | ✅ Fixed | `response_utils` keeps multi-line answers; old `split("\n")[0]` removed. |
| P1-5 Multi-profile config non-functional for Qwen3 | ✅ Fixed | Profiles carry `chat_template`/`instruction_part`/`response_part`; train + merge read them. |
| P1-6 Uncensored bot on 0.0.0.0 without auth | ✅ Fixed | Localhost-default bind; prod behind Cloudflare Access + Gradio auth (`KAYA_WEB_USER`/`KAYA_WEB_PASS`). |
| P2-7 Naive substring member matching | ✅ Fixed | Word-boundary regex in `build_vector_db.py` and retriever person extraction. |
| P2-8 Train/serve RAG format skew | ✅ Fixed | `format_direct_training.py` emits the same `=== Conversas relevantes do grupo ===` block as inference. |
| P2-9 `config.docker.yaml` drifted duplicate | ✅ Fixed | File deleted; single `config.yaml` used everywhere (stale references cleaned from tests/README 2026-07-18). |
| P2-10 Dead code / competing token heuristics | ✅ Fixed | Dead `to_instruction_chunks`, unused imports and dead config keys removed (2026-07-18 sweep). |
| P2-11 Import-time side effects + unsafe singleton | ✅ Fixed | `get_retriever()` is a double-checked-locking singleton. |
| P2-12 `UnboundLocalError` in providers | ✅ Fixed | Providers guard missing conversation keys. |
| P3-13 Bare `except:` blocks | ✅ Fixed | No bare `except:` remains in `src/`. |
| P3-14 Redundant embedding/DB calls | ✅ Fixed | Query embedded once and reused across conversation + KB search. |
| P3-17 Config read without `load_config()` | ✅ Fixed | All remaining raw `yaml.safe_load` call sites migrated to `src.config_loader.load_config` (2026-07-18). |
| P3-19 Two virtualenv conventions | ✅ Fixed | Stray bare `.venv/` removed (2026-07-18); `kaya_chatbot_env/` is the single venv. |

Open items: **P0-2 history purge** (destructive — owner must approve), P3-15/P3-16/P3-18 (low-priority ergonomics, unverified).
