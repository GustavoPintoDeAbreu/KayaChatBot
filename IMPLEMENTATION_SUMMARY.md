# RAG Chatbot Full Fix - Implementation Summary

## 🎯 Goal
Transform the chatbot from a "casual group member" to a "knowledgeable assistant" that can:
1. Answer factual questions using RAG-retrieved context
2. Also support casual conversation when appropriate  
3. Know the difference between Q&A and casual chat

## 📋 Changes Implemented

### 1. ✅ Upgraded Embedding Model

**Changed:** `sentence-transformers/all-MiniLM-L6-v2` → `Alibaba-NLP/gte-multilingual-base`

**Why:**
- Better Portuguese performance (52.3 MIRACL score vs generic multilingual)
- 8192 token context (vs 512) - handles longer messages
- Modern 2024 architecture optimized for RAG
- 305M params, ~580MB model size

**Files Modified:**
- [config.yaml](config.yaml#L144-L150) - Updated `rag.embedding_model`
- [config.docker.yaml](config.docker.yaml#L130-L136) - Updated `rag.embedding_model`
- [src/data/build_vector_db.py](src/data/build_vector_db.py#L219-L223) - Added `trust_remote_code=True`
- [src/chat/retriever.py](src/chat/retriever.py#L62-L63) - Added `trust_remote_code=True`

---

### 2. ✅ Fixed Prompt Engineering in Chat

**Problem:** Model told to "Continue the conversation" even when asked questions.

**Solution:** Detect question vs casual message, use different prompt formats.

**Files Modified:**
- [src/chat/chat.py](src/chat/chat.py#L103-L145)

**Changes:**
```python
# Question detection
is_question = any(keyword in user_input.lower() for keyword in [
    'o que', 'como', 'quando', 'onde', 'quem', 'porque', ...
])

# Q&A mode with RAG
if is_question and context:
    user_message = f"{context}\n\nCom base nestas conversas passadas, responde:\n{user_input}"

# Casual conversation mode
else:
    user_message = f"Conversa recente:\n{recent_history}\n\n{user_name}: {user_input}"
```

**Before:** 
```
User message: "Continue the conversation"
```

**After:**
```
Q&A: "Com base nestas conversas passadas, responde: {question}"
Casual: "Conversa recente: {...} User: {message}"
```

---

### 3. ✅ Updated System Prompts

**Changed:** From "You are a member of a Portuguese WhatsApp group" → Dual-mode assistant

**New Prompt (Portuguese):**
```
És um assistente que conhece bem um grupo de WhatsApp português. 
Quando te dão histórico de conversas passadas, responde às perguntas com base no que foi dito. 
Também podes conversar casualmente como membro do grupo. 
Fala português europeu informal com calão. Usa mensagens curtas estilo WhatsApp.
```

**Files Modified:**
- [config.yaml](config.yaml#L75-L76)
- [config.docker.yaml](config.docker.yaml#L61-L62)

**Why:** Explicitly instructs model it can do BOTH Q&A (using provided context) and casual conversation.

---

### 4. ✅ Improved RAG Context Formatting

**Problem:** Context was unstructured, model didn't understand it was reference material.

**Solution:** Add clear delimiters and numbering.

**Files Modified:**
- [src/chat/retriever.py](src/chat/retriever.py#L154-L177)

**Before:**
```
[Relevant past conversations:]

[2024-04-15 14:30] Peter: texto...
Gil: resposta...
```

**After:**
```
=== Conversas relevantes do grupo ===

--- Conversa 1 [2024-04-15] ---
Peter: texto...
Gil: resposta...

--- Conversa 2 [2024-04-16] ---
David: outro texto...

=== Fim das conversas ===
```

---

### 5. ✅ Updated Synthetic Data Generation for RAG Format

**Problem:** Training data had Q&A from memory, not from provided context. Model never learned to use RAG context.

**Solution:** Generate training examples that include RAG context format.

**Files Modified:**
- [src/data/generate_synthetic_data.py](src/data/generate_synthetic_data.py#L28-L90)

**New Training Format:**
```json
{
  "turns": [
    {
      "role": "user",
      "content": "=== Conversas relevantes do grupo ===\n\n--- Conversa 1 ---\n[relevant snippet]\n\n=== Fim das conversas ===\n\nCom base nestas conversas passadas, responde:\nQuem é o Peter?"
    },
    {
      "role": "assistant",
      "content": "O Peter é fixe ahahah, sempre a organizar cenas."
    }
  ]
}
```

**Key Change:** FIRST user message in each conversation now includes RAG context format, teaching the model to use provided information.

---

### 6. ✅ Removed General Portuguese Alpaca Data

**Problem:** General Portuguese Q&A data (not RAG-aware) would confuse the model.

**Solution:** Train exclusively on RAG-formatted Kaya data.

**Files Modified:**
- [src/data/merge_datasets.py](src/data/merge_datasets.py#L12-L30)
- [src/kaya_chatbot/data.py](src/kaya_chatbot/data.py#L626-L668)

**Changes:**
```python
merger = SyntheticDatasetMerger(
    kaya_file=f"{data_dir}/synthetic_kaya.jsonl",
    portuguese_file=None,  # REMOVED: Not RAG-aware
    kaya_ratio=1.0  # 100% RAG-aware Kaya Q&A data
)
```

**Why:** Every training example must include RAG context format to teach the model this new behavior.

---

## 🔄 Pipeline Updates Required

### Step 1: Rebuild Vector Database ✅ (In Progress)
```bash
docker-compose run --rm kaya-chatbot python src/data/build_vector_db.py
```
- Deletes old embeddings
- Recreates with Alibaba-NLP/gte-multilingual-base
- 1750 chunks with better Portuguese semantic understanding

### Step 2: Generate New RAG-Aware Training Data
```bash
docker-compose run --rm kaya-chatbot python src/data/generate_synthetic_data.py
```
- Uses updated prompts that include RAG context format
- Each Q&A example will have conversation history embedded
- Teaches model to answer based on provided context

### Step 3: Merge Datasets  
```bash
docker-compose run --rm kaya-chatbot python src/data/merge_datasets.py
```
- 100% RAG-aware Kaya data
- No general Portuguese data (would dilute RAG training)

### Step 4: Retrain Model
```bash
docker-compose run --rm kaya-chatbot python src/finetuning/train.py
```
- Fine-tune on RAG-formatted examples
- Model learns to use provided context in responses
- Learns difference between Q&A and casual conversation

### Step 5: Test
```bash
docker-compose run --rm kaya-chatbot python src/chat/chat.py
```
Test both modes:
- Q&A: "O que é que o grupo acha do ventura?" → Should use RAG context
- Casual: "olá pessoal" → Should respond conversationally

---

## 📊 Expected Improvements

### Before Fix
```
User: O que é que o grupo acha do ventura?
📚 Retrieved 5 chunks (not used)
Kaya: O Gil faltou? Ninguém quer ir?  ❌ Ignored context, random response
```

### After Fix
```
User: O que é que o grupo acha do ventura?
   [Mode: Q&A with RAG]
📚 Retrieved 5 chunks
Kaya: Baseado nas conversas, o Peter disse que não gosta do Ventura, 
      e o Gil acha que é maluco lmao  ✅ Uses retrieved context!
```

---

## 🔑 Key Technical Decisions

1. **Dual-mode detection** - Use keyword matching (`o que`, `como`, `?`) to detect questions vs casual messages

2. **100% RAG training** - No mixing with general Portuguese data to ensure consistent RAG behavior

3. **Context in FIRST turn only** - Follow-up questions don't need RAG format (natural conversation flow)

4. **GTE embedding model** - Best Portuguese performance, 8192 context for long messages

5. **Explicit instructions** - "Com base nestas conversas passadas, responde:" tells model exactly what to do

---

## 📁 Files Changed Summary

### Configuration
- `config.yaml` - Embedding model + system prompt
- `config.docker.yaml` - Embedding model + system prompt

### RAG System
- `src/data/build_vector_db.py` - trust_remote_code for GTE
- `src/chat/retriever.py` - Better context formatting + trust_remote_code
- `src/chat/chat.py` - Question detection + dual-mode prompting

### Training Data Generation  
- `src/data/generate_synthetic_data.py` - RAG-aware format
- `src/data/merge_datasets.py` - Remove Portuguese alpaca data
- `src/kaya_chatbot/data.py` - Handle None portuguese_file

---

## 🚀 Next Steps

1. ✅ Vector DB rebuild (in progress)
2. ⏳ Generate RAG-aware training data (~15-30 min)
3. ⏳ Merge datasets
4. ⏳ Retrain model (~2-3 hours on RTX 3090)
5. ⏳ Test both Q&A and casual conversation modes

---

## 💡 Why This Will Work

The root cause was **training/inference mismatch**:
- Training: Q&A from memory
- Inference: Q&A with provided context (RAG)
- Model: Ignores context because never trained to use it

The fix:
- Training: Q&A with provided context (RAG format)
- Inference: Q&A with provided context (RAG format)
- Model: Uses context because that's what it was trained on ✅

Plus we added:
- Better Portuguese embeddings (GTE)
- Dual-mode support (Q&A vs casual)
- Clear mode detection and prompting
