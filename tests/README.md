# Tests Directory

Organized test suite for KayaChatBot.

## Structure

### `rag/` - RAG System Tests
- **test_core_logic.py** - Core RAG logic without model downloads (fast, unit tests)
- **test_retrieval.py** - Quick RAG retrieval validation
- **test_vector_db.py** - Vector database functionality tests
- **test_rag_full.py** - Full end-to-end RAG integration test

### `pipeline/` - Data Pipeline Tests
- **test_pipeline.py** - Full data processing pipeline
- **test_llm_cleaning.py** - LLM-based data cleaning tests
- **validate_pipeline.py** - Pipeline validation and sanity checks

## Running Tests

### RAG Tests
```bash
# Quick core logic test (no downloads, ~10 seconds)
docker-compose run --rm kaya-chatbot python tests/rag/test_core_logic.py

# Quick retrieval test (~30 seconds)
docker-compose run --rm kaya-chatbot python tests/rag/test_retrieval.py

# Vector database test
docker-compose run --rm kaya-chatbot python tests/rag/test_vector_db.py

# Full RAG test (slower, downloads models)
docker-compose run --rm kaya-chatbot python tests/rag/test_rag_full.py
```

### Pipeline Tests
```bash
# Data pipeline
docker-compose run --rm kaya-chatbot python tests/pipeline/test_pipeline.py

# LLM cleaning
docker-compose run --rm kaya-chatbot python tests/pipeline/test_llm_cleaning.py

# Validation
docker-compose run --rm kaya-chatbot python tests/pipeline/validate_pipeline.py
```

## Test Descriptions

### RAG Tests

**test_core_logic.py** - Tests chunking, person extraction, and context formatting without downloading embedding models. Fastest test for development.

**test_retrieval.py** - Validates RAG retrieval with actual vector database. Tests semantic search, person filtering, and context formatting.

**test_vector_db.py** - Tests vector database operations, embedding generation, and ChromaDB functionality.

**test_rag_full.py** - Complete end-to-end RAG test including model loading, retrieval, and chat integration.

### Pipeline Tests

**test_pipeline.py** - Tests the complete data processing pipeline from raw messages to training data.

**test_llm_cleaning.py** - Tests LLM-based message filtering and cleaning.

**validate_pipeline.py** - Validates data quality, format consistency, and pipeline integrity.
