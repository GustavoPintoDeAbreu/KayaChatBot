"""
Test script for RAG components.
Tests chunking, vector database, and retrieval functionality.
"""

import os
import sys
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load configuration
CONFIG_PATH = Path("/app/config.yaml")
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path("config.yaml")
print(f"Loading config from: {CONFIG_PATH}")
print(f"Config exists: {CONFIG_PATH.exists()}")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def test_rag_components():
    """Test all RAG components end-to-end."""
    print("=" * 60)
    print("🧪 RAG COMPONENTS TEST")
    print("=" * 60)

    # Test 1: Check if cleaned messages exist
    print("\n1️⃣  Testing data availability...")
    cleaned_file = Path("data/all_messages_cleaned.jsonl")
    if not cleaned_file.exists():
        print("❌ Cleaned messages file not found. Run extract_all_messages.py first!")
        return False

    print(f"✅ Found cleaned messages: {cleaned_file}")

    # Test 2: Test chunking
    print("\n2️⃣  Testing conversation chunking...")
    try:
        from src.data.build_vector_db import ConversationChunker, load_cleaned_messages

        # Load a small sample of messages
        messages = load_cleaned_messages(limit=100)  # Test with 100 messages

        chunker = ConversationChunker()
        chunks = chunker.create_conversation_chunks(messages)

        print(f"✅ Created {len(chunks)} chunks from {len(messages)} messages")

        # Show sample chunk
        if chunks:
            sample = chunks[0]
            print(f"   Sample chunk: {sample['message_count']} messages, {sample['token_count']} tokens")
            print(f"   Participants: {', '.join(sample['participants'])}")
            print(f"   Mentioned: {', '.join(sample['mentioned'])}")

    except Exception as e:
        print(f"❌ Chunking test failed: {e}")
        return False

    # Test 3: Test vector database building (mocked)
    print("\n3️⃣  Testing vector database logic...")
    try:
        # Test ChromaDB operations without SentenceTransformer
        import chromadb
        from pathlib import Path

        # Create a test database
        test_db_path = Path("data/test_rag_db")
        if test_db_path.exists():
            import shutil
            shutil.rmtree(test_db_path)

        # Create ChromaDB collection
        client = chromadb.PersistentClient(path=str(test_db_path))
        collection_name = "kaya_conversations"
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Kaya chatbot conversation chunks for RAG"}
        )

        # Mock embeddings (random vectors)
        import numpy as np
        def mock_encode(texts):
            return np.random.rand(len(texts), 384).tolist()

        # Add test chunks
        test_chunks = chunks[:3] if len(chunks) >= 3 else chunks
        
        ids = [chunk['id'] for chunk in test_chunks]
        documents = [chunk['text'] for chunk in test_chunks]
        metadatas = [chunk['metadata'] for chunk in test_chunks]
        embeddings = mock_encode(documents)

        # Add to collection
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        # Test query
        query_embedding = mock_encode(["test query"])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            include=['documents', 'metadatas', 'distances']
        )

        assert len(results['documents'][0]) == 2, "Query should return 2 results"
        print(f"✅ Vector database operations work (returned {len(results['documents'][0])} results)")

        # Cleanup test database
        shutil.rmtree(test_db_path)

    except Exception as e:
        print(f"❌ Vector database test failed: {e}")
        return False

    # Test 4: Test retriever
    print("\n4️⃣  Testing retriever...")
    try:
        from src.chat.retriever import ConversationRetriever

        # Create test collection
        test_db_path = Path("data/test_rag_db")
        client = chromadb.PersistentClient(path=str(test_db_path))
        collection = client.create_collection(name="kaya_conversations")
        
        # Add test data
        test_chunks = chunks[:3] if len(chunks) >= 3 else chunks
        ids = [chunk['id'] for chunk in test_chunks]
        documents = [chunk['text'] for chunk in test_chunks]
        metadatas = [chunk['metadata'] for chunk in test_chunks]
        embeddings = [mock_encode([doc])[0] for doc in documents]
        
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

        # Test retriever
        retriever = ConversationRetriever(config)
        retriever.client = client
        retriever.collection = collection
        
        # Mock encoder
        class MockEncoder:
            def encode(self, queries):
                return mock_encode(queries)
        retriever.encoder = MockEncoder()

        # Test retrieval
        test_queries = [
            "What does Peter think about music?",
            "Tell me about Gil"
        ]

        for query in test_queries:
            results = retriever.retrieve(query, top_k=2)
            print(f"   Query: '{query}' → {len(results)} results")

        # Test context formatting
        context = retriever.format_context(results[:1])
        print(f"   Context formatted: {len(context)} characters")

        # Cleanup
        import shutil
        shutil.rmtree(test_db_path)

        print("✅ Retriever test passed")

    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        return False

    # Test 5: Test person extraction
    print("\n5️⃣  Testing person extraction...")
    try:
        retriever = ConversationRetriever(config)
        test_queries = [
            "What did Peter say about the music?",
            "Tell me about Gil and Rafa",
            "What does David think?",
            "General question about the group"
        ]

        for query in test_queries:
            persons = retriever.extract_query_persons(query)
            print(f"   '{query}' → persons: {persons}")

        print("✅ Person extraction test passed")

    except Exception as e:
        print(f"❌ Person extraction test failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ ALL RAG TESTS PASSED!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python src/data/build_vector_db.py (to build full database)")
    print("  2. Run: python src/chat/chat.py (to test chat with RAG)")
    print("  3. Test in Docker: docker-compose up --build")

    return True

if __name__ == "__main__":
    success = test_rag_components()
    sys.exit(0 if success else 1)