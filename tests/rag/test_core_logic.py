"""
Minimal RAG test - just validates core logic without external downloads.
"""

import os
import sys
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load configuration
CONFIG_PATH = Path("/app/config.yaml")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def test_rag_logic():
    """Test core RAG logic without external dependencies."""
    print("=" * 60)
    print("🧪 MINIMAL RAG LOGIC TEST")
    print("=" * 60)

    # Test 1: Check if cleaned messages exist
    print("\n1️⃣  Testing data availability...")
    cleaned_file = Path("data/all_messages_cleaned.jsonl")
    if not cleaned_file.exists():
        print("❌ Cleaned messages file not found. Run extract_all_messages.py first!")
        return False

    print(f"✅ Found cleaned messages: {cleaned_file}")

    # Test 2: Test chunking logic
    print("\n2️⃣  Testing conversation chunking...")
    try:
        from src.data.build_vector_db import ConversationChunker, load_cleaned_messages

        # Load a small sample of messages
        messages = load_cleaned_messages(limit=50)  # Test with 50 messages

        chunker = ConversationChunker()
        chunks = chunker.create_conversation_chunks(messages)

        print(f"✅ Created {len(chunks)} chunks from {len(messages)} messages")

        # Validate chunk structure
        if chunks:
            sample = chunks[0]
            required_keys = ['id', 'text', 'messages', 'token_count', 'message_count', 'participants', 'mentioned', 'metadata']
            for key in required_keys:
                assert key in sample, f"Chunk missing key: {key}"

            print(f"   Sample chunk: {sample['message_count']} messages, {sample['token_count']} tokens")
            print(f"   Participants: {', '.join(sample['participants'][:3])}...")
            print(f"   Mentioned: {', '.join(sample['mentioned'][:3])}...")

    except Exception as e:
        print(f"❌ Chunking test failed: {e}")
        return False

    # Test 3: Test person extraction logic
    print("\n3️⃣  Testing person extraction...")
    try:
        from src.chat.retriever import ConversationRetriever

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

        print("✅ Person extraction works")

    except Exception as e:
        print(f"❌ Person extraction test failed: {e}")
        return False

    # Test 4: Test context formatting
    print("\n4️⃣  Testing context formatting...")
    try:
        retriever = ConversationRetriever(config)

        # Mock retrieval results
        mock_results = [
            {
                'text': 'Peter: I love this music\nGil: Me too!',
                'metadata': {'timestamp_start': '2020-01-01T10:00:00'},
                'similarity_score': 0.95
            }
        ]

        context = retriever.format_context(mock_results)
        print(f"   Context formatted: {len(context)} characters")
        assert "[Relevant past conversations:]" in context
        print("✅ Context formatting works")

    except Exception as e:
        print(f"❌ Context formatting test failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ CORE RAG LOGIC TESTS PASSED!")
    print("=" * 60)
    print("\n🎯 RAG Implementation Status:")
    print("  ✅ Conversation chunking with metadata")
    print("  ✅ Person extraction from queries")
    print("  ✅ Context formatting for model input")
    print("  ⚠️  Vector database operations (requires model download)")
    print("  ⚠️  Full retrieval pipeline (requires model download)")
    print("\nNext steps:")
    print("  1. Build vector database: python src/data/build_vector_db.py")
    print("  2. Test full chat: python src/chat/chat.py")
    print("  3. The RAG logic is implemented and ready!")

    return True

if __name__ == "__main__":
    success = test_rag_logic()
    sys.exit(0 if success else 1)