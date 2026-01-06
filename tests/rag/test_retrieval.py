"""
Quick RAG test - verify retrieval and chat integration work
"""

import os
import sys
from pathlib import Path

# Add src to sys.path for Docker compatibility
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml

# Load config
config_path = Path("config.yaml")
if not config_path.exists():
    config_path = Path("/app/config.yaml")

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("=" * 60)
print("🧪 QUICK RAG TEST")
print("=" * 60)

# Test 1: Check database exists
print("\n1️⃣  Checking RAG database...")
rag_db_path = Path("data/rag_db") if Path("data").exists() else Path("/app/data/rag_db")
if not rag_db_path.exists():
    print(f"❌ RAG database not found at {rag_db_path}")
    print("   Run: python src/data/build_vector_db.py")
    sys.exit(1)

print(f"✅ RAG database found at {rag_db_path}")

# Test 2: Import and initialize retriever
print("\n2️⃣  Testing retriever import...")
try:
    # Try both import methods
    try:
        from src.chat.retriever import get_retriever
        print("   ✅ Imported via 'from src.chat.retriever'")
    except ImportError:
        from chat.retriever import get_retriever
        print("   ✅ Imported via 'from chat.retriever'")
    
    print("\n3️⃣  Initializing retriever...")
    retriever = get_retriever(config)
    print("✅ Retriever initialized!")
    
    # Test 3: Retrieve some results
    print("\n4️⃣  Testing retrieval...")
    test_queries = [
        "What did Peter say about music?",
        "Tell me about Gil",
        "What does David think?"
    ]
    
    for query in test_queries:
        results = retriever.retrieve(query, top_k=3)
        print(f"\n   Query: '{query}'")
        print(f"   Retrieved {len(results)} chunks:")
        
        if results:
            for i, result in enumerate(results[:2], 1):
                print(f"      {i}. Similarity: {result['similarity_score']:.3f}, {result['message_count']} messages")
                # Show first 50 chars of text
                text_preview = result['text'].replace('\n', ' ')[:80] + "..."
                print(f"         Preview: {text_preview}")
    
    # Test 4: Context formatting
    print("\n5️⃣  Testing context formatting...")
    context = retriever.format_context(results[:2])
    print(f"   Context length: {len(context)} characters")
    print(f"   Preview: {context[:150]}...")
    
    print("\n" + "=" * 60)
    print("✅ ALL RAG TESTS PASSED!")
    print("=" * 60)
    print("\n🎯 RAG is fully functional and ready to use!")
    print("\nTo test in chat:")
    print("  docker-compose run --rm kaya-chatbot python src/chat/chat.py")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
