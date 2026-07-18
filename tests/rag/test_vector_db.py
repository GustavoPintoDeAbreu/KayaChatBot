"""
Test RAG retrieval directly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load config
from src.config_loader import load_config
CONFIG_PATH = Path("/app/config.yaml")
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
config = load_config(str(CONFIG_PATH))

def test_retrieval():
    print("Testing RAG retrieval...")

    try:
        from src.chat.retriever import get_retriever

        print("Initializing retriever...")
        retriever = get_retriever(config)

        print("Testing retrieval...")
        results = retriever.retrieve("What did Peter say about music?", top_k=3)

        print(f"Retrieved {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Similarity: {result['similarity_score']:.3f}")
            print(f"   Messages: {result['message_count']}")
            print(f"   Text preview: {result['text'][:100]}...")

        print("✅ RAG retrieval works!")

    except Exception as e:
        print(f"❌ RAG retrieval failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieval()