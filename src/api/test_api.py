"""
Simple test script to verify the API structure without running the server.
Tests imports, models, and basic logic.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import yaml
        import requests
        from fastapi import FastAPI, HTTPException, Depends
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, EmailStr
        from passlib.context import CryptContext
        print("✓ All FastAPI dependencies imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True

def test_config():
    """Test config file loading."""
    print("\nTesting config loading...")
    
    try:
        import yaml
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['inference', 'rag', 'data']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False
        
        print(f"✓ Config loaded successfully")
        print(f"  - RAG enabled: {config.get('rag', {}).get('enabled', False)}")
        print(f"  - System prompt: {config.get('data', {}).get('system_prompt', '')[:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_pydantic_models():
    """Test Pydantic model definitions."""
    print("\nTesting Pydantic models...")
    
    try:
        from pydantic import BaseModel, EmailStr
        
        class LoginRequest(BaseModel):
            email: EmailStr
        
        class ChatRequest(BaseModel):
            message: str
            conversation_id: str = None
        
        # Test model validation
        login = LoginRequest(email="test@example.com")
        chat = ChatRequest(message="Hello")
        
        print("✓ Pydantic models working correctly")
        return True
    except Exception as e:
        print(f"✗ Pydantic error: {e}")
        return False

def test_rag_retriever_import():
    """Test RAG retriever import."""
    print("\nTesting RAG retriever import...")
    
    try:
        from src.chat.retriever import ConversationRetriever
        print("✓ RAG retriever imported successfully")
        
        # Check if RAG database exists
        rag_db_path = Path(__file__).parent.parent.parent / "data" / "rag_db"
        if rag_db_path.exists():
            print(f"✓ RAG database found at: {rag_db_path}")
        else:
            print(f"⚠ RAG database not found at: {rag_db_path}")
            print("  (This is expected if you haven't built it yet)")
        
        return True
    except ImportError as e:
        print(f"✗ RAG import error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("API Structure Verification")
    print("=" * 60)
    
    tests = [
        ("Import test", test_imports),
        ("Config test", test_config),
        ("Pydantic models", test_pydantic_models),
        ("RAG retriever", test_rag_retriever_import),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! API structure is valid.")
    else:
        print("\n⚠ Some tests failed. Review the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
