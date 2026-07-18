#!/usr/bin/env python3
"""
Test script for LLM-based data cleaning.
Tests the WhatsAppReader and InstagramReader with LLM cleaning enabled.
"""

import sys
import os
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config_loader import load_config as _load_config
from src.data.readers import WhatsAppReader, InstagramReader


def load_config():
    """Load configuration via the single entry point (src.config_loader)."""
    config_paths = ["/app/config.yaml", str(Path(__file__).parent.parent.parent / "config.yaml")]
    for path in config_paths:
        if os.path.exists(path):
            return _load_config(path)
    raise FileNotFoundError("Could not find config file")


def run_whatsapp_cleaning():
    """Test WhatsApp reader with LLM cleaning."""
    print("\n" + "="*60)
    print("🧪 TESTING WHATSAPP LLM CLEANING")
    print("="*60)

    config = load_config()

    # Test messages that should be cleaned
    test_messages = [
        "ahahah",  # Should be discarded (noise)
        "O Peter é fixe ahahah",  # Should be cleaned (remove filler)
        "sim",  # Should be kept (substantive short response)
        "ok",  # Should be kept (substantive short response)
        "Baza fixe lmao vamos marcar?",  # Should be cleaned (remove lmao)
        "não",  # Should be kept (substantive short response)
        "wtf aconteceu?",  # Should be kept (substantive)
        "lol",  # Should be kept (substantive short response)
    ]

    print(f"Testing {len(test_messages)} messages...")

    # Create a mock WhatsApp file content
    mock_content = ""
    for i, msg in enumerate(test_messages):
        mock_content += f"1/1/24, 10:0{i} - TestUser: {msg}\n"

    # Create temporary file
    temp_file = Path("/tmp/test_whatsapp.txt") if os.path.exists('/app') else Path("test_whatsapp.txt")
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(mock_content)

    try:
        # Test with LLM cleaning
        reader = WhatsAppReader(str(temp_file), config)
        messages = reader.read()

        print(f"✅ LLM cleaning processed {len(messages)} messages")

        for msg in messages:
            print(f"  ✓ '{msg.content}' (from {msg.sender})")

        # Verify we got some messages
        if len(messages) == 0:
            print("❌ ERROR: No messages were kept!")
            return False

        print("✅ WhatsApp LLM cleaning test passed!")
        return True

    except Exception as e:
        print(f"❌ ERROR in WhatsApp cleaning: {e}")
        return False
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()


def run_instagram_cleaning():
    """Test Instagram reader with LLM cleaning."""
    print("\n" + "="*60)
    print("🧪 TESTING INSTAGRAM LLM CLEANING")
    print("="*60)

    config = load_config()

    # Test messages
    test_messages = [
        "ahahah",  # Should be discarded
        "O Peter é fixe ahahah",  # Should be cleaned
        "sim",  # Should be kept
        "Baza fixe lmao",  # Should be cleaned
    ]

    # Create mock Instagram JSON
    mock_data = {
        "participants": [{"name": "TestUser"}],
        "messages": []
    }

    for i, msg in enumerate(test_messages):
        mock_data["messages"].append({
            "sender_name": "TestUser",
            "timestamp_ms": 1700000000000 + (i * 60000),  # 1 minute apart
            "content": msg
        })

    # Create temporary file
    temp_file = Path("/tmp/test_instagram.json") if os.path.exists('/app') else Path("test_instagram.json")
    with open(temp_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(mock_data, f)

    try:
        # Test with LLM cleaning
        reader = InstagramReader(str(temp_file), config)
        messages = reader.read()

        print(f"✅ LLM cleaning processed {len(messages)} messages")

        for msg in messages:
            print(f"  ✓ '{msg.content}' (from {msg.sender})")

        # Verify we got some messages
        if len(messages) == 0:
            print("❌ ERROR: No messages were kept!")
            return False

        print("✅ Instagram LLM cleaning test passed!")
        return True

    except Exception as e:
        print(f"❌ ERROR in Instagram cleaning: {e}")
        return False
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()


def main():
    """Run all tests."""
    print("🚀 TESTING LLM-BASED DATA CLEANING")
    print("This test will use the configured LLM provider (xAI by default)")

    # Check if .env exists
    env_path = Path("/app/.env") if os.path.exists('/app') else Path(".env")
    if not env_path.exists():
        print("❌ ERROR: .env file not found! Please create it with your API keys.")
        print("   For xAI: XAI_API_KEY=your_key_here")
        print("   For Azure: AZURE_OPENAI_API_KEY=your_key_here")
        return False

    config = load_config()

    # Check if LLM cleaning is enabled
    if not config.get('data', {}).get('cleaning', {}).get('enabled', False):
        print("❌ ERROR: LLM cleaning is not enabled in config!")
        print("   Set data.cleaning.enabled: true in config.yaml")
        return False

    print(f"✅ Using LLM provider: {config['generation']['provider']}")

    # Run tests
    whatsapp_ok = run_whatsapp_cleaning()
    instagram_ok = run_instagram_cleaning()

    if whatsapp_ok and instagram_ok:
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("LLM-based data cleaning is working correctly.")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("❌ SOME TESTS FAILED!")
        print("Check the errors above and fix the implementation.")
        print("="*60)
        return False


def test_whatsapp_cleaning():
    assert run_whatsapp_cleaning()


def test_instagram_cleaning():
    assert run_instagram_cleaning()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)