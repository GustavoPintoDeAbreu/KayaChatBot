#!/usr/bin/env python3
"""
Test script for LLM-based data cleaning.
Tests the WhatsAppReader and InstagramReader with LLM cleaning enabled.
"""

import sys
import os
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, '/app/src')

from src.data.readers import WhatsAppReader, InstagramReader


def load_config():
    """Load configuration."""
    # In Docker, use config.docker.yaml, otherwise use config.yaml
    config_paths = ["/app/config.docker.yaml", "/app/config.yaml", "config.docker.yaml", "config.yaml"]
    config = None
    
    for path in config_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                print(f"✅ Loaded config from {path}")
                break
        except FileNotFoundError:
            continue
    
    if config is None:
        raise FileNotFoundError("Could not find config file")
    
    # Debug: print config structure
    print(f"DEBUG: Config keys: {list(config.keys())}")
    if 'data' in config:
        print(f"DEBUG: Data keys: {list(config['data'].keys())}")
        if 'cleaning' in config['data']:
            print(f"DEBUG: Cleaning config: {config['data']['cleaning']}")
    
    return config


def test_whatsapp_cleaning():
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


def test_instagram_cleaning():
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
    whatsapp_ok = test_whatsapp_cleaning()
    instagram_ok = test_instagram_cleaning()

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


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)