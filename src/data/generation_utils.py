"""
Shared utilities for synthetic conversation generation.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import yaml


def load_config() -> Dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_base_dir() -> Path:
    """Get the base directory, detecting Docker environment."""
    if os.path.exists('/app'):
        return Path("/app")
    else:
        return Path("C:/Users/guga/Desktop/KayaChatBot")


def get_output_paths():
    """Get standard input/output file paths."""
    base_dir = get_base_dir()
    return {
        'finetune_chunks': base_dir / "data/finetune_chunks.jsonl",
        'output': base_dir / "data/synthetic_kaya.jsonl"
    }


def get_llm_provider(config: Dict):
    """Get the configured LLM provider."""
    from src.llm_providers import get_provider
    return get_provider(config)


def load_finetune_chunks(limit: int = None) -> List[Dict]:
    """Load finetune chunks from file."""
    paths = get_output_paths()
    finetune_chunks = []

    with open(paths['finetune_chunks'], 'r', encoding='utf-8') as f:
        for line in f:
            finetune_chunks.append(json.loads(line))

            if limit and len(finetune_chunks) >= limit:
                break

    return finetune_chunks


def save_conversation(conversation: Dict, file_handle):
    """Save a single conversation to JSONL."""
    file_handle.write(json.dumps(conversation, ensure_ascii=False) + '\n')
    file_handle.flush()