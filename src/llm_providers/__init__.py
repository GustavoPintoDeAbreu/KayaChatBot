"""
LLM provider factory and imports.
"""

from typing import Dict, Any
from .base import LLMProvider
from .azure_provider import AzureProvider
from .xai_provider import XAIProvider
from .data_cleaning import DataCleaningLLM


def get_provider(config: Dict[str, Any]) -> LLMProvider:
    """Factory function to get the appropriate LLM provider."""
    provider_name = config['generation']['provider'].lower()

    if provider_name == 'azure':
        return AzureProvider(config)
    elif provider_name == 'xai':
        return XAIProvider(config)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: 'azure', 'xai'")


def get_data_cleaning_llm(config: Dict[str, Any]) -> DataCleaningLLM:
    """Factory function to get the data cleaning LLM."""
    provider = get_provider(config)
    return DataCleaningLLM(config, provider)


__all__ = ['LLMProvider', 'AzureProvider', 'XAIProvider', 'DataCleaningLLM', 'get_provider', 'get_data_cleaning_llm']