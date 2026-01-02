"""
LLM provider factory and imports.
"""

from typing import Dict, Any
from .base import LLMProvider
from .azure_provider import AzureProvider
from .xai_provider import XAIProvider


def get_provider(config: Dict[str, Any]) -> LLMProvider:
    """Factory function to get the appropriate LLM provider."""
    provider_name = config['generation']['provider'].lower()

    if provider_name == 'azure':
        return AzureProvider(config)
    elif provider_name == 'xai':
        return XAIProvider(config)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: 'azure', 'xai'")


__all__ = ['LLMProvider', 'AzureProvider', 'XAIProvider', 'get_provider']