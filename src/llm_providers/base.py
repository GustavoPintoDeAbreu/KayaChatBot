"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_attempts = config.get('retry', {}).get('max_attempts', 5)
        self.delay_seconds = config.get('retry', {}).get('delay_seconds', 30)

    @abstractmethod
    def generate_conversations(self, prompt: str) -> List[Dict]:
        """Generate conversations from a prompt."""
        pass

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat completion request and return the response text."""
        pass

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if 'rate' in error_msg or '429' in error_msg or 'quota' in error_msg:
                    if attempt < self.max_attempts - 1:
                        print(f"Rate limit hit, retrying in {self.delay_seconds} seconds... (attempt {attempt + 1}/{self.max_attempts})")
                        time.sleep(self.delay_seconds)
                        continue
                    else:
                        raise e
                else:
                    # Non-rate-limit errors, don't retry
                    raise e