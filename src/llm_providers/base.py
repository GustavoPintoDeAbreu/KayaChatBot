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
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a raw text response given system and user prompts.

        Unlike ``generate_conversations``, this method returns the raw text
        content of the model response without any post-processing, making it
        suitable for structured JSON extraction tasks.
        """
        pass

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat completion request and return the response text."""

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff on rate-limit errors."""
        import logging
        import random as _random
        logger = logging.getLogger(__name__)
        base_delay = self.delay_seconds
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if 'rate' in error_msg or '429' in error_msg or 'quota' in error_msg:
                    if attempt < self.max_attempts - 1:
                        # Exponential backoff with jitter: base * 2^attempt + jitter
                        delay = base_delay * (2 ** attempt) + _random.uniform(0, base_delay * 0.1)
                        logger.warning(
                            "Rate limit hit, retrying in %.1fs... (attempt %d/%d)",
                            delay, attempt + 1, self.max_attempts
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise e
                else:
                    # Non-rate-limit errors: fail immediately
                    raise e