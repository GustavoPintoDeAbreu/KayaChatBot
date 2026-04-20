"""
xAI (Grok) provider implementation.

Self-Censorship Investigation Notes
-------------------------------------
The xAI GetCompletionsRequest proto has no safety filter field. Grok is
designed to be less restricted than most models by default and has no
API-level safety toggle. Available proto fields (for reference):
    messages, model, user, n, max_tokens, seed, stop, temperature, top_p,
    logprobs, top_logprobs, tools, tool_choice, response_format,
    frequency_penalty, presence_penalty, reasoning_effort,
    search_parameters, parallel_tool_calls, previous_response_id,
    store_messages, use_encrypted_content, max_turns, include, agent_count.

  - ``use_encrypted_content`` is for private/encrypted data routing, not
    content policy.
  - To minimize self-censoring: prepend the uncensored system prompt preamble
    from config.yaml (chat.uncensored_system_prompt). The generate_text()
    and generate_conversations() methods accept a system_prompt kwarg or can
    have the preamble injected into the caller's system message.
"""

import json
import os
from typing import List, Dict, Any
from xai_sdk import Client
from xai_sdk.chat import system, user
from dotenv import load_dotenv

from .base import LLMProvider


class XAIProvider(LLMProvider):
    """xAI (Grok) provider for conversation generation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.xai_config = config['generation']['xai']
        self.client = self._initialize_client()
        self._has_xai_assistant = self._check_xai_assistant()

    @staticmethod
    def _check_xai_assistant() -> bool:
        """Check once at init whether xai_sdk exports an assistant helper."""
        try:
            from xai_sdk.chat import assistant  # noqa: F401
            return True
        except ImportError:
            return False

    def _initialize_client(self) -> Client:
        """Initialize xAI client."""
        load_dotenv()

        api_key = os.getenv('XAI_API_KEY')
        if not api_key:
            raise ValueError("xAI API key not found! Set XAI_API_KEY in .env")

        return Client(api_key=api_key)

    def generate_conversations(self, prompt: str) -> List[Dict]:
        """Generate conversations using xAI (Grok)."""
        def _generate():
            # xAI uses system() and user() functions from xai_sdk.chat
            chat = self.client.chat.create(
                model=self.xai_config['model'],
                messages=[
                    system("You generate training data in JSON format. Output only valid JSON."),
                    user(prompt)
                ]
            )
            
            response = chat.sample()
            # xAI response might be different - check the content attribute
            content = response.content if hasattr(response, 'content') else str(response)
            return self._parse_response(content.strip())

        return self._retry_with_backoff(_generate)

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a raw text response using xAI (Grok)."""
        def _generate():
            chat = self.client.chat.create(
                model=self.xai_config['model'],
                messages=[
                    system(system_prompt),
                    user(user_prompt)
                ]
            )
            response = chat.sample()
            content = response.content if hasattr(response, 'content') else str(response)
            return content.strip()

        return self._retry_with_backoff(_generate)

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat completion request and return the response text."""
        def _complete():
            if self._has_xai_assistant:
                from xai_sdk.chat import assistant as xai_assistant
            else:
                xai_assistant = None

            formatted = []
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    formatted.append(system(content))
                elif role == 'user':
                    formatted.append(user(content))
                elif role == 'assistant' and xai_assistant is not None:
                    formatted.append(xai_assistant(content))
                # Skip assistant messages if xai_sdk doesn't support them

            chat = self.client.chat.create(
                model=self.xai_config['model'],
                messages=formatted,
            )
            response = chat.sample()
            content = response.content if hasattr(response, 'content') else str(response)
            return content.strip()

        return self._retry_with_backoff(_complete)

    def _parse_response(self, content: str) -> List[Dict]:
        """Parse the response content into conversations."""
        # Remove markdown code blocks if present
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()
            if content.endswith('```'):
                content = content[:-3].strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try basic JSON repairs
            try:
                repaired = content.replace(',]', ']').replace(',}', '}')
                import re
                repaired = re.sub(r'"(role|content|turns|conversations)"\s+"', r'"\1": "', repaired)
                result = json.loads(repaired)
            except:
                raise ValueError(f"Failed to parse JSON response: {content[:300]}...")

        # Extract conversations
        if isinstance(result, dict) and 'conversations' in result:
            conversations = result['conversations']
        elif isinstance(result, dict) and 'conversation' in result:
            conversations = [result['conversation']]  # xAI returns single conversation
        elif isinstance(result, list):
            conversations = result
        else:
            raise ValueError(f"Unexpected response format: {list(result.keys()) if isinstance(result, dict) else type(result)}")

        # Format conversations
        formatted_conversations = []
        for conv in conversations:
            # Extract turns
            if isinstance(conv, dict):
                if 'turns' in conv:
                    turns = conv['turns']
                elif 'conversation' in conv:
                    turns = conv['conversation']
                elif 'messages' in conv:
                    turns = conv['messages']
            elif isinstance(conv, list):
                turns = conv

            if not turns or not isinstance(turns, list) or len(turns) < 2:
                continue

            # Validate turns have role/content
            if not all(isinstance(t, dict) and 'role' in t and 'content' in t for t in turns):
                continue

            formatted_conversations.append(turns)

        return formatted_conversations