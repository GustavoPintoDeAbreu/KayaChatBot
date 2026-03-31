"""
xAI (Grok) provider implementation.
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