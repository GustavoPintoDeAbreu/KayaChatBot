"""
Azure OpenAI provider implementation.
"""

import json
import os
from typing import List, Dict, Any
from openai import AzureOpenAI
from dotenv import load_dotenv

from .base import LLMProvider


class AzureProvider(LLMProvider):
    """Azure OpenAI provider for conversation generation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.azure_config = config['generation']['azure']
        self.client = self._initialize_client()

    def _initialize_client(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client."""
        load_dotenv()

        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Azure OpenAI API key not found! Set AZURE_OPENAI_API_KEY in .env")

        endpoint = self.azure_config.get('endpoint', '')
        if not (endpoint.startswith('https://') and '.openai.azure.com' in endpoint):
            raise ValueError(
                f"Invalid Azure OpenAI endpoint: '{endpoint}'. "
                "Must be https://<resource-name>.openai.azure.com/"
            )

        return AzureOpenAI(
            api_key=api_key,
            api_version=self.azure_config['api_version'],
            azure_endpoint=self.azure_config['endpoint']
        )

    def generate_conversations(self, prompt: str) -> List[Dict]:
        """Generate conversations using Azure OpenAI."""
        def _generate():
            response = self.client.chat.completions.create(
                model=self.azure_config['model'],
                messages=[
                    {"role": "system", "content": "You generate training data in JSON format. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.azure_config['temperature'],
                max_tokens=self.azure_config['max_tokens'],
                timeout=self.azure_config['timeout']
            )

            return self._parse_response(response.choices[0].message.content.strip())

        return self._retry_with_backoff(_generate)

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a raw text response using Azure OpenAI.

        Uses ``generation.azure.extraction_temperature`` (default 0.3) for
        factual/structured extraction tasks rather than the higher creative
        temperature used by ``generate_conversations``.
        """
        def _generate():
            response = self.client.chat.completions.create(
                model=self.azure_config['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.azure_config.get('extraction_temperature', 0.3),
                max_tokens=self.azure_config['max_tokens'],
                timeout=self.azure_config['timeout']
            )
            return response.choices[0].message.content.strip()

        return self._retry_with_backoff(_generate)

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat completion request and return the response text."""
        def _complete():
            response = self.client.chat.completions.create(
                model=self.azure_config['model'],
                messages=messages,
                temperature=self.azure_config['temperature'],
                max_tokens=self.azure_config['max_tokens'],
                timeout=self.azure_config['timeout']
            )
            return response.choices[0].message.content.strip()

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