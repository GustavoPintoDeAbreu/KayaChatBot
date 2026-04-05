"""
Azure OpenAI provider implementation.
Supports both standard Azure OpenAI (chat.openai.azure.com) and
Azure Cognitive Services endpoints (cognitiveservices.azure.com).

Pass ``config_key='azure_gpt53'`` to use the GPT-5.3-chat deployment.
GPT-5.x models use the Azure Responses API (``client.responses.create``);
set ``use_responses_api: true`` in the config section to enable this.
"""

import json
import os
from typing import List, Dict, Any
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

from .base import LLMProvider


class AzureProvider(LLMProvider):
    """Azure OpenAI provider for conversation generation.

    Args:
        config:      Full config dict (``config.yaml``).
        config_key:  Which sub-key under ``generation`` to read.
                     Defaults to ``'azure'`` (gpt-4.1-mini).
                     Pass ``'azure_gpt53'`` for the GPT-5.3-chat deployment.
    """

    def __init__(self, config: Dict[str, Any], config_key: str = 'azure'):
        super().__init__(config)
        if config_key not in config.get('generation', {}):
            raise ValueError(
                f"Config section 'generation.{config_key}' not found. "
                "Check config.yaml."
            )
        self.azure_config = config['generation'][config_key]
        self.use_responses_api = self.azure_config.get('use_responses_api', False)
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize client, supporting model-specific API keys and Responses API."""
        load_dotenv()

        # Model-specific key takes precedence; fall back to generic key.
        api_key_env = self.azure_config.get('api_key_env', 'AZURE_OPENAI_API_KEY')
        api_key = os.getenv(api_key_env) or os.getenv('AZURE_OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                f"Azure OpenAI API key not found! "
                f"Set {api_key_env} (or AZURE_OPENAI_API_KEY) in .env"
            )

        endpoint = self.azure_config.get('endpoint', '').rstrip('/')
        if not endpoint.startswith('https://'):
            raise ValueError(
                f"Invalid Azure endpoint: '{endpoint}'. "
                "Must start with https://"
            )

        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        base_host = f"{parsed.scheme}://{parsed.netloc}"

        if self.use_responses_api:
            # GPT-5.x models: use plain OpenAI client pointed at /openai/v1/
            # The Responses API lives at {host}/openai/v1/responses
            return OpenAI(
                api_key=api_key,
                base_url=f"{base_host}/openai/v1/",
            )
        else:
            # Standard Chat Completions via AzureOpenAI SDK
            return AzureOpenAI(
                api_key=api_key,
                api_version=self.azure_config['api_version'],
                azure_endpoint=f"{base_host}/",
            )

    def generate_conversations(self, prompt: str) -> List[Dict]:
        """Generate conversations using Azure OpenAI."""
        def _generate():
            if self.use_responses_api:
                response = self.client.responses.create(
                    model=self.azure_config['model'],
                    input=[
                        {"role": "system", "content": "You generate training data in JSON format. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.azure_config['temperature'],
                    max_output_tokens=self.azure_config['max_tokens'],
                )
                return self._parse_response(response.output_text)
            else:
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
        """Generate a raw text response using Azure OpenAI."""
        def _generate():
            if self.use_responses_api:
                response = self.client.responses.create(
                    model=self.azure_config['model'],
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.azure_config.get('extraction_temperature', 0.3),
                    max_output_tokens=self.azure_config['max_tokens'],
                )
                return response.output_text
            else:
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
            if self.use_responses_api:
                response = self.client.responses.create(
                    model=self.azure_config['model'],
                    input=messages,
                    temperature=self.azure_config['temperature'],
                    max_output_tokens=self.azure_config['max_tokens'],
                )
                return response.output_text
            else:
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