"""
LLM-based data cleaning for chat messages.
Uses LLM to intelligently filter out noise and filler content.
"""

import json
import hashlib
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from .base import LLMProvider


@dataclass
class CleaningDecision:
    """Represents a cleaning decision made by the LLM."""
    message: str
    is_substantive: bool  # True if message should be kept
    is_filler: bool  # True if message contains filler words that should be cleaned
    cleaned_content: Optional[str]  # Cleaned version (None if should be discarded)
    confidence: float  # 0.0 to 1.0
    reasoning: str


class DataCleaningLLM:
    """
    Uses LLM to intelligently clean chat data by:
    1. Classifying short messages as substantive vs. noise
    2. Identifying and cleaning filler words/phrases in context
    """

    def __init__(self, config: Dict[str, Any], provider: LLMProvider):
        self.config = config
        self.provider = provider
        self.batch_size = config.get('data_cleaning', {}).get('batch_size', 10)
        self.confidence_threshold = config.get('data_cleaning', {}).get('confidence_threshold', 0.7)

        # Caching
        self.cache_file = Path(config.get('data_cleaning', {}).get('cache_file', 'data/.cleaning_cache.json'))
        self.cache_file.parent.mkdir(exist_ok=True, parents=True)
        self.cache = self._load_cache()

        # Statistics
        self.stats_file = Path(config.get('data_cleaning', {}).get('stats_file', 'data/cleaning_stats.json'))
        self.stats = self._load_stats()

    def _load_cache(self) -> Dict[str, CleaningDecision]:
        """Load cleaning decisions cache."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert back to CleaningDecision objects
                    return {
                        k: CleaningDecision(**v) for k, v in data.items()
                    }
            except Exception as e:
                print(f"Warning: Could not load cleaning cache: {e}")
        return {}

    def _save_cache(self):
        """Save cleaning decisions cache."""
        try:
            # Convert CleaningDecision objects to dicts
            data = {
                k: {
                    'message': v.message,
                    'is_substantive': v.is_substantive,
                    'is_filler': v.is_filler,
                    'cleaned_content': v.cleaned_content,
                    'confidence': v.confidence,
                    'reasoning': v.reasoning
                }
                for k, v in self.cache.items()
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cleaning cache: {e}")

    def _load_stats(self) -> Dict[str, Any]:
        """Load cleaning statistics."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cleaning stats: {e}")
        return {
            'total_messages_processed': 0,
            'messages_kept': 0,
            'messages_discarded': 0,
            'messages_cleaned': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'errors': 0
        }

    def _save_stats(self):
        """Save cleaning statistics."""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cleaning stats: {e}")

    def _get_message_hash(self, message: str) -> str:
        """Generate hash for message caching."""
        return hashlib.md5(message.strip().lower().encode('utf-8')).hexdigest()

    def _get_short_message_prompt(self, messages: List[str]) -> str:
        """Generate prompt for classifying short messages."""
        examples = """
EXEMPLOS DE MENSAGENS SUBSTANTIVAS (manter):
- "sim" (resposta direta a pergunta)
- "não, obrigado" (resposta educada)
- "estou bem, e tu?" (conversa natural)
- "fds" (abreviação comum em português)
- "baza fixe" (expressão positiva)
- "lol" (riso, mas em contexto de resposta)

EXEMPLOS DE RUÍDO (descartar):
- "ahahah" (apenas riso isolado)
- "wtf" (surpresa isolada sem contexto)
- "lmao" (riso isolado)
- "ok" (isolado, sem contexto)
- "ya" (informal mas vazio)
- "sim" (isolado, sem pergunta anterior)
- "não" (isolado, sem contexto)
"""

        messages_list = "\n".join(f"{i+1}. \"{msg}\"" for i, msg in enumerate(messages))

        return f"""Avalia estas mensagens curtas de uma conversa do WhatsApp em português europeu.

{examples}

TAREFA: Para cada mensagem, decide se é SUBSTANTIVA (deve ser mantida) ou RUÍDO (deve ser descartada).

MENSAGENS A AVALIAR:
{messages_list}

INSTRUÇÕES:
- Considera o contexto do WhatsApp português: abreviações como "fds", "baza", expressões como "lol", "ahah"
- Mantém respostas diretas e conversacionais
- Descarta risos isolados, exclamações vazias, e mensagens sem conteúdo
- Seja rigoroso: mensagens muito curtas sem significado devem ser descartadas

RESPOSTA OBRIGATÓRIA em JSON:
{{
  "decisions": [
    {{
      "index": 1,
      "is_substantive": true/false,
      "confidence": 0.0-1.0,
      "reasoning": "explicação breve"
    }},
    ...
  ]
}}

OUTPUT ONLY VALID JSON."""

    def _get_filler_cleaning_prompt(self, messages: List[str]) -> str:
        """Generate prompt for cleaning filler words."""
        examples = """
EXEMPLOS DE LIMPEZA:

ORIGINAL: "O Peter é fixe ahahah, sempre a organizar cenas ahah"
LIMPO: "O Peter é fixe, sempre a organizar cenas"

ORIGINAL: "Não sei ahahah, pergunta ao Gil lol"
LIMPO: "Não sei, pergunta ao Gil"

ORIGINAL: "Baza fixe lmao, vamos marcar?"
LIMPO: "Baza fixe, vamos marcar?"

ORIGINAL: "Sim ahah, estou bem"
LIMPO: "Sim, estou bem"

ORIGINAL: "Wtf aconteceu? ahahah não acredito"
LIMPO: "Wtf aconteceu? não acredito"
"""

        messages_list = "\n".join(f"{i+1}. \"{msg}\"" for i, msg in enumerate(messages))

        return f"""Limpa palavras de enchimento (fillers) destas mensagens de WhatsApp em português europeu.

PALAVRAS DE ENCHIMENTO COMUNS:
- ahahah, ahah, ahaha, ahahha (risos)
- lol, lmao, rofl (risos em inglês)
- wtf (surpresa)
- omg, wow (exclamações vazias)
- ok, ya (respostas vazias isoladas)

{examples}

TAREFA: Remove palavras de enchimento excessivas, mantendo no máximo 1 ocorrência por mensagem.

MENSAGENS A LIMPAR:
{messages_list}

INSTRUÇÕES:
- Mantém 1 ocorrência de filler se for natural na conversa
- Remove fillers do início das mensagens
- Limpa fillers excessivos (mais que 1 por mensagem)
- Preserva o significado e tom da mensagem
- Se a mensagem ficar vazia após limpeza, indica que deve ser descartada

RESPOSTA OBRIGATÓRIA em JSON:
{{
  "cleaned_messages": [
    {{
      "index": 1,
      "original": "mensagem original",
      "cleaned": "mensagem limpa ou null se deve descartar",
      "changes_made": "descrição das mudanças"
    }},
    ...
  ]
}}

OUTPUT ONLY VALID JSON."""

    def _parse_short_message_response(self, content: str, messages: List[str]) -> List[CleaningDecision]:
        """Parse LLM response for short message classification."""
        try:
            # Remove markdown if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)

            if 'decisions' not in result:
                raise ValueError("Missing 'decisions' key in response")

            decisions = []
            for decision_data in result['decisions']:
                index = decision_data.get('index', 0) - 1  # Convert to 0-based
                if 0 <= index < len(messages):
                    message = messages[index]
                    decisions.append(CleaningDecision(
                        message=message,
                        is_substantive=decision_data.get('is_substantive', False),
                        is_filler=False,  # Not applicable for this task
                        cleaned_content=message if decision_data.get('is_substantive', False) else None,
                        confidence=min(1.0, max(0.0, decision_data.get('confidence', 0.5))),
                        reasoning=decision_data.get('reasoning', 'No reasoning provided')
                    ))

            return decisions

        except Exception as e:
            raise ValueError(f"Failed to parse short message cleaning response: {e}")

    def _parse_filler_cleaning_response(self, content: str, messages: List[str]) -> List[CleaningDecision]:
        """Parse LLM response for filler cleaning."""
        try:
            # Remove markdown if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)

            if 'cleaned_messages' not in result:
                raise ValueError("Missing 'cleaned_messages' key in response")

            decisions = []
            for cleaned_data in result['cleaned_messages']:
                index = cleaned_data.get('index', 0) - 1  # Convert to 0-based
                if 0 <= index < len(messages):
                    original = messages[index]
                    cleaned = cleaned_data.get('cleaned')

                    # Determine if substantive based on cleaning result
                    is_substantive = cleaned is not None and len(cleaned.strip()) > 0
                    is_filler = original != cleaned if cleaned else False

                    decisions.append(CleaningDecision(
                        message=original,
                        is_substantive=is_substantive,
                        is_filler=is_filler,
                        cleaned_content=cleaned.strip() if cleaned else None,
                        confidence=0.9,  # High confidence for filler cleaning
                        reasoning=cleaned_data.get('changes_made', 'Filler words cleaned')
                    ))

            return decisions

        except Exception as e:
            raise ValueError(f"Failed to parse filler cleaning response: {e}")

    def clean_short_messages(self, messages: List[str]) -> List[CleaningDecision]:
        """
        Use LLM to classify short messages as substantive or noise.
        Only processes messages that aren't in cache.
        """
        if not messages:
            return []

        # Check cache first
        uncached_messages = []
        uncached_indices = []

        for i, msg in enumerate(messages):
            msg_hash = self._get_message_hash(msg)
            if msg_hash in self.cache:
                self.stats['cache_hits'] += 1
            else:
                uncached_messages.append(msg)
                uncached_indices.append(i)

        # Process uncached messages in batches
        all_decisions = [None] * len(messages)

        for i in range(0, len(uncached_messages), self.batch_size):
            batch = uncached_messages[i:i + self.batch_size]
            batch_indices = uncached_indices[i:i + self.batch_size]

            try:
                prompt = self._get_short_message_prompt(batch)
                # Use the provider's generate_conversations method but adapt for our use case
                # Since we need a different interface, we'll call the provider directly
                decisions = self._call_llm_for_cleaning(prompt, batch, is_short_message_task=True)

                # Store in cache and results
                for j, decision in enumerate(decisions):
                    if j < len(batch_indices):
                        global_idx = batch_indices[j]
                        all_decisions[global_idx] = decision

                        # Cache the decision
                        msg_hash = self._get_message_hash(batch[j])
                        self.cache[msg_hash] = decision

                self.stats['api_calls'] += 1

            except Exception as e:
                self.stats['errors'] += 1
                error_msg = f"LLM cleaning failed for batch {i//self.batch_size + 1}: {e}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)

        # Fill in cached decisions
        for i, msg in enumerate(messages):
            if all_decisions[i] is None:
                msg_hash = self._get_message_hash(msg)
                if msg_hash in self.cache:
                    all_decisions[i] = self.cache[msg_hash]

        # Update statistics
        for decision in all_decisions:
            if decision:
                self.stats['total_messages_processed'] += 1
                if decision.is_substantive and decision.cleaned_content:
                    self.stats['messages_kept'] += 1
                else:
                    self.stats['messages_discarded'] += 1

        # Save cache and stats
        self._save_cache()
        self._save_stats()

        return all_decisions

    def clean_filler_words(self, messages: List[str]) -> List[CleaningDecision]:
        """
        Use LLM to clean filler words from messages.
        Only processes messages that aren't in cache.
        """
        if not messages:
            return []

        # Check cache first
        uncached_messages = []
        uncached_indices = []

        for i, msg in enumerate(messages):
            msg_hash = self._get_message_hash(msg)
            if msg_hash in self.cache:
                self.stats['cache_hits'] += 1
            else:
                uncached_messages.append(msg)
                uncached_indices.append(i)

        # Process uncached messages in batches
        all_decisions = [None] * len(messages)

        for i in range(0, len(uncached_messages), self.batch_size):
            batch = uncached_messages[i:i + self.batch_size]
            batch_indices = uncached_indices[i:i + self.batch_size]

            try:
                prompt = self._get_filler_cleaning_prompt(batch)
                decisions = self._call_llm_for_cleaning(prompt, batch, is_short_message_task=False)

                # Store in cache and results
                for j, decision in enumerate(decisions):
                    if j < len(batch_indices):
                        global_idx = batch_indices[j]
                        all_decisions[global_idx] = decision

                        # Cache the decision
                        msg_hash = self._get_message_hash(batch[j])
                        self.cache[msg_hash] = decision

                self.stats['api_calls'] += 1

            except Exception as e:
                self.stats['errors'] += 1
                error_msg = f"LLM filler cleaning failed for batch {i//self.batch_size + 1}: {e}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)

        # Fill in cached decisions
        for i, msg in enumerate(messages):
            if all_decisions[i] is None:
                msg_hash = self._get_message_hash(msg)
                if msg_hash in self.cache:
                    all_decisions[i] = self.cache[msg_hash]

        # Update statistics
        for decision in all_decisions:
            if decision:
                self.stats['total_messages_processed'] += 1
                if decision.is_substantive and decision.cleaned_content:
                    self.stats['messages_kept'] += 1
                elif decision.cleaned_content and decision.cleaned_content != decision.message:
                    self.stats['messages_cleaned'] += 1
                else:
                    self.stats['messages_discarded'] += 1

        # Save cache and stats
        self._save_cache()
        self._save_stats()

        return all_decisions

    def _call_llm_for_cleaning(self, prompt: str, messages: List[str], is_short_message_task: bool) -> List[CleaningDecision]:
        """
        Call the LLM provider for cleaning tasks.
        This is a custom method since the provider's generate_conversations is for conversation generation.
        """
        def _generate():
            # Create a simple chat completion call
            if hasattr(self.provider, 'client'):
                # xAI provider
                from xai_sdk.chat import system, user
                chat = self.provider.client.chat.create(
                    model=self.provider.xai_config['model'],
                    messages=[
                        system("You are a data cleaning assistant for Portuguese WhatsApp messages. Output only valid JSON."),
                        user(prompt)
                    ]
                )
                response = chat.sample()
                content = response.content if hasattr(response, 'content') else str(response)
                return content.strip()
            else:
                # Azure provider
                response = self.provider.client.chat.completions.create(
                    model=self.provider.azure_config['model'],
                    messages=[
                        {"role": "system", "content": "You are a data cleaning assistant for Portuguese WhatsApp messages. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent cleaning decisions
                    max_tokens=2000
                )
                return response.choices[0].message.content.strip()

        # Use retry logic from base class
        content = self.provider._retry_with_backoff(_generate)

        # Parse response based on task type
        if is_short_message_task:
            return self._parse_short_message_response(content, messages)
        else:
            return self._parse_filler_cleaning_response(content, messages)