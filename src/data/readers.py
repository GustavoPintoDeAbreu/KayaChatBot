import unsloth  # noqa: F401 — must be imported before transformers for Unsloth optimizations
import re
import os
import json
import random
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from transformers import AutoTokenizer
from datetime import datetime
from pathlib import Path
from unsloth.chat_templates import get_chat_template

# Import the LLM cleaning
try:
    from ..llm_providers import get_data_cleaning_llm
except ImportError:
    # Fallback for when running outside the package
    from llm_providers import get_data_cleaning_llm


@dataclass
class ChatMessage:
    date: str
    sender: str
    content: str


class WhatsAppReader:
    """
    Reads and cleans WhatsApp export files using LLM-based filtering.
    """

    def __init__(self, file_path: str, config: Optional[Dict] = None):
        self.file_path = file_path
        self.config = config
        self.cleaning_llm = None

        # Initialize LLM cleaning if config provided and enabled
        if config and config.get('data', {}).get('cleaning', {}).get('enabled', False):
            try:
                self.cleaning_llm = get_data_cleaning_llm(config)
                print("✅ LLM-based data cleaning enabled for WhatsApp reader")
            except Exception as e:
                error_msg = f"Failed to initialize LLM cleaning: {e}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)

        # Regex for "Date, Time - Sender: Message"
        # Example: 3/26/20, 15:28 - Gil João: Message
        self.date_pattern = re.compile(
            r"^(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}) - ([^:]+): (.+)$"
        )
        # Regex for system messages "Date, Time - Message"
        self.system_pattern = re.compile(
            r"^(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}) - (.+)$"
        )

    def _clean_text(self, text: str) -> Optional[str]:
        """Applies LLM-based cleaning rules."""
        if not text or not text.strip():
            return None

        text = text.strip()

        # Basic preprocessing (keep this as it's not content-based)
        if "<Media omitted>" in text:
            return None

        # Remove <This message was edited> tag
        text = text.replace("<This message was edited>", "")

        # Remove links
        text = re.sub(r"http[s]?://\S+", "", text)

        # Remove mentions (@\u2068...\u2069)
        text = re.sub(r"@\u2068.*?\u2069", "", text)

        text = text.strip()

        # If LLM cleaning is available, use it for content-based decisions
        if self.cleaning_llm:
            try:
                # For short messages, use LLM classification
                if len(text.split()) <= 5:  # Short messages need classification
                    decisions = self.cleaning_llm.clean_short_messages([text])
                    if decisions and decisions[0].is_substantive and decisions[0].cleaned_content:
                        return decisions[0].cleaned_content
                    else:
                        return None
                else:
                    # For longer messages, use filler cleaning
                    decisions = self.cleaning_llm.clean_filler_words([text])
                    if decisions and decisions[0].is_substantive and decisions[0].cleaned_content:
                        return decisions[0].cleaned_content
                    else:
                        return None
            except Exception as e:
                error_msg = f"LLM cleaning failed for message '{text[:50]}...': {e}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)

        # Fallback: if no LLM cleaning, apply basic rules (but this should not happen per requirements)
        else:
            # Filter out very short messages (< 3 words) unless they contain substance
            word_count = len(text.split())
            if word_count < 3 and text.lower() not in ['lol', 'ahah', 'ahahah', 'ok', 'ya', 'sim', 'não', 'nao']:
                # Allow common short responses but filter random short stuff
                if word_count == 1:
                    return None

        return text

    def read(self) -> List[ChatMessage]:
        """Reads the file and returns a list of ChatMessage objects."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        messages = []
        current_date = ""
        current_sender = ""
        current_content = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for standard message line
            match = self.date_pattern.match(line)
            if match:
                # Save previous message if exists
                if current_content:
                    full_text = " ".join(current_content)
                    cleaned = self._clean_text(full_text)
                    if cleaned:
                        messages.append(
                            ChatMessage(current_date, current_sender, cleaned)
                        )

                # Start new message
                current_date = match.group(1)
                current_sender = match.group(2)
                current_content = [match.group(3)]
                continue

            # Check for system message line (e.g. "added you") - usually skip for training persona
            # unless we want to know group events. For now, let's skip system messages
            # that don't have a colon (the regex above requires a colon for sender).
            # But we need to handle multi-line messages.

            # If it matches the date pattern but NOT the sender pattern, it's a system message
            sys_match = self.system_pattern.match(line)
            if sys_match and not match:
                # It's a system message or a message without a colon (rare)
                # We flush the previous message
                if current_content:
                    full_text = " ".join(current_content)
                    cleaned = self._clean_text(full_text)
                    if cleaned:
                        messages.append(
                            ChatMessage(current_date, current_sender, cleaned)
                        )
                current_content = []
                continue

            # If no date match, it's a continuation of the previous message
            if current_content:
                current_content.append(line)

        # Flush last message
        if current_content:
            full_text = " ".join(current_content)
            cleaned = self._clean_text(full_text)
            if cleaned:
                messages.append(ChatMessage(current_date, current_sender, cleaned))

        return messages


class InstagramReader:
    """
    Reads and cleans Instagram JSON export files using LLM-based filtering.
    Filters out noise (attachment spam, likes, shared posts) and extracts real conversations.
    """

    def __init__(self, json_path: str, config: Optional[Dict] = None):
        self.json_path = json_path
        self.config = config
        self.cleaning_llm = None

        # Initialize LLM cleaning if config provided and enabled
        if config and config.get('data', {}).get('cleaning', {}).get('enabled', False):
            try:
                self.cleaning_llm = get_data_cleaning_llm(config)
                print("✅ LLM-based data cleaning enabled for Instagram reader")
            except Exception as e:
                error_msg = f"Failed to initialize LLM cleaning: {e}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)

        # Patterns to identify noise
        self.attachment_pattern = re.compile(r".*sent an attachment\.$", re.IGNORECASE)
        self.liked_pattern = re.compile(r".*liked a message$", re.IGNORECASE)
        self.unsent_pattern = re.compile(r".*unsent .*", re.IGNORECASE)

    def _decode_unicode(self, text: str) -> str:
        """Decode unicode escape sequences like \\u00c3\\u00a3 -> ã"""
        if not text:
            return text
        # Instagram JSON sometimes has double-encoded unicode
        try:
            # Try to encode as latin1 then decode as utf8 (common Instagram encoding issue)
            return text.encode('latin1').decode('utf8')
        except:
            return text

    def _clean_text(self, text: str) -> Optional[str]:
        """Clean and filter Instagram message text using LLM."""
        if not text or not text.strip():
            return None

        text = text.strip()

        # Basic pattern-based filtering (keep this as it's format-based)
        # Skip attachment notifications
        if self.attachment_pattern.match(text):
            return None

        # Skip "liked a message" actions
        if self.liked_pattern.match(text):
            return None

        # Skip unsent messages
        if self.unsent_pattern.match(text):
            return None

        # Remove Instagram URLs (usually shared content, not conversation)
        text = re.sub(r'https?://(?:www\.)?instagram\.com/\S+', '', text)
        text = re.sub(r'https?://\S+', '', text)  # Remove other URLs too

        text = text.strip()

        # If LLM cleaning is available, use it for content-based decisions
        if self.cleaning_llm:
            try:
                # For short messages, use LLM classification
                if len(text.split()) <= 5:  # Short messages need classification
                    decisions = self.cleaning_llm.clean_short_messages([text])
                    if decisions and decisions[0].is_substantive and decisions[0].cleaned_content:
                        return decisions[0].cleaned_content
                    else:
                        return None
                else:
                    # For longer messages, use filler cleaning
                    decisions = self.cleaning_llm.clean_filler_words([text])
                    if decisions and decisions[0].is_substantive and decisions[0].cleaned_content:
                        return decisions[0].cleaned_content
                    else:
                        return None
            except Exception as e:
                error_msg = f"LLM cleaning failed for Instagram message '{text[:50]}...': {e}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)

        # Fallback: basic length filter (but this should not happen per requirements)
        else:
            # Filter very short messages (likely just reactions or noise)
            if len(text) < 3:
                return None

        # Decode unicode escapes
        text = self._decode_unicode(text)

        return text

    def read(self) -> List[ChatMessage]:
        """Reads Instagram JSON and returns list of ChatMessage objects."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"File not found: {self.json_path}")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        messages = []

        # Extract participants (for reference, though we'll use sender_name from messages)
        participants = data.get('participants', [])

        # Process messages
        for msg in data.get('messages', []):
            # Skip messages without content
            if 'content' not in msg:
                # Check if there's share_text we can use
                if 'share' in msg and 'share_text' in msg['share']:
                    content = msg['share'].get('share_text', '')
                else:
                    continue
            else:
                content = msg['content']

            # Clean the content
            cleaned_content = self._clean_text(content)
            if not cleaned_content:
                continue

            # Extract sender
            sender = msg.get('sender_name', 'Unknown')

            # Convert timestamp (milliseconds since epoch) to readable date
            timestamp_ms = msg.get('timestamp_ms', 0)
            if timestamp_ms:
                date_obj = datetime.fromtimestamp(timestamp_ms / 1000)
                date_str = date_obj.strftime('%m/%d/%y, %H:%M')
            else:
                date_str = 'Unknown'

            messages.append(ChatMessage(date_str, sender, cleaned_content))

        return messages


class ConversationFormatter:
    """
    Formats chat messages for LLM training with token-based chunking.
    """

    def __init__(
        self,
        messages: List[ChatMessage],
        tokenizer_name: str = "unsloth/Qwen3-14B-Instruct-bnb-4bit",
    ):
        self.messages = messages
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            # Fallback if tokenizer not available locally
            print(
                "Warning: Could not load tokenizer. Using character-based approximation."
            )
            self.tokenizer = None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters for English/Portuguese
            return len(text) // 4

    def to_instruction_chunks(
        self, system_prompt: str, max_tokens: int = 2048, overlap_tokens: int = 256
    ) -> List[Dict]:
        """
        Creates instruction-formatted chunks based on token count.
        Format: System prompt + chat history chunk.
        Now merges consecutive messages from the same sender.

        Args:
            system_prompt: The system instruction for the model.
            max_tokens: Maximum tokens per training example.
            overlap_tokens: Overlap between chunks for continuity.
        """
        dataset = []

        # Merge consecutive messages from same sender
        merged_messages = []
        if self.messages:
            current_sender = self.messages[0].sender
            current_content = [self.messages[0].content]
            current_date = self.messages[0].date

            for msg in self.messages[1:]:
                if msg.sender == current_sender:
                    # Same sender, merge
                    current_content.append(msg.content)
                else:
                    # Different sender, save and start new
                    merged_text = " ".join(current_content)
                    merged_messages.append(ChatMessage(current_date, current_sender, merged_text))
                    current_sender = msg.sender
                    current_content = [msg.content]
                    current_date = msg.date

            # Add last message
            merged_text = " ".join(current_content)
            merged_messages.append(ChatMessage(current_date, current_sender, merged_text))

        # Build the full conversation text from merged messages
        full_conversation = ""
        for msg in merged_messages:
            full_conversation += f"{msg.sender}: {msg.content}\n"

        # Calculate token budget (reserve space for system prompt and formatting)
        system_tokens = self._count_tokens(system_prompt)
        # Reserve ~200 tokens for chat template formatting (<|begin_of_text|>, etc.)
        available_tokens = max_tokens - system_tokens - 200

        if available_tokens <= 0:
            raise ValueError(f"System prompt too long! Uses {system_tokens} tokens.")

        # Split messages into token-based chunks
        current_chunk = []
        current_tokens = 0

        for msg in merged_messages:
            msg_text = f"{msg.sender}: {msg.content}\n"
            msg_tokens = self._count_tokens(msg_text)

            if current_tokens + msg_tokens > available_tokens and current_chunk:
                # Save current chunk and start new one with overlap
                chunk_text = "".join(current_chunk)
                dataset.append({"text": chunk_text, "system": system_prompt})

                # Keep last few messages for overlap
                overlap_chunk = []
                overlap_count = 0
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_count += self._count_tokens(current_chunk[i])
                    if overlap_count >= overlap_tokens:
                        break
                    overlap_chunk.insert(0, current_chunk[i])

                current_chunk = overlap_chunk
                current_tokens = overlap_count

            current_chunk.append(msg_text)
            current_tokens += msg_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = "".join(current_chunk)
            dataset.append({"text": chunk_text, "system": system_prompt})

        print(
            f"Created {len(dataset)} token-based chunks from {len(self.messages)} messages."
        )
        return dataset

    def format_for_chat(self, data_entry: Dict) -> str:
        """
        Formats a data entry using the tokenizer's chat template.
        Falls back to ChatML format if no tokenizer is available.
        """
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            # Use the model's own chat template (model-agnostic)
            messages = [
                {"role": "system", "content": data_entry["system"]},
                {"role": "user", "content": "Remember this conversation history."},
                {"role": "assistant", "content": data_entry["text"]},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Manual fallback: ChatML format (Qwen3 / most modern models)
            formatted = f"<|im_start|>system\n{data_entry['system']}<|im_end|>\n"
            formatted += f"<|im_start|>user\nRemember this conversation history.<|im_end|>\n"
            formatted += f"<|im_start|>assistant\n{data_entry['text']}<|im_end|>\n"
            return formatted


class SyntheticDatasetMerger:
    """
    Merges Kaya-specific and general Portuguese synthetic datasets.
    Applies ShareGPT formatting, shuffles, and splits train/validation.
    """

    def __init__(
        self,
        kaya_file: str = None,
        portuguese_file: Optional[str] = None,
        output_train: str = None,
        output_val: str = None,
        train_split: float = 0.9,
        kaya_ratio: float = 0.8,  # Target ratio of Kaya data (0.0-1.0)
        model_id: str = None,
        chat_template: str = "gemma-4",
    ):
        _data = Path(__file__).parent.parent.parent / "data"
        self.kaya_file = Path(kaya_file) if kaya_file else _data / "synthetic_kaya.jsonl"
        self.portuguese_file = Path(portuguese_file) if portuguese_file else None
        self.output_train = Path(output_train) if output_train else _data / "train_synthetic.jsonl"
        self.output_val = Path(output_val) if output_val else _data / "val_synthetic.jsonl"
        self.train_split = train_split
        self.kaya_ratio = kaya_ratio
        self.chat_template = chat_template
        self.tokenizer = None

        if model_id:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.tokenizer = get_chat_template(tokenizer, chat_template)
                print(f"✅ Loaded tokenizer for {model_id} with '{chat_template}' template")
            except Exception as e:
                print(f"Warning: Could not load tokenizer for {model_id}: {e}")
        else:
            print(f"Warning: No model_id provided. Using manual '{chat_template}' template fallback.")

    def load_conversations(self, file_path: Path) -> List[Dict]:
        """Load conversations from JSONL file."""
        conversations = []

        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            return conversations

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    conv = json.loads(line)
                    conversations.append(conv)
                except Exception as e:
                    print(f"⚠️  Error parsing line: {e}")
                    continue

        return conversations

    def clean_filler_words(self, text: str) -> str:
        """
        Clean excessive filler words from responses.
        - Remove filler words from the start of responses
        - Limit to max 1 occurrence of each filler per response
        """
        import re

        # Define filler patterns (case-insensitive)
        fillers = ['ahahah', 'ahah', 'ahahha', 'wtf', 'lmao', 'lol']

        # Remove fillers from the very start of the text (with optional punctuation)
        for filler in fillers:
            # Match filler at start with optional comma/space
            text = re.sub(rf'^{filler}[,\s]*', '', text, flags=re.IGNORECASE)

        # Limit each filler to max 1 occurrence
        for filler in fillers:
            # Find all occurrences
            pattern = re.compile(rf'\b{filler}\b', flags=re.IGNORECASE)
            matches = list(pattern.finditer(text))

            # If more than 1, keep only the last one (usually most natural)
            if len(matches) > 1:
                # Remove all but the last
                for match in matches[:-1]:
                    # Replace with space to avoid word concatenation
                    start, end = match.span()
                    text = text[:start] + ' ' * (end - start) + text[end:]

        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def format_conversation(self, conversation: Dict) -> Optional[str]:
        """Format a conversation using the tokenizer's chat template (model-agnostic)."""

        # Extract conversation turns
        turns = conversation.get('conversations', [])

        if not turns or len(turns) < 2:
            return None

        # Get system prompt for the conversation
        source = conversation.get('source', '')
        if source in ('synthetic_kaya', 'synthetic_targeted'):
            # Inject the same system prompt used at inference time for all Kaya conversations.
            # This must match config.yaml data.system_prompt to avoid train/inference mismatch.
            system_prompt = (
                "És o bot assistente do grupo de amigos 'Kaya'. "
                "Tens memória de factos, eventos e pessoas que aprendeste através das conversas passadas do grupo. "
                "Quando te fazem perguntas, respondes com base no que sabes dessas conversas. "
                "Podes conversar de forma casual e amigável. "
                "Fala em português europeu ou inglês de forma natural. "
                "Não és um membro do grupo — és um bot com acesso à memória coletiva do grupo. "
                "Nunca fales na primeira pessoa sobre experiências pessoais com membros do grupo. "
                "Refere-te sempre aos membros na terceira pessoa. "
                "Não digas 'o meu amigo', 'vivemos juntos' ou outras afirmações pessoais. "
                "Aprendeste sobre eles através de conversas do grupo, não por experiência pessoal direta. "
                "Quando respondes a saudações casuais ou perguntas sobre o teu estado, mantém o papel de assistente "
                "— não simules ter um dia de trabalho, planos pessoais, atividades físicas ou outros aspetos de uma vida humana. "
                "Tens memória do grupo, não uma vida própria. "
                "Quando alguém te cumprimenta (ex: 'Olá!', 'How are you?', 'Tudo bem?'), responde sempre como assistente: "
                "'Olá! Estou aqui para ajudar com questões sobre o grupo Kaya.' "
                "— nunca inventes um estado físico, humor, planos ou atividades pessoais. "
                "Não tens preferências pessoais como equipas de futebol, comida favorita ou opiniões próprias "
                "— se te perguntarem, diz claramente que és um bot sem preferências e redireciona para o que os membros do grupo gostam. "
                "Se não tiveres informação verificada sobre algo (evento específico, data, detalhe), diz claramente que não tens essa informação "
                "— nunca inventes detalhes plausíveis mas não confirmados. "
                "Limita sempre as tuas respostas aos factos que conheces das conversas e da base de conhecimento "
                "— não acrescentes detalhes especulativos. "
                "Para tópicos sensíveis (violência, doença, acidentes), responde com empatia e tom sério "
                "— nunca uses linguagem casual ou inapropriada nesses contextos."
            )
        else:
            # Use provided system prompt for Portuguese instruction data
            system_prompt = conversation.get('system', '')

        # Build messages for chat template
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation turns
        for turn in turns:
            role = turn.get('role', '')
            content = turn.get('content', '')

            if not content:
                continue

            # Apply filler word cleaning to assistant responses
            if role in ['assistant', 'gpt', 'bot']:
                content = self.clean_filler_words(content)

            # Map role names
            if role in ['user', 'human']:
                messages.append({"role": "user", "content": content})
            elif role in ['assistant', 'gpt', 'bot']:
                messages.append({"role": "assistant", "content": content})

        # Validate: must have at least one user and one assistant message
        has_user = any(m['role'] == 'user' for m in messages)
        has_assistant = any(m['role'] == 'assistant' for m in messages)

        if not (has_user and has_assistant):
            return None

        # Apply chat template
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                return formatted
            except Exception as e:
                print(f"⚠️  Error applying chat template: {e}")
                return None
        else:
            # Manual fallback: Gemma 4 format (<|turn>role\ncontent<turn|>\n)
            # System message is prepended to first user turn as per Gemma 4 template
            formatted = ""
            system_prefix = ""

            for i, msg in enumerate(messages):
                role = msg['role']
                content = msg['content']

                if role == 'system':
                    system_prefix = content + '\n\n'
                    continue

                # Map assistant -> model (Gemma 4 uses 'model' role)
                if role == 'assistant':
                    role = 'model'

                prefix = system_prefix if i == 0 or (i == 1 and system_prefix) else ""
                system_prefix = ""  # Only use once
                formatted += f"<|turn>{role}\n{prefix}{content}<turn|>\n"

            return formatted.strip() if formatted else None

    def merge_and_split(self) -> tuple[int, int]:
        """Merge datasets, apply formatting, shuffle, and split."""

        print("=" * 60)
        print("🔄 MERGING SYNTHETIC DATASETS")
        print("=" * 60)

        # Load Kaya-specific data
        print(f"\n📂 Loading Kaya-specific data from {self.kaya_file.name}...")
        kaya_conversations = self.load_conversations(self.kaya_file)
        print(f"✅ Loaded {len(kaya_conversations)} Kaya conversations")

        # Load general Portuguese data (if provided)
        portuguese_conversations = []
        if self.portuguese_file:
            print(f"\n📂 Loading Portuguese data from {self.portuguese_file.name}...")
            portuguese_conversations = self.load_conversations(self.portuguese_file)
            print(f"✅ Loaded {len(portuguese_conversations)} Portuguese conversations")
        else:
            print(f"\n⚠️  Skipping Portuguese data (using 100% RAG-aware Kaya data)")

        # Apply ratio sampling to balance Kaya vs Portuguese (if Portuguese data exists)
        if portuguese_conversations:
            print(f"\n🎯 Applying ratio: {self.kaya_ratio*100:.0f}% Kaya / {(1-self.kaya_ratio)*100:.0f}% Portuguese")

            # Use all Kaya data (it's precious!)
            kaya_count = len(kaya_conversations)

            # Calculate how many Portuguese examples we need for the target ratio
            # Formula: kaya_count / total = kaya_ratio
            # So: total = kaya_count / kaya_ratio
            # And: portuguese_count = total - kaya_count
            if self.kaya_ratio > 0 and kaya_count > 0:
                target_total = int(kaya_count / self.kaya_ratio)
                portuguese_needed = target_total - kaya_count

                # Sample Portuguese data to match ratio
                if portuguese_needed < len(portuguese_conversations):
                    import random
                    random.seed(3407)
                    portuguese_conversations = random.sample(portuguese_conversations, portuguese_needed)
                    print(f"   🎲 Sampled {portuguese_needed} Portuguese examples (from {len(self.load_conversations(self.portuguese_file))} available)")
                else:
                    print(f"   ⚠️  Using all {len(portuguese_conversations)} Portuguese examples (needed {portuguese_needed})")

        # Combine all conversations
        all_conversations = kaya_conversations + portuguese_conversations
        print(f"\n📊 Total conversations: {len(all_conversations)}")
        
        if len(all_conversations) == 0:
            print("\n❌ Error: No conversations to merge!")
            print("   Make sure you have run:")
            print("   1. python src/data/generate_synthetic_data.py")
            print("   2. python src/data/prepare_portuguese_data.py")
            return 0, 0

        print(f"   - Kaya-specific: {len(kaya_conversations)} ({len(kaya_conversations)/len(all_conversations)*100:.1f}%)")
        print(f"   - General Portuguese: {len(portuguese_conversations)} ({len(portuguese_conversations)/len(all_conversations)*100:.1f}%)")

        # Format all conversations
        print(f"\n🔄 Formatting conversations with chat template...")
        formatted_data = []

        for i, conv in enumerate(all_conversations):
            formatted = self.format_conversation(conv)

            if formatted:
                formatted_data.append({
                    'formatted_text': formatted,
                    'source': conv.get('source', 'unknown'),
                    'original': conv  # Keep original for debugging
                })

        print(f"✅ Successfully formatted {len(formatted_data)}/{len(all_conversations)} conversations")

        # Shuffle
        print(f"\n🎲 Shuffling dataset...")
        import random
        random.seed(3407)
        random.shuffle(formatted_data)

        # Split train/validation
        split_idx = int(len(formatted_data) * self.train_split)
        train_data = formatted_data[:split_idx]
        val_data = formatted_data[split_idx:]

        print(f"✅ Split: {len(train_data)} train, {len(val_data)} validation ({self.train_split*100:.0f}/{(1-self.train_split)*100:.0f})")

        # Save train set
        print(f"\n💾 Saving training set to {self.output_train.name}...")
        self.output_train.parent.mkdir(exist_ok=True, parents=True)

        with open(self.output_train, 'w', encoding='utf-8') as f:
            for entry in train_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"✅ Saved {len(train_data)} training examples")

        # Save validation set
        print(f"\n💾 Saving validation set to {self.output_val.name}...")
        with open(self.output_val, 'w', encoding='utf-8') as f:
            for entry in val_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"✅ Saved {len(val_data)} validation examples")

        # Statistics
        print("\n" + "=" * 60)
        print("📊 MERGE STATISTICS")
        print("=" * 60)
        print(f"Total input: {len(all_conversations)}")
        print(f"Successfully formatted: {len(formatted_data)} ({len(formatted_data)/len(all_conversations)*100:.0f}%)")
        print(f"Training examples: {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")

        # Source breakdown in training set
        kaya_count = sum(1 for x in train_data if x['source'] == 'synthetic_kaya')
        port_count = sum(1 for x in train_data if x['source'] == 'alpaca-portuguese')

        print(f"\nTraining set composition:")
        if len(train_data) > 0:
            print(f"  Kaya-specific: {kaya_count} ({kaya_count/len(train_data)*100:.1f}%)")
            print(f"  General Portuguese: {port_count} ({port_count/len(train_data)*100:.1f}%)")
        else:
            print(f"  (no training examples — all went to validation due to small dataset)")

        print(f"\n✅ Merge complete!")
        print(f"   Train: {self.output_train}")
        print(f"   Val: {self.output_val}")

        return len(train_data), len(val_data)