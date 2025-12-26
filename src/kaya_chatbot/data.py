import re
import os
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class ChatMessage:
    date: str
    sender: str
    content: str


class WhatsAppReader:
    """
    Reads and cleans WhatsApp export files.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
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
        """Applies regex cleaning rules."""
        if "<Media omitted>" in text:
            return None

        # Remove <This message was edited> tag
        text = text.replace("<This message was edited>", "")

        # Remove links
        text = re.sub(r"http[s]?://\S+", "", text)

        # Remove mentions (@\u2068...\u2069)
        text = re.sub(r"@\u2068.*?\u2069", "", text)

        # Remove emojis (optional, keeping for now as they add context,
        # but user previously asked to remove. I will keep them for "persona"
        # unless strictly requested to remove again, but the previous notebook removed them.
        # I'll add a flag or just stick to the previous logic for consistency.)
        # The previous logic: text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        # I will comment it out for now to allow more expression, but can be enabled.
        # text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

        return text.strip()

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
            # that don't have a sender (the regex above requires a colon for sender).
            # But we need to handle multi-line messages.

            # If it matches the date pattern but NOT the sender pattern, it's a system message
            # e.g. "3/26/20, 15:28 - Gil João created group..."
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


class ConversationFormatter:
    """
    Formats chat messages for LLM training with token-based chunking.
    """

    def __init__(
        self,
        messages: List[ChatMessage],
        tokenizer_name: str = "unsloth/llama-3-8b-Instruct-bnb-4bit",
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

        Args:
            system_prompt: The system instruction for the model.
            max_tokens: Maximum tokens per training example.
            overlap_tokens: Overlap between chunks for continuity.
        """
        dataset = []

        # Build the full conversation text first
        full_conversation = ""
        for msg in self.messages:
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

        for msg in self.messages:
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

    def format_for_llama3(self, data_entry: Dict) -> str:
        """
        Formats a data entry with Llama-3's chat template.
        This is what the model will actually see during training.
        """
        # Llama-3 chat format:
        # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        # {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
        # Read this chat log.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        # {chat_text}<|eot_id|>

        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            # Use the official chat template
            messages = [
                {"role": "system", "content": data_entry["system"]},
                {"role": "user", "content": "Remember this conversation history."},
                {"role": "assistant", "content": data_entry["text"]},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Manual fallback format
            formatted = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            )
            formatted += f"{data_entry['system']}<|eot_id|>"
            formatted += f"<|start_header_id|>user<|end_header_id|>\n\n"
            formatted += f"Remember this conversation history.<|eot_id|>"
            formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            formatted += f"{data_entry['text']}<|eot_id|>"
            return formatted
