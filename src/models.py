"""
Pydantic models for type safety across the KayaChatBot pipeline.

This module defines data models for each phase of the pipeline:
- Raw data sources (WhatsApp, Instagram)
- Processed data (cleaned messages, finetune chunks)
- Synthetic generation (conversations)
- Training format (formatted examples)
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationInfo


# ============================================================================
# Raw Data Models
# ============================================================================

class WhatsAppMessage(BaseModel):
    """Raw WhatsApp message from TXT export.
    
    Example:
        [26/3/20, 15:28] Gil João: Message here
    """
    timestamp: datetime
    sender: str
    text: str
    source: Literal["whatsapp"] = "whatsapp"
    
    model_config = ConfigDict(frozen=False)
    
    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message text cannot be empty")
        return v.strip()


class InstagramMessage(BaseModel):
    """Raw Instagram message from JSON export.
    
    Instagram exports use double-encoded UTF-8 (latin1 → utf8).
    """
    timestamp_ms: int
    sender_name: str
    content: Optional[str] = None
    source: Literal["instagram"] = "instagram"
    
    # Optional fields for media/reactions
    photos: Optional[List[Dict[str, Any]]] = None
    videos: Optional[List[Dict[str, Any]]] = None
    reactions: Optional[List[Dict[str, Any]]] = None
    share: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(frozen=False)
    
    @property
    def timestamp(self) -> datetime:
        """Convert timestamp_ms to datetime."""
        return datetime.fromtimestamp(self.timestamp_ms / 1000)


# ============================================================================
# Processed Data Models
# ============================================================================

class CleanedMessage(BaseModel):
    """Cleaned and standardized message from any source.
    
    This is the unified format after preprocessing WhatsApp and Instagram data.
    """
    timestamp: datetime
    sender: str
    text: str
    source: Literal["whatsapp", "instagram"]
    
    model_config = ConfigDict(frozen=False)
    
    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message text cannot be empty")
        return v.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with ISO timestamp for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "text": self.text,
            "source": self.source
        }


class FinetuneChunk(BaseModel):
    """A chunk of messages for synthetic data generation.
    
    Messages are chunked into ~50K token groups to fit within
    Azure GPT-4.1-mini context window for synthetic generation.
    """
    chunk_id: int = Field(ge=0)
    messages: List[CleanedMessage]
    text: str  # Formatted text representation for LLM
    token_count: int = Field(ge=0)
    message_count: int = Field(ge=0)
    
    model_config = ConfigDict(frozen=False)
    
    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: List[CleanedMessage]) -> List[CleanedMessage]:
        if not v:
            raise ValueError("Chunk must contain at least one message")
        return v
    
    @field_validator("message_count")
    @classmethod
    def validate_message_count(cls, v: int, info: ValidationInfo) -> int:
        """Ensure message_count matches actual messages."""
        if "messages" in info.data and v != len(info.data["messages"]):
            raise ValueError(f"message_count ({v}) doesn't match messages length ({len(info.data['messages'])})")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "text": self.text,
            "token_count": self.token_count,
            "message_count": self.message_count
        }


# ============================================================================
# Synthetic Generation Models
# ============================================================================

class ConversationTurn(BaseModel):
    """A single turn in a conversation (user or assistant)."""
    role: Literal["user", "assistant", "system"]
    content: str
    
    model_config = ConfigDict(frozen=False)
    
    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Conversation turn content cannot be empty")
        return v.strip()


class SyntheticConversation(BaseModel):
    """A synthetic multi-turn conversation generated from message chunks.
    
    Uses ShareGPT format compatible with training libraries.
    """
    conversations: List[ConversationTurn] = Field(min_length=2)
    source: Literal["synthetic_kaya", "synthetic_portuguese"] = "synthetic_kaya"
    chunk_id: Optional[int] = None
    
    model_config = ConfigDict(frozen=False)
    
    @field_validator("conversations")
    @classmethod
    def validate_conversation_structure(cls, v: List[ConversationTurn]) -> List[ConversationTurn]:
        """Ensure conversation has valid turn-taking structure."""
        if len(v) < 2:
            raise ValueError("Conversation must have at least 2 turns")
        
        # Check alternating user/assistant (system can be at start)
        non_system_turns = [turn for turn in v if turn.role != "system"]
        if not non_system_turns:
            raise ValueError("Conversation must have non-system turns")
        
        # First non-system turn should be user
        if non_system_turns[0].role != "user":
            raise ValueError("First non-system turn must be from user")
        
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result = {
            "conversations": [{"role": turn.role, "content": turn.content} for turn in self.conversations],
            "source": self.source
        }
        if self.chunk_id is not None:
            result["chunk_id"] = self.chunk_id
        return result


# ============================================================================
# Training Format Models
# ============================================================================

class TrainingExample(BaseModel):
    """A formatted training example with Llama-3.1 chat template applied.
    
    This is the final format ready for fine-tuning.
    """
    formatted_text: str  # Text with chat template applied
    source: Literal["synthetic_kaya", "synthetic_portuguese"]
    original: Dict[str, Any]  # Original conversation data
    
    model_config = ConfigDict(frozen=False)
    
    @field_validator("formatted_text")
    @classmethod
    def formatted_text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Formatted text cannot be empty")
        return v
    
    @field_validator("formatted_text")
    @classmethod
    def check_chat_template(cls, v: str) -> str:
        """Validate that Llama-3.1 chat template tokens are present."""
        required_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>"]
        if not all(token in v for token in required_tokens):
            raise ValueError("Formatted text missing required Llama-3.1 chat template tokens")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "formatted_text": self.formatted_text,
            "source": self.source,
            "original": self.original
        }
