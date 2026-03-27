"""
Build vector database for RAG from cleaned message data.
Creates chunks of conversation history and stores them with embeddings.
"""

import json
import os
import re
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
from collections import defaultdict
import chromadb
from sentence_transformers import SentenceTransformer
import tiktoken

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# RAG Configuration
RAG_CONFIG = config['rag']
CHUNK_SIZE_TOKENS = RAG_CONFIG['chunk_size_tokens']
CHUNK_OVERLAP_TOKENS = RAG_CONFIG['chunk_overlap_tokens']
EMBEDDING_MODEL = RAG_CONFIG['embedding_model']

# Detect if running in Docker
_BASE_DIR = Path("/app") if os.path.exists('/app') else Path(__file__).parent.parent.parent
DATA_DIR = _BASE_DIR / "data"
DB_DIR = _BASE_DIR / "data" / "rag_db"

# Input/Output paths
INPUT_CLEANED = DATA_DIR / "all_messages_cleaned.jsonl"
DB_DIR.mkdir(exist_ok=True)

# Initialize tokenizer for token counting
tokenizer = tiktoken.encoding_for_model("gpt-4")


class ConversationChunker:
    """Chunk conversation messages into retrievable units."""

    def __init__(self, chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
                 overlap_tokens: int = CHUNK_OVERLAP_TOKENS):
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(tokenizer.encode(text))

    def extract_participants(self, messages: List[Dict]) -> List[str]:
        """Extract unique participants from messages."""
        return list(set(msg['sender'] for msg in messages))

    def extract_mentioned_people(self, text: str) -> List[str]:
        """Extract mentioned people from text using heuristics."""
        # Common group member names (case insensitive)
        group_members = [
            'peter', 'gil', 'gustavo', 'david', 'manuel', 'carnall', 'frederico',
            'mateus', 'rafa', 'bernardo', 'chamusca', 'gilao', 'pedro', 'kaya'
        ]

        mentioned = []
        text_lower = text.lower()

        for member in group_members:
            if member in text_lower:
                mentioned.append(member)

        return list(set(mentioned))

    def create_conversation_chunks(self, messages: List[Dict]) -> List[Dict]:
        """Create conversation chunks with temporal grouping and metadata."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_start_time = None
        chunk_end_time = None

        for msg in messages:
            # Format message for chunking
            formatted_msg = f"{msg['sender']}: {msg['text']}"
            msg_tokens = self.count_tokens(formatted_msg)

            # Check if we need to start a new chunk
            if current_tokens + msg_tokens > self.chunk_size_tokens and current_chunk:
                # Create chunk from current messages
                chunk_text = '\n'.join([
                    f"{m['sender']}: {m['text']}" for m in current_chunk
                ])

                # Extract metadata
                participants = self.extract_participants(current_chunk)
                all_mentioned = []
                for m in current_chunk:
                    all_mentioned.extend(self.extract_mentioned_people(m['text']))
                mentioned = list(set(all_mentioned))

                chunks.append({
                    'id': f"chunk_{len(chunks)}",
                    'text': chunk_text,
                    'messages': current_chunk,
                    'token_count': current_tokens,
                    'message_count': len(current_chunk),
                    'participants': participants,
                    'mentioned': mentioned,
                    'timestamp_start': chunk_start_time,
                    'timestamp_end': chunk_end_time,
                    'metadata': {
                        'participants': ','.join(participants),
                        'mentioned': ','.join(mentioned),
                        'message_count': len(current_chunk),
                        'token_count': current_tokens,
                        'timestamp_start': chunk_start_time,
                        'timestamp_end': chunk_end_time
                    }
                })

                # Start new chunk with overlap
                overlap_messages = self._get_overlap_messages(current_chunk, self.overlap_tokens)
                current_chunk = overlap_messages
                current_tokens = sum(self.count_tokens(f"{m['sender']}: {m['text']}") for m in overlap_messages)
                chunk_start_time = overlap_messages[0]['timestamp'] if overlap_messages else None

            # Add message to current chunk
            current_chunk.append(msg)
            current_tokens += msg_tokens

            # Update timestamps
            if not chunk_start_time:
                chunk_start_time = msg['timestamp']
            chunk_end_time = msg['timestamp']

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join([
                f"{m['sender']}: {m['text']}" for m in current_chunk
            ])

            participants = self.extract_participants(current_chunk)
            all_mentioned = []
            for m in current_chunk:
                all_mentioned.extend(self.extract_mentioned_people(m['text']))
            mentioned = list(set(all_mentioned))

            chunks.append({
                'id': f"chunk_{len(chunks)}",
                'text': chunk_text,
                'messages': current_chunk,
                'token_count': current_tokens,
                'message_count': len(current_chunk),
                'participants': participants,
                'mentioned': mentioned,
                'timestamp_start': chunk_start_time,
                'timestamp_end': chunk_end_time,
                'metadata': {
                    'participants': ','.join(participants),
                    'mentioned': ','.join(mentioned),
                    'message_count': len(current_chunk),
                    'token_count': current_tokens,
                    'timestamp_start': chunk_start_time,
                    'timestamp_end': chunk_end_time
                }
            })

        return chunks

    def _get_overlap_messages(self, messages: List[Dict], overlap_tokens: int) -> List[Dict]:
        """Get messages for overlap from the end of current chunk."""
        overlap_messages = []
        tokens_used = 0

        # Take messages from the end until we reach overlap_tokens
        for msg in reversed(messages):
            formatted = f"{msg['sender']}: {msg['text']}"
            msg_tokens = self.count_tokens(formatted)

            if tokens_used + msg_tokens > overlap_tokens:
                break

            overlap_messages.insert(0, msg)
            tokens_used += msg_tokens

        return overlap_messages


class VectorDatabaseBuilder:
    """Build and populate vector database for RAG."""

    def __init__(self, db_path: Path, embedding_model: str = EMBEDDING_MODEL):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = None
        self.encoder = None

    def initialize(self):
        """Initialize the vector database and embedding model."""
        print(f"🔧 Initializing vector database at {self.db_path}")

        # Load embedding model FIRST to get embedding dimension
        print(f"🤖 Loading embedding model: {self.embedding_model}")
        # GTE models require trust_remote_code=True
        self.encoder = SentenceTransformer(self.embedding_model, trust_remote_code=True)
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"✅ Embedding model loaded (dimension: {embedding_dim})")

        # Create or get collection
        collection_name = "kaya_conversations"
        collection_needs_rebuild = False
        
        try:
            existing_collection = self.client.get_collection(name=collection_name)
            print(f"📚 Found existing collection '{collection_name}' with {existing_collection.count()} documents")
            
            # Check if embedding dimension matches by testing with a dummy embedding
            test_embedding = self.encoder.encode(["test"], normalize_embeddings=True)[0].tolist()
            test_dim = len(test_embedding)
            
            # Try adding a test document to check dimension compatibility
            try:
                # This will fail if dimensions don't match
                import uuid
                test_id = f"__test_dim_check_{uuid.uuid4().hex[:8]}__"
                existing_collection.add(
                    ids=[test_id],
                    embeddings=[test_embedding],
                    documents=["dimension test"],
                    metadatas=[{"test": True}]
                )
                # If successful, delete the test document
                existing_collection.delete(ids=[test_id])
                print(f"✅ Dimension check passed (768)")
                self.collection = existing_collection
            except Exception as e:
                if "dimension" in str(e).lower():
                    print(f"⚠️  Dimension mismatch detected: {e}")
                    collection_needs_rebuild = True
                else:
                    raise
        except Exception as e:
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                print(f"📚 Collection '{collection_name}' does not exist")
                collection_needs_rebuild = True
            else:
                print(f"❌ Unexpected error: {e}")
                raise
        
        # Rebuild collection if needed
        if collection_needs_rebuild:
            try:
                print(f"🗑️  Deleting old collection '{collection_name}'...")
                self.client.delete_collection(name=collection_name)
            except:
                pass  # Collection might not exist
            
            print(f"📚 Creating new collection '{collection_name}' with dimension {embedding_dim}...")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Kaya chatbot conversation chunks for RAG",
                    "embedding_dimension": embedding_dim,
                    "embedding_model": self.embedding_model
                }
            )
            print(f"✅ Collection created")

    def add_chunks(self, chunks: List[Dict]):
        """Add conversation chunks to the vector database."""
        if not chunks:
            print("⚠️  No chunks to add")
            return

        print(f"📥 Adding {len(chunks)} chunks to vector database...")

        # Prepare data for batch insertion
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            ids.append(chunk['id'])
            documents.append(chunk['text'])
            metadatas.append(chunk['metadata'])

        # Generate embeddings in batches
        print("🧮 Generating embeddings...")
        batch_size = 32
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_embeddings = self.encoder.encode(batch_docs, show_progress_bar=False)
            embeddings.extend(batch_embeddings.tolist())

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"✅ Added {len(chunks)} chunks to vector database")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection.name
        }


def load_cleaned_messages(limit: int = None) -> List[Dict]:
    """Load cleaned messages from file."""
    messages = []

    if not INPUT_CLEANED.exists():
        raise FileNotFoundError(f"Cleaned messages file not found: {INPUT_CLEANED}")

    print(f"📖 Loading cleaned messages from {INPUT_CLEANED}")

    with open(INPUT_CLEANED, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                msg = json.loads(line.strip())
                messages.append(msg)

                if limit and len(messages) >= limit:
                    break
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed line {line_num}: {e}")
                continue

    print(f"✅ Loaded {len(messages)} messages")
    return messages


def main():
    """Main RAG database building pipeline."""
    print("=" * 60)
    print("🚀 RAG VECTOR DATABASE BUILDER")
    print("=" * 60)

    # Check if cleaned messages exist
    if not INPUT_CLEANED.exists():
        print(f"❌ Cleaned messages file not found: {INPUT_CLEANED}")
        print("   Run extract_all_messages.py first!")
        return

    # Load messages
    messages = load_cleaned_messages()

    if not messages:
        print("❌ No messages loaded!")
        return

    # Create chunks
    print(f"\n🔨 Creating conversation chunks (max {CHUNK_SIZE_TOKENS} tokens each)")
    chunker = ConversationChunker(
        chunk_size_tokens=CHUNK_SIZE_TOKENS,
        overlap_tokens=CHUNK_OVERLAP_TOKENS
    )
    chunks = chunker.create_conversation_chunks(messages)

    print(f"✅ Created {len(chunks)} conversation chunks")

    # Initialize vector database
    builder = VectorDatabaseBuilder(DB_DIR, EMBEDDING_MODEL)
    builder.initialize()

    # Add chunks to database
    builder.add_chunks(chunks)

    # Statistics
    stats = builder.get_stats()
    print("\n" + "=" * 60)
    print("📊 RAG DATABASE STATISTICS")
    print("=" * 60)
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Collection: {stats['collection_name']}")
    print(f"Database location: {DB_DIR}")

    # Chunk statistics
    total_messages = sum(chunk['message_count'] for chunk in chunks)
    total_tokens = sum(chunk['token_count'] for chunk in chunks)
    avg_messages_per_chunk = total_messages / len(chunks)
    avg_tokens_per_chunk = total_tokens / len(chunks)

    print(f"Total messages in chunks: {total_messages}")
    print(f"Total tokens in chunks: {total_tokens:,}")
    print(f"Average messages per chunk: {avg_messages_per_chunk:.1f}")
    print(f"Average tokens per chunk: {avg_tokens_per_chunk:.0f}")

    # Metadata statistics
    all_participants = set()
    all_mentioned = set()
    for chunk in chunks:
        all_participants.update(chunk['participants'])
        all_mentioned.update(chunk['mentioned'])

    print(f"Unique participants: {len(all_participants)}")
    print(f"Unique mentioned people: {len(all_mentioned)}")

    print("\n✅ RAG database build complete!")
    print(f"\nNext steps:")
    print(f"  1. Test retrieval with src/chat/retriever.py")
    print(f"  2. Update src/chat/chat.py to use RAG")
    print(f"  3. Run chat interface to test RAG functionality")


if __name__ == "__main__":
    main()