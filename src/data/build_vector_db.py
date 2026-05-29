"""
Build vector database for RAG from cleaned message data.
Creates chunks of conversation history and stores them with embeddings.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import chromadb
from sentence_transformers import SentenceTransformer
import tiktoken

# Make src/ importable when run directly as a script (python src/data/build_vector_db.py)
# or as a subprocess from incremental_update.py.
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load configuration
CONFIG_PATH = _REPO_ROOT / "config.yaml"
from src.config_loader import load_config
config = load_config(str(CONFIG_PATH))

# RAG Configuration
RAG_CONFIG = config['rag']
CHUNK_SIZE_TOKENS = RAG_CONFIG['chunk_size_tokens']
CHUNK_OVERLAP_TOKENS = RAG_CONFIG['chunk_overlap_tokens']
EMBEDDING_MODEL = RAG_CONFIG['embedding_model']

# Detect if running in Docker
_BASE_DIR = Path("/app") if os.path.exists('/app') else Path(__file__).parent.parent.parent
DATA_DIR = _BASE_DIR / "data"
DB_DIR = _BASE_DIR / "data" / "rag_db"

# Load group members from JSON (single source of truth)
_members_file = DATA_DIR / "group_members.json"
if _members_file.exists():
    with open(_members_file, 'r', encoding='utf-8') as _f:
        _members_data = json.load(_f)
    GROUP_MEMBERS = [
        alias.lower()
        for m in _members_data.get('members', [])
        for alias in m.get('aliases', [])
    ]
else:
    GROUP_MEMBERS = [
        'peter', 'gil', 'gustavo', 'david', 'manuel', 'carnall', 'frederico',
        'mateus', 'rafa', 'bernardo', 'chamusca', 'gilao', 'pedro'
    ]

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
        mentioned = []
        text_lower = text.lower()

        # Word-boundary match so short aliases don't fire inside unrelated words.
        for member in GROUP_MEMBERS:
            if re.search(rf"\b{re.escape(member)}\b", text_lower):
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
        """Initialize the vector database and embedding model.

        The conversation collection is always rebuilt from scratch with cosine
        space so its distance metric stays consistent with the normalized
        bge-m3 embeddings. bge-m3 is trained for cosine similarity; the previous
        default (L2 on un-normalized vectors) degraded ranking and let
        ``1 - distance`` go negative. Always rebuilding also avoids duplicate
        chunk IDs from re-adding to an existing collection.
        """
        print(f"🔧 Initializing vector database at {self.db_path}")

        # Load embedding model FIRST to get embedding dimension
        print(f"🤖 Loading embedding model: {self.embedding_model}")
        self.encoder = SentenceTransformer(self.embedding_model, trust_remote_code=True)
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"✅ Embedding model loaded (dimension: {embedding_dim})")

        collection_name = "kaya_conversations"

        try:
            self.client.delete_collection(name=collection_name)
            print(f"🗑️  Deleted existing collection '{collection_name}'")
        except Exception:
            pass  # Collection did not exist yet

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={
                "description": "Kaya chatbot conversation chunks for RAG",
                "embedding_dimension": embedding_dim,
                "embedding_model": self.embedding_model,
                "hnsw:space": "cosine",
            },
        )
        print(f"✅ Collection created (dimension {embedding_dim}, space=cosine)")

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
            batch_embeddings = self.encoder.encode(batch_docs, show_progress_bar=False, normalize_embeddings=True)
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


class KnowledgeBaseBuilder:
    """Build and populate a curated knowledge base ChromaDB collection from group_knowledge.json."""

    def __init__(self, db_path: Path, embedding_model: str = EMBEDDING_MODEL):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = None
        self.encoder = None

    def initialize(self, collection_name: str = "kaya_knowledge_base"):
        """Initialize the KB collection and embedding model (reusing already-loaded encoder if possible)."""
        print(f"🔧 Initializing knowledge base collection '{collection_name}'")

        self.encoder = SentenceTransformer(self.embedding_model, trust_remote_code=True)

        # Always rebuild the KB collection so it stays in sync with the JSON file
        try:
            self.client.delete_collection(name=collection_name)
            print(f"🗑️  Deleted existing collection '{collection_name}'")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={
                "description": "Curated group knowledge facts for Kaya chatbot",
                "embedding_model": self.embedding_model,
                "hnsw:space": "cosine",
            }
        )
        print(f"✅ Knowledge base collection created (space=cosine)")

    def build_from_json(self, knowledge_file: Path):
        """Load facts from group_knowledge.json and embed them into the collection."""
        if not knowledge_file.exists():
            print(f"⚠️  Knowledge file not found: {knowledge_file} — skipping KB build")
            return

        with open(knowledge_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        facts = data.get('facts', [])
        if not facts:
            print("⚠️  No facts found in knowledge file")
            return

        print(f"📥 Embedding {len(facts)} knowledge facts...")

        ids = [fact['id'] for fact in facts]
        documents = [fact['text'] for fact in facts]
        metadatas = [
            {'category': fact.get('category', ''), 'subject': fact.get('subject', '')}
            for fact in facts
        ]

        batch_size = 32
        embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_embeddings = self.encoder.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            embeddings.extend(batch_embeddings.tolist())

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"✅ Added {len(facts)} facts to knowledge base collection")

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count() if self.collection else 0
        return {'total_facts': count, 'collection_name': 'kaya_knowledge_base'}


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

    # ------------------------------------------------------------------ #
    #  Part 1: Conversation history collection                             #
    # ------------------------------------------------------------------ #
    print(f"\n🔨 Creating conversation chunks (max {CHUNK_SIZE_TOKENS} tokens each)")
    chunker = ConversationChunker(
        chunk_size_tokens=CHUNK_SIZE_TOKENS,
        overlap_tokens=CHUNK_OVERLAP_TOKENS
    )
    chunks = chunker.create_conversation_chunks(messages)
    print(f"✅ Created {len(chunks)} conversation chunks")

    builder = VectorDatabaseBuilder(DB_DIR, EMBEDDING_MODEL)
    builder.initialize()
    builder.add_chunks(chunks)

    stats = builder.get_stats()
    print("\n" + "=" * 60)
    print("📊 CONVERSATION RAG STATISTICS")
    print("=" * 60)
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Collection: {stats['collection_name']}")
    print(f"Database location: {DB_DIR}")

    total_messages = sum(chunk['message_count'] for chunk in chunks)
    total_tokens = sum(chunk['token_count'] for chunk in chunks)
    print(f"Total messages in chunks: {total_messages}")
    print(f"Total tokens in chunks: {total_tokens:,}")
    print(f"Average messages per chunk: {total_messages / len(chunks):.1f}")
    print(f"Average tokens per chunk: {total_tokens / len(chunks):.0f}")

    # ------------------------------------------------------------------ #
    #  Part 2: Curated knowledge base collection                           #
    # ------------------------------------------------------------------ #
    kb_config = RAG_CONFIG.get('knowledge_base', {})
    if kb_config.get('enabled', True):
        print("\n" + "=" * 60)
        print("🧠 KNOWLEDGE BASE BUILDER")
        print("=" * 60)

        knowledge_file_str = kb_config.get('file', str(DATA_DIR / 'group_knowledge.json'))
        knowledge_file = Path(knowledge_file_str) if os.path.isabs(knowledge_file_str) else _BASE_DIR / knowledge_file_str.lstrip('./')
        kb_collection_name = kb_config.get('collection_name', 'kaya_knowledge_base')

        kb_builder = KnowledgeBaseBuilder(DB_DIR, EMBEDDING_MODEL)
        kb_builder.initialize(kb_collection_name)
        kb_builder.build_from_json(knowledge_file)

        kb_stats = kb_builder.get_stats()
        print(f"Total facts: {kb_stats['total_facts']}")
        print(f"Collection: {kb_stats['collection_name']}")

    print("\n✅ All RAG collections built!")
    print(f"\nNext steps:")
    print(f"  1. Fill in notes/descriptions in data/group_members.json and data/group_knowledge.json")
    print(f"  2. Re-run this script after updating the JSON files to rebuild the knowledge base")
    print(f"  3. Run: python src/chat/chat.py")


if __name__ == "__main__":
    main()