"""
RAG Retriever for conversation history.
Retrieves relevant conversation chunks based on user queries.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import yaml

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# RAG Configuration
RAG_CONFIG = config['rag']
VECTOR_DB = RAG_CONFIG['vector_db']
EMBEDDING_MODEL = RAG_CONFIG['embedding_model']
TOP_K = RAG_CONFIG['top_k']
FILTER_BY_PERSON = RAG_CONFIG['filter_by_person']

# Detect if running in Docker
DB_DIR = Path("/app/data/rag_db") if os.path.exists('/app') else Path(__file__).parent.parent.parent / "data" / "rag_db"


class ConversationRetriever:
    """Retrieve relevant conversation chunks for RAG."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_config = config.get('rag', {})
        self.client = None
        self.collection = None
        self.knowledge_collection = None
        self.encoder = None

        # Load group members from JSON file (single source of truth)
        members_file = config.get('data', {}).get('group_members_file')
        if members_file and Path(members_file).exists():
            with open(members_file, 'r', encoding='utf-8') as f:
                members_data = json.load(f)
            self.group_members = set()
            for m in members_data.get('members', []):
                for alias in m.get('aliases', []):
                    self.group_members.add(alias.lower())
        else:
            # Fallback if JSON not available
            self.group_members = {
                'peter', 'gil', 'gustavo', 'david', 'manuel', 'carnall', 'frederico',
                'mateus', 'rafa', 'bernardo', 'chamusca', 'gilao', 'pedro'
            }

    def initialize(self):
        """Initialize the retriever with vector database and embedding model."""
        if not DB_DIR.exists():
            raise FileNotFoundError(f"RAG database not found at {DB_DIR}. Run build_vector_db.py first!")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(DB_DIR))

        # Get conversation history collection
        collection_name = "kaya_conversations"
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as e:
            raise RuntimeError(f"Could not load collection '{collection_name}': {e}")

        # Try loading the curated knowledge base collection (optional — built separately)
        kb_config = self.rag_config.get('knowledge_base', {})
        kb_collection_name = kb_config.get('collection_name', 'kaya_knowledge_base')
        try:
            self.knowledge_collection = self.client.get_collection(name=kb_collection_name)
            print(f"✅ Knowledge base collection loaded ({self.knowledge_collection.count()} facts)")
        except Exception:
            self.knowledge_collection = None
            print("ℹ️  No knowledge base collection found — run build_vector_db.py to create it")

        # Load embedding model (GTE requires trust_remote_code)
        self.encoder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

        print(f"✅ RAG Retriever initialized with {self.collection.count()} conversation chunks")

    def extract_query_persons(self, query: str) -> List[str]:
        """Extract person names mentioned in the query."""
        query_lower = query.lower()
        mentioned = []

        for member in self.group_members:
            if member in query_lower:
                mentioned.append(member)

        return mentioned

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant conversation chunks for a query.

        Args:
            query: The user's query
            top_k: Number of chunks to retrieve (overrides config)

        Returns:
            List of retrieved chunks with metadata
        """
        if not self.collection or not self.encoder:
            raise RuntimeError("Retriever not initialized. Call initialize() first.")

        if top_k is None:
            top_k = self.rag_config.get('top_k', TOP_K)

        # Extract mentioned persons for filtering
        query_persons = self.extract_query_persons(query) if FILTER_BY_PERSON else []

        # Generate query embedding
        query_embedding = self.encoder.encode([query])[0]

        # NOTE: ChromaDB doesn't support $contains, so we retrieve more results and filter post-query
        # Retrieve extra results to account for filtering
        n_results_to_fetch = top_k * 3 if query_persons else top_k

        # Query the vector database without where clause
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results_to_fetch, self.collection.count()),  # Don't exceed collection size
            include=['documents', 'metadatas', 'distances']
        )

        # Format results
        retrieved_chunks = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Post-query filtering by person if needed
            if query_persons:
                participants_list = metadata.get('participants', '').split(',') if metadata.get('participants') else []
                mentioned_list = metadata.get('mentioned', '').split(',') if metadata.get('mentioned') else []
                
                # Check if any query person is in participants or mentioned
                person_found = any(
                    person in participants_list or person in mentioned_list
                    for person in query_persons
                )
                
                if not person_found:
                    continue  # Skip this chunk
            
            retrieved_chunks.append({
                'rank': len(retrieved_chunks) + 1,
                'text': doc,
                'metadata': metadata,
                'similarity_score': 1 - distance,  # Convert distance to similarity
                'distance': distance,
                'participants': metadata.get('participants', '').split(',') if metadata.get('participants') else [],
                'mentioned': metadata.get('mentioned', '').split(',') if metadata.get('mentioned') else [],
                'message_count': metadata.get('message_count', 0),
                'token_count': metadata.get('token_count', 0),
                'timestamp_start': metadata.get('timestamp_start'),
                'timestamp_end': metadata.get('timestamp_end')
            })
            
            # Stop when we have enough results after filtering
            if len(retrieved_chunks) >= top_k:
                break

        return retrieved_chunks

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved conversation chunks into context string for the model."""
        if not retrieved_chunks:
            return ""

        context_parts = ["=== Conversas relevantes do grupo ==="]

        for i, chunk in enumerate(retrieved_chunks, 1):
            # Format timestamp
            timestamp_info = ""
            if chunk.get('timestamp_start'):
                try:
                    from datetime import datetime
                    start_dt = datetime.fromisoformat(chunk['timestamp_start'])
                    timestamp_info = f" [{start_dt.strftime('%Y-%m-%d')}]"
                except:
                    pass

            context_parts.append(f"\n--- Conversa {i}{timestamp_info} ---")
            context_parts.append(chunk['text'])

        context_parts.append("\n=== Fim das conversas ===")

        return "\n".join(context_parts)

    def retrieve_knowledge(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant facts from the curated knowledge base collection."""
        if not self.knowledge_collection or not self.encoder:
            return []

        kb_config = self.rag_config.get('knowledge_base', {})
        if top_k is None:
            top_k = kb_config.get('top_k', 3)

        query_embedding = self.encoder.encode([query])[0]

        results = self.knowledge_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.knowledge_collection.count()),
            include=['documents', 'metadatas', 'distances']
        )

        knowledge_chunks = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            knowledge_chunks.append({
                'text': doc,
                'subject': metadata.get('subject', ''),
                'category': metadata.get('category', ''),
                'similarity_score': 1 - distance,
            })

        return knowledge_chunks

    def format_knowledge_context(self, knowledge_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved knowledge base facts into context string."""
        if not knowledge_chunks:
            return ""

        context_parts = ["=== Conhecimento sobre o grupo ==="]
        for chunk in knowledge_chunks:
            subject = chunk.get('subject', '')
            header = f"\n--- {subject} ---" if subject else "\n---"
            context_parts.append(header)
            context_parts.append(chunk['text'])
        context_parts.append("\n=== Fim do conhecimento ===")

        return "\n".join(context_parts)

    def retrieve_all(self, query: str, knowledge_approach: str = "both") -> str:
        """
        Retrieve context from all active sources and return a combined formatted context block.

        knowledge_approach:
          "both"         — JSON members (injected via system prompt externally) + KB retrieval + conversation RAG
          "json_only"    — conversation RAG only (JSON injection handled in chat.py)
          "chromadb_only"— KB retrieval + conversation RAG (no JSON injection)
          "none"         — conversation RAG only (baseline)
        """
        context_parts = []

        # Always retrieve conversation history
        conv_chunks = self.retrieve(query)
        if conv_chunks:
            context_parts.append(self.format_context(conv_chunks))

        # Retrieve from knowledge base if approach calls for it
        if knowledge_approach in ("both", "chromadb_only"):
            kb_config = self.rag_config.get('knowledge_base', {})
            if kb_config.get('enabled', False):
                kb_chunks = self.retrieve_knowledge(query)
                if kb_chunks:
                    context_parts.insert(0, self.format_knowledge_context(kb_chunks))

        return "\n\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        if not self.collection:
            return {"error": "Retriever not initialized"}

        stats = {
            'total_conversation_chunks': self.collection.count(),
            'embedding_model': EMBEDDING_MODEL,
            'top_k_default': TOP_K,
            'filter_by_person': FILTER_BY_PERSON,
        }
        if self.knowledge_collection:
            stats['total_knowledge_facts'] = self.knowledge_collection.count()
        return stats


# Global retriever instance
_retriever_instance = None

def get_retriever(config: Dict[str, Any]) -> ConversationRetriever:
    """Get or create retriever instance (singleton pattern)."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = ConversationRetriever(config)
        _retriever_instance.initialize()
    return _retriever_instance