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

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
from src.config_loader import load_config
config = load_config(str(CONFIG_PATH))

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
            self._members_data = members_data.get('members', [])
            self.group_members = set()
            for m in self._members_data:
                for alias in m.get('aliases', []):
                    self.group_members.add(alias.lower())
        else:
            # Fallback if JSON not available — log a warning
            import logging as _logging
            _logging.warning(
                "group_members.json not found at '%s' — using hardcoded member fallback. "
                "Member-filtered RAG may be incomplete.", members_file or "(not configured)"
            )
            self._members_data = []
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

        conv_count = self.collection.count()
        if conv_count == 0:
            import logging as _logging
            _logging.warning(
                "Collection 'kaya_conversations' is empty — RAG will return no results. "
                "Run build_vector_db.py first."
            )
        print(f"✅ RAG Retriever initialized with {conv_count} conversation chunks")

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

        # Relevance floor: with normalized embeddings + cosine space,
        # similarity_score is a true cosine similarity in [-1, 1]. Chunks below
        # this score are dropped so always-on RAG doesn't inject the
        # least-irrelevant chunks for off-topic queries. 0.0 disables filtering.
        min_similarity = self.rag_config.get('min_similarity', 0.0)

        # Extract mentioned persons for filtering
        query_persons = self.extract_query_persons(query) if FILTER_BY_PERSON else []

        # Generate query embedding (normalized to match the stored vectors)
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0]

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
            similarity = 1 - distance  # cosine distance → cosine similarity
            if similarity < min_similarity:
                continue  # Below relevance floor — skip

            # Post-query filtering by person if needed
            if query_persons:
                participants_list = [p.lower() for p in metadata.get('participants', '').split(',')] if metadata.get('participants') else []
                mentioned_list = [m.lower() for m in metadata.get('mentioned', '').split(',')] if metadata.get('mentioned') else []

                # Check if any query person is in participants or mentioned (case-insensitive)
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
                'similarity_score': similarity,
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

        query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0]

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
            # Truncate to first 3 sentences to stay within the model's token budget
            text = chunk['text']
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            truncated = '. '.join(sentences[:3]) + ('.' if sentences else '')
            context_parts.append(truncated)
        context_parts.append("\n=== Fim do conhecimento ===")

        return "\n".join(context_parts)

    def _count_tokens(self, text: str) -> int:
        """Approximate token count for Portuguese/English mixed text.

        Portuguese subword tokenizers produce ~20-25% more tokens per word than
        English (more inflection, diacritics, clitics). Using 0.60 words/token
        instead of the English 0.75 approximation to stay within the RAG budget.
        """
        if not text:
            return 0
        return int(len(text.split()) / 0.60)

    def _format_recent_summaries(self, query_persons: List[str]) -> str:
        """Format recent summaries for members mentioned in the query."""
        if not query_persons or not self._members_data:
            return ""

        summaries = []
        for member in self._members_data:
            aliases = [a.lower() for a in member.get('aliases', [])]
            if any(p in aliases for p in query_persons):
                summary = member.get('recent_summary', '').strip()
                if summary:
                    summaries.append(f"[Resumo recente — {member['name']}] {summary}")

        if not summaries:
            return ""

        lines = ["=== Resumos recentes dos membros ==="] + summaries + ["=== Fim dos resumos ==="]
        return "\n".join(lines)

    def retrieve_all(self, query: str, knowledge_approach: str = "both") -> str:
        """
        Retrieve context from all active sources and return a combined formatted context block.

        knowledge_approach:
          "both"         — JSON members (injected via system prompt externally) + KB retrieval + conversation RAG
          "json_only"    — conversation RAG only (JSON injection handled in chat.py)
          "chromadb_only"— KB retrieval + conversation RAG (no JSON injection)
          "none"         — conversation RAG only (baseline)
        """
        max_tokens = self.rag_config.get('max_context_tokens', 3000)
        inject_recent_summaries = self.rag_config.get('inject_recent_summaries', True)

        # Always retrieve conversation history (sorted by similarity descending)
        conv_chunks = self.retrieve(query)

        # Retrieve from knowledge base if approach calls for it
        kb_chunks = []
        if knowledge_approach in ("both", "chromadb_only"):
            kb_config = self.rag_config.get('knowledge_base', {})
            if kb_config.get('enabled', False):
                kb_chunks = self.retrieve_knowledge(query)

        # Inject recent summaries for members mentioned in the query
        recent_summaries_text = ""
        if inject_recent_summaries:
            query_persons = self.extract_query_persons(query)
            recent_summaries_text = self._format_recent_summaries(query_persons)

        # Enforce token budget — truncate lowest-priority context first:
        #   1. Conversation chunks (lowest similarity first, i.e. from the end)
        #   2. Knowledge facts (all at once)
        #   3. Recent summaries
        def _total() -> int:
            return (
                self._count_tokens(self.format_context(conv_chunks))
                + self._count_tokens(self.format_knowledge_context(kb_chunks))
                + self._count_tokens(recent_summaries_text)
            )

        while conv_chunks and _total() > max_tokens:
            # retrieve() returns chunks sorted by similarity descending, so the last
            # item is the lowest-similarity chunk — remove it first.
            conv_chunks.pop()

        if kb_chunks and _total() > max_tokens:
            kb_chunks = []

        if recent_summaries_text and _total() > max_tokens:
            recent_summaries_text = ""

        # Assemble final context: knowledge → recent summaries → conversations
        context_parts = []
        if kb_chunks:
            context_parts.append(self.format_knowledge_context(kb_chunks))
        if recent_summaries_text:
            context_parts.append(recent_summaries_text)
        if conv_chunks:
            context_parts.append(self.format_context(conv_chunks))

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