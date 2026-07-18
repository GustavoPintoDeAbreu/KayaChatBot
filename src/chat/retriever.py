"""
RAG Retriever for conversation history.
Retrieves relevant conversation chunks based on user queries.
"""

import os
import re
import json
import threading
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer

# Query keywords that signal the user is asking about *when* something happened
# or how recent it is. Date metadata is only surfaced into the injected context
# when one of these matches, so normal questions stay date-free.
_TEMPORAL_INTENT_PATTERNS = [
    # Portuguese
    r"\bquando\b", r"\bh[áa] quanto tempo\b", r"\brecente", r"\bnão? h[áa]\b",
    r"\b[úu]ltima vez\b", r"\bque dia\b", r"\bque ano\b", r"\bque m[êe]s\b",
    r"\bdesde quando\b", r"\bh[áa] quantos\b", r"\bantig", r"\bnovidade",
    r"\batualizad", r"\bda altura\b", r"\bnaquela altura\b",
    # English
    r"\bwhen\b", r"\bhow long ago\b", r"\brecent", r"\blast time\b",
    r"\bhow recent", r"\bwhat year\b", r"\bwhat day\b", r"\bsince when\b",
    r"\bhow old\b", r"\blatest\b", r"\bup to date\b", r"\bnowadays\b",
]
_TEMPORAL_INTENT_RE = re.compile("|".join(_TEMPORAL_INTENT_PATTERNS), re.IGNORECASE)


def _has_temporal_intent(query: str) -> bool:
    """Return True if the query asks about timing/recency of something."""
    if not query:
        return False
    return bool(_TEMPORAL_INTENT_RE.search(query))


# Function words dropped from the lexical (exact-term) signal so hybrid fusion
# keys on content tokens — proper nouns, nicknames, brands — not "o que é que".
_LEXICAL_STOPWORDS = {
    "que", "qual", "quais", "quem", "onde", "quando", "como", "porque", "para",
    "com", "sem", "dos", "das", "num", "numa", "the", "what", "who", "where",
    "when", "how", "why", "does", "did", "was", "were", "and", "for", "with",
    "about", "tem", "está", "esta", "sao", "são", "uma", "uns", "umas", "seu",
    "sua", "dele", "dela", "isto", "isso", "aqui", "meu", "minha", "nao", "não",
}


def _lexical_tokens(text: str) -> List[str]:
    """Content tokens (accent-stripped, ≥3 chars, non-stopword) for lexical match."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in _LEXICAL_STOPWORDS]


def _relative_age(iso_date: Optional[str], today: Optional[datetime] = None) -> str:
    """Render an ISO date as a coarse relative age in European Portuguese.

    Returns e.g. "hoje", "há ~3 dias", "há ~2 meses", "há ~1 ano". Returns an
    empty string when the date is missing or unparseable so callers can skip it.
    """
    if not iso_date:
        return ""
    try:
        then = datetime.fromisoformat(str(iso_date))
    except (ValueError, TypeError):
        return ""
    now = today or datetime.now()
    # Compare by calendar date so a same-day timestamp reads "hoje" regardless
    # of the time-of-day component.
    days = (now.date() - then.date()).days
    if days < 0:
        return ""
    if days == 0:
        return "hoje"
    if days < 14:
        return f"há ~{days} dia{'s' if days != 1 else ''}"
    if days < 60:
        weeks = round(days / 7)
        return f"há ~{weeks} semana{'s' if weeks != 1 else ''}"
    if days < 365:
        months = round(days / 30)
        return f"há ~{months} {'mês' if months == 1 else 'meses'}"
    years = round(days / 365)
    return f"há ~{years} ano{'s' if years != 1 else ''}"

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
            # Map every alias to its canonical member name so person-filtering can
            # compare *identities* rather than raw strings. Raw chat sender strings
            # ("Gil João", "fredericop167") never equal an alias ("gil", "frederico"),
            # so the old exact-match filter silently dropped those members' chunks.
            self._alias_to_member: Dict[str, str] = {}
            for m in self._members_data:
                name = m.get('name', '')
                for alias in m.get('aliases', []):
                    self.group_members.add(alias.lower())
                    self._alias_to_member[alias.lower()] = name
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
            self._alias_to_member = {a: a for a in self.group_members}

        # Longest aliases first so multi-word aliases ("benny pereira") win over
        # their substrings ("benny") when normalising a raw sender string.
        self._aliases_by_len = sorted(self._alias_to_member, key=len, reverse=True)

        # Explicit sender-string → member overrides for display names that collide
        # with another member's alias ("Gil João" contains "joão", also a Murgeiro
        # alias) or have no matchable token ("fredericop167"). Sourced from the same
        # config.data.sender_aliases map the extraction SenderResolver uses, so both
        # paths agree on identity. Checked before the alias scan.
        self._sender_overrides = {
            str(k).lower(): v
            for k, v in (config.get('data', {}).get('sender_aliases', {}) or {}).items()
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

        if self.rag_config.get('hybrid', {}).get('enabled', False):
            self._build_lexical_index()

    def _build_lexical_index(self) -> None:
        """Build an in-memory idf-weighted inverted index over chunk texts.

        The dense channel misses rare exact terms (a project name in 1 message,
        a beach in a dozen) because their chunks fall outside the dense top-k.
        This lexical channel retrieves by content token so those chunks can enter
        the candidate pool and be fused in. Cheap: ~2.4k short chunks in memory.
        """
        from collections import defaultdict
        import math
        data = self.collection.get(include=['documents', 'metadatas'])
        ids = data.get('ids', []) or []
        docs = data.get('documents', []) or []
        metas = data.get('metadatas', []) or []
        self._lex_ids = ids
        self._lex_docs = docs
        self._lex_metas = metas
        self._lex_postings: Dict[str, set] = defaultdict(set)
        for idx, text in enumerate(docs):
            for tok in set(_lexical_tokens(text)):
                self._lex_postings[tok].add(idx)
        n = max(len(docs), 1)
        self._lex_idf = {
            tok: math.log(1.0 + n / len(postings))
            for tok, postings in self._lex_postings.items()
        }
        print(f"✅ Lexical index built ({len(self._lex_postings)} terms over {len(docs)} chunks)")

    def extract_query_persons(self, query: str) -> List[str]:
        """Extract person names mentioned in the query."""
        query_lower = query.lower()
        mentioned = []

        # Word-boundary match so short aliases (e.g. "gil", "rafa", "pedro")
        # don't fire inside unrelated words ("ágil", "garrafa", ...).
        for member in self.group_members:
            if re.search(rf"\b{re.escape(member)}\b", query_lower):
                mentioned.append(member)

        return mentioned

    def _canonical_members(self, token: str) -> set:
        """Map a raw participant/mentioned token to canonical member name(s).

        Chat sender strings ("Gil João", "fredericop167", "João Gil") never
        equal an alias, so identity comparison must normalise them: a member is
        present when one of its aliases appears as a whole word in the token, or
        (for glued handles like "fredericop167") as a substring. Longest aliases
        are tried first so "benny pereira" wins over "benny".
        """
        if not token:
            return set()
        tl = token.lower().strip()
        # Explicit override wins (resolves cross-member sender-name collisions).
        if tl in self._sender_overrides:
            return {self._sender_overrides[tl]}
        # Whole-word alias match first (precise); fall back to substring only for
        # glued handles ("fredericop167") when the whole-word pass found nothing.
        matched = {
            self._alias_to_member[a]
            for a in self._aliases_by_len
            if re.search(rf"\b{re.escape(a)}\b", tl)
        }
        if not matched:
            matched = {
                self._alias_to_member[a] for a in self._aliases_by_len if a in tl
            }
        return matched

    def _person_in_chunk(self, query_members: set, metadata: Dict[str, Any]) -> bool:
        """True if any queried member authored or is mentioned in the chunk.

        Compares canonical member identities on both sides so raw sender-string
        variants (the dominant form for several members) are matched correctly.
        """
        chunk_members = set()
        for field in ('participants', 'mentioned'):
            raw = metadata.get(field, '')
            if not raw:
                continue
            for tok in raw.split(','):
                chunk_members |= self._canonical_members(tok)
        return bool(query_members & chunk_members)

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 query_embedding: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant conversation chunks for a query.

        Args:
            query: The user's query
            top_k: Number of chunks to retrieve (overrides config)
            query_embedding: Precomputed normalized query embedding. When None it
                is computed here; retrieve_all passes one in so the query is only
                embedded once per turn.

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

        # Extract mentioned persons for filtering, normalised to canonical
        # member identities so they compare against sender-string variants.
        query_persons = self.extract_query_persons(query) if FILTER_BY_PERSON else []
        query_members = {
            self._alias_to_member.get(p, p) for p in query_persons
        }

        # Generate query embedding (normalized to match the stored vectors),
        # unless the caller already computed it.
        if query_embedding is None:
            query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0]

        # Over-fetch a candidate pool, then filter (min_similarity + person) and
        # re-rank. Hybrid fusion re-ranks a larger pool; the person filter also
        # needs headroom because ChromaDB can't filter by participant server-side.
        hybrid_cfg = self.rag_config.get('hybrid', {})
        hybrid_enabled = hybrid_cfg.get('enabled', False)
        pool = hybrid_cfg.get('candidate_pool', 40) if hybrid_enabled else 0
        n_results_to_fetch = max(pool, top_k * 3 if query_persons else top_k)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results_to_fetch, self.collection.count()),  # Don't exceed collection size
            include=['documents', 'metadatas', 'distances']
        )

        # Collect every eligible dense candidate (cosine order preserved), keyed
        # by chunk id so the lexical channel can union without duplicates.
        docs0 = results['documents'][0]
        metas0 = results['metadatas'][0]
        dists0 = results['distances'][0]
        dense_ids = (results.get('ids') or [[]])[0]
        if len(dense_ids) != len(docs0):  # e.g. mocked query() without ids
            dense_ids = [f"_pos_{i}" for i in range(len(docs0))]
        candidates: List[Dict[str, Any]] = []
        seen_ids = set()
        for cid, doc, metadata, distance in zip(
            dense_ids, docs0, metas0, dists0
        ):
            similarity = 1 - distance  # cosine distance → cosine similarity
            if similarity < min_similarity:
                continue  # Below relevance floor — skip
            if query_members and not self._person_in_chunk(query_members, metadata):
                continue
            seen_ids.add(cid)
            candidates.append(self._make_chunk(doc, metadata, similarity))

        # Lexical channel: pull chunks that contain rare query terms but fell
        # outside the dense pool, so exact proper-noun / rare-term matches can be
        # fused in rather than being unreachable (dense-only never fetched them).
        if hybrid_enabled and getattr(self, '_lex_postings', None):
            for extra in self._lexical_candidates(
                query, query_embedding, min_similarity, query_members, exclude=seen_ids
            ):
                candidates.append(extra)

        # Rank: dense-only keeps cosine order; hybrid fuses cosine with an
        # idf-weighted lexical ranking via Reciprocal Rank Fusion.
        if hybrid_enabled and candidates:
            candidates = self._rrf_rerank(
                query, candidates,
                hybrid_cfg.get('rrf_k', 60),
                hybrid_cfg.get('lexical_weight', 0.5),
            )

        retrieved_chunks = candidates[:top_k]
        for i, chunk in enumerate(retrieved_chunks, 1):
            chunk['rank'] = i
        return retrieved_chunks

    def _make_chunk(self, doc: str, metadata: Dict[str, Any],
                    similarity: float) -> Dict[str, Any]:
        """Build the standard retrieved-chunk dict from a document + metadata."""
        return {
            'text': doc,
            'metadata': metadata,
            'similarity_score': similarity,
            'distance': 1 - similarity,
            'participants': metadata.get('participants', '').split(',') if metadata.get('participants') else [],
            'mentioned': metadata.get('mentioned', '').split(',') if metadata.get('mentioned') else [],
            'message_count': metadata.get('message_count', 0),
            'token_count': metadata.get('token_count', 0),
            'timestamp_start': metadata.get('timestamp_start'),
            'timestamp_end': metadata.get('timestamp_end'),
        }

    def _lexical_query_scores(self, query: str) -> Dict[int, float]:
        """idf-weighted lexical score per indexed chunk for the query terms."""
        scores: Dict[int, float] = {}
        for term in set(_lexical_tokens(query)):
            postings = self._lex_postings.get(term)
            if not postings:
                continue
            weight = self._lex_idf.get(term, 0.0)
            for idx in postings:
                scores[idx] = scores.get(idx, 0.0) + weight
        return scores

    def _lexical_candidates(self, query: str, query_embedding: Any,
                            min_similarity: float, query_members: set,
                            exclude: set, limit: int = 20) -> List[Dict[str, Any]]:
        """Return eligible chunks that match query terms but weren't dense-fetched.

        Rarer terms (higher idf) dominate the ranking so a project name in one
        message beats a chunk that merely shares common words. Cosine is computed
        from the stored embedding so these candidates still respect the relevance
        floor and carry a real similarity_score.
        """
        scores = self._lexical_query_scores(query)
        if not scores:
            return []
        import numpy as np
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        want_idx = [idx for idx, _ in ranked[: limit * 3]
                    if self._lex_ids[idx] not in exclude]
        if not want_idx:
            return []
        want_ids = [self._lex_ids[idx] for idx in want_idx]
        fetched = self.collection.get(ids=want_ids, include=['documents', 'metadatas', 'embeddings'])
        emb_by_id = dict(zip(fetched['ids'], fetched['embeddings']))
        meta_by_id = dict(zip(fetched['ids'], fetched['metadatas']))
        doc_by_id = dict(zip(fetched['ids'], fetched['documents']))
        q = np.asarray(query_embedding, dtype=float)
        out = []
        for cid in want_ids:
            emb = emb_by_id.get(cid)
            if emb is None:
                continue
            similarity = float(np.dot(q, np.asarray(emb, dtype=float)))
            if similarity < min_similarity:
                continue
            metadata = meta_by_id.get(cid, {})
            if query_members and not self._person_in_chunk(query_members, metadata):
                continue
            out.append(self._make_chunk(doc_by_id.get(cid, ''), metadata, similarity))
            if len(out) >= limit:
                break
        return out

    def _rrf_rerank(self, query: str, candidates: List[Dict[str, Any]],
                    rrf_k: int, lexical_weight: float = 0.5) -> List[Dict[str, Any]]:
        """Fuse a cosine ranking with an idf-weighted lexical ranking via RRF.

        Reciprocal Rank Fusion is score-scale-free: each list contributes
        ``1 / (rrf_k + rank)``. The dense list ranks all candidates by cosine; the
        lexical list ranks them by idf-weighted query-term overlap, so exact
        proper-noun / nickname matches surface even when their dense score is
        mid-pack (or when they entered only through the lexical channel).
        ``lexical_weight`` (<1) keeps the lexical channel additive — it pulls rare
        exact-term chunks up without letting common-word overlap displace strong
        dense matches on person-scoped queries.
        """
        # Dense list: rank every candidate by cosine (candidates may include
        # lexical-channel entries, so re-sort rather than trust arrival order).
        by_cosine = sorted(candidates, key=lambda c: c['similarity_score'], reverse=True)
        rrf = {id(c): 1.0 / (rrf_k + rank) for rank, c in enumerate(by_cosine, 1)}

        query_terms = set(_lexical_tokens(query))
        if query_terms:
            def lex_weight(chunk):
                terms = set(_lexical_tokens(chunk['text']))
                return sum(self._lex_idf.get(t, 0.0) for t in (query_terms & terms))
            lexical = sorted(candidates, key=lex_weight, reverse=True)
            for rank, chunk in enumerate(lexical, 1):
                if lex_weight(chunk) > 0:
                    rrf[id(chunk)] += lexical_weight / (rrf_k + rank)

        return sorted(candidates, key=lambda c: rrf[id(c)], reverse=True)

    def format_context(self, retrieved_chunks: List[Dict[str, Any]],
                       show_dates: bool = False) -> str:
        """Format retrieved conversation chunks into context string for the model.

        Dates are only attached when ``show_dates`` is True (the query asks about
        timing), so normal answers aren't cluttered with timestamps every turn.
        """
        if not retrieved_chunks:
            return ""

        context_parts = ["=== Conversas relevantes do grupo ==="]

        for i, chunk in enumerate(retrieved_chunks, 1):
            # Only surface the chunk's date when the user asked about timing.
            timestamp_info = ""
            if show_dates and chunk.get('timestamp_start'):
                try:
                    start_dt = datetime.fromisoformat(chunk['timestamp_start'])
                    rel = _relative_age(chunk['timestamp_start'])
                    date_str = start_dt.strftime('%Y-%m-%d')
                    timestamp_info = f" [{date_str}{f', {rel}' if rel else ''}]"
                except (ValueError, TypeError):
                    pass

            context_parts.append(f"\n--- Conversa {i}{timestamp_info} ---")
            context_parts.append(chunk['text'])

        context_parts.append("\n=== Fim das conversas ===")

        return "\n".join(context_parts)

    def retrieve_knowledge(self, query: str, top_k: Optional[int] = None,
                           query_embedding: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant facts from the curated knowledge base collection."""
        if not self.knowledge_collection or not self.encoder:
            return []

        kb_config = self.rag_config.get('knowledge_base', {})
        if top_k is None:
            top_k = kb_config.get('top_k', 3)

        if query_embedding is None:
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
                # Date fields (mixed rule): explicit text hint wins over the
                # source message range. Absent on facts built before dating.
                'event_date_hint': metadata.get('event_date_hint', ''),
                'last_updated': metadata.get('last_updated', ''),
                'source_date_start': metadata.get('source_date_start', ''),
                'source_date_end': metadata.get('source_date_end', ''),
            })

        return knowledge_chunks

    def _fact_date_suffix(self, chunk: Dict[str, Any]) -> str:
        """Build a recency suffix for a knowledge fact (mixed rule).

        Prefers an explicit temporal expression stated in the source text
        (``event_date_hint``); otherwise falls back to the source message dates.
        Returns "" when the fact carries no date info.
        """
        hint = (chunk.get('event_date_hint') or '').strip()
        if hint:
            return f" (referência temporal: {hint})"
        anchor = chunk.get('last_updated') or chunk.get('source_date_end')
        rel = _relative_age(anchor)
        if rel:
            return f" (atualizado {rel})"
        return ""

    def format_knowledge_context(self, knowledge_chunks: List[Dict[str, Any]],
                                 show_dates: bool = False) -> str:
        """Format retrieved knowledge base facts into context string.

        Date/recency suffixes are only attached when ``show_dates`` is True.
        """
        if not knowledge_chunks:
            return ""

        context_parts = ["=== Conhecimento sobre o grupo ==="]
        for chunk in knowledge_chunks:
            subject = chunk.get('subject', '')
            date_suffix = self._fact_date_suffix(chunk) if show_dates else ""
            header = f"\n--- {subject}{date_suffix} ---" if subject else f"\n---{date_suffix}"
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

        # Only attach dates/recency to the context when the query asks about timing,
        # so normal answers stay date-free (per the date-aware-facts design).
        show_dates = _has_temporal_intent(query)

        # Embed the query once and reuse it for both conversation and KB search.
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0] if self.encoder else None

        # Always retrieve conversation history (sorted by similarity descending)
        conv_chunks = self.retrieve(query, query_embedding=query_embedding)

        # Retrieve from knowledge base if approach calls for it
        kb_chunks = []
        if knowledge_approach in ("both", "chromadb_only"):
            kb_config = self.rag_config.get('knowledge_base', {})
            if kb_config.get('enabled', False):
                kb_chunks = self.retrieve_knowledge(query, query_embedding=query_embedding)

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
                self._count_tokens(self.format_context(conv_chunks, show_dates=show_dates))
                + self._count_tokens(self.format_knowledge_context(kb_chunks, show_dates=show_dates))
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
            context_parts.append(self.format_knowledge_context(kb_chunks, show_dates=show_dates))
        if recent_summaries_text:
            context_parts.append(recent_summaries_text)
        if conv_chunks:
            context_parts.append(self.format_context(conv_chunks, show_dates=show_dates))

        return "\n\n".join(context_parts)

    def best_similarity(self, query: str, query_embedding=None) -> float:
        """Top cosine similarity for ``query`` across both collections.

        A cheap relevance probe used to decide whether a question is *about the
        group* at all: a low best score means RAG has nothing close, i.e. it's a
        general-knowledge / out-of-group question (the web-search trigger). Returns
        the max top-1 similarity over the conversation + knowledge collections, or
        0.0 if nothing is available. No person filter / no min_similarity floor —
        we want the raw best match.
        """
        if not self.encoder:
            return 0.0
        if query_embedding is None:
            query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0]
        best = 0.0
        for collection in (self.collection, self.knowledge_collection):
            if not collection:
                continue
            try:
                if collection.count() == 0:
                    continue
                res = collection.query(
                    query_embeddings=[query_embedding], n_results=1, include=['distances']
                )
                dists = (res.get('distances') or [[]])[0]
                if dists:
                    best = max(best, 1 - dists[0])  # cosine distance → similarity
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  best_similarity probe failed: {exc}")
        return best

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


# Global retriever instance (thread-safe singleton — the web UI may call this
# from multiple request threads).
_retriever_instance = None
_retriever_lock = threading.Lock()

def get_retriever(config: Dict[str, Any]) -> ConversationRetriever:
    """Get or create retriever instance (double-checked locking singleton)."""
    global _retriever_instance
    if _retriever_instance is None:
        with _retriever_lock:
            if _retriever_instance is None:
                instance = ConversationRetriever(config)
                instance.initialize()
                _retriever_instance = instance
    return _retriever_instance