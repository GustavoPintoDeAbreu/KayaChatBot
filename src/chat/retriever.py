"""
RAG Retriever for conversation history.
Retrieves relevant conversation chunks based on user queries.
"""

import os
import re
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from src.chat.scope import SHARED, is_readable, parse_iso, scope_filter


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
    # Plain recency words. These were missing, and they are the ones people
    # actually use: "o que é que o grupo fez ontem?" matched NOTHING here, so the
    # answer came back "Não tenho registos claros" after retrieving 6,109 chars
    # of topically-near but months-old chunks. See _recency_window.
    r"\bontem\b", r"\bhoje\b", r"\besta semana\b", r"\bsemana passada\b",
    r"\beste fim de semana\b", r"\bno fim de semana\b", r"\bna sexta\b",
    r"\bontem [àa] noite\b", r"\bnestes [úu]ltimos\b",
    r"\byesterday\b", r"\btoday\b", r"\blast night\b", r"\bthis week\b",
    r"\blast week\b", r"\bthis weekend\b", r"\blast weekend\b",
]
_TEMPORAL_INTENT_RE = re.compile("|".join(_TEMPORAL_INTENT_PATTERNS), re.IGNORECASE)


def _has_temporal_intent(query: str) -> bool:
    """Return True if the query asks about timing/recency of something."""
    if not query:
        return False
    return bool(_TEMPORAL_INTENT_RE.search(query))


# A query that names a period, mapped to how many days back it starts and ends.
# Ordered longest-first so "semana passada" is not shadowed by "semana".
_RECENCY_WINDOWS = [
    (r"\bontem [àa] noite\b|\blast night\b", 1, 1, "ontem à noite"),
    (r"\bantes de ontem\b|\bday before yesterday\b", 2, 2, "anteontem"),
    (r"\bsemana passada\b|\blast week\b", 14, 7, "a semana passada"),
    (r"\beste fim de semana\b|\bno fim de semana\b|\bthis weekend\b|\blast weekend\b",
     4, 0, "o fim de semana"),
    (r"\besta semana\b|\bthis week\b|\bnestes [úu]ltimos dias\b", 7, 0, "esta semana"),
    (r"\bontem\b|\byesterday\b", 1, 1, "ontem"),
    (r"\bhoje\b|\btoday\b", 0, 0, "hoje"),
]
_RECENCY_WINDOWS = [(re.compile(pattern, re.IGNORECASE), start, end, label)
                    for pattern, start, end, label in _RECENCY_WINDOWS]


def _recency_window(query: str, now: Optional[datetime] = None
                    ) -> Optional[Tuple[str, str, str]]:
    """The calendar window a query is asking about, as ``(start, end, label)``.

    Semantic search cannot answer "o que é que o grupo fez ontem?". The question
    shares almost no vocabulary with the answer — an evening of chatter about a
    restaurant — so nearest-neighbour returns whatever is topically closest from
    any point in six years of history. Live example, 2026-08-29: the bot answered
    "Não tenho registos claros sobre o que aconteceu ontem" after retrieving
    6,109 characters, while the dinner it was asked about sat in ChromaDB,
    correctly chunked and embedded, from the previous evening.

    So when the query names a period, that period is fetched by DATE and shown
    alongside the semantic hits rather than instead of them. Returned bounds are
    inclusive ISO dates; timestamps are stored as ISO-8601 strings, which sort
    lexicographically, so a plain string comparison is a correct date comparison.

    Returns None when the query names no period, which is the common case and
    leaves retrieval exactly as it was.
    """
    if not query:
        return None
    now = now or datetime.now()
    for pattern, days_back_start, days_back_end, label in _RECENCY_WINDOWS:
        if pattern.search(query):
            start = (now - timedelta(days=days_back_start)).date().isoformat()
            end = (now - timedelta(days=days_back_end)).date().isoformat()
            # End of day, so a chunk at 23:41 on the last day is inside.
            return start + "T00:00:00", end + "T23:59:59", label
    return None


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


def open_documents_collection(client, name: str):
    """The documents collection, CREATED if it does not exist yet. None on failure.

    get_or_create, not get, and that distinction was a live bug. The collection
    is created by the first document anybody shares, which is almost never before
    the serving process starts — and this resolves ONCE, at startup. `get_collection`
    therefore left ``documents_collection = None`` for the life of the process, so
    ``retrieve_documents`` returned [] on its first line and document RAG was
    silently dead until the next restart.

    Found in production on 2026-09-05, and only because the answers looked right:
    three papers indexed correctly into 162 chunks, while the bot answered from the
    "[Documento: …]" synopsis line carried in the recent window. The telemetry is
    what gave it away — ``retrieved_chars: 164`` on a question about a 35-page paper.

    Never raises: a missing document store is not worth refusing to boot over.
    """
    try:
        return client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"})
    except Exception as exc:  # noqa: BLE001
        print(f"ℹ️  documents collection unavailable: {exc}")
        return None


class ConversationRetriever:
    """Retrieve relevant conversation chunks for RAG."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_config = config.get('rag', {})
        self.client = None
        self.collection = None
        self.knowledge_collection = None
        self.documents_collection = None
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

        # Documents shared in the chat (src/chat/documents.py). Optional and
        # created on first upload, so a store that has never seen one is normal.
        docs_name = (self.config.get('documents', {}) or {}).get(
            'collection_name', 'kaya_documents')
        self.documents_collection = open_documents_collection(self.client, docs_name)
        if self.documents_collection is not None:
            print(f"✅ Documents collection loaded "
                  f"({self.documents_collection.count()} chunks)")

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

        # Word-boundary match so short aliases (e.g. "gil", "rafa", "pedro")
        # don't fire inside unrelated words ("ágil", "garrafa", ...).
        for member in self.group_members:
            if re.search(rf"\b{re.escape(member)}\b", query_lower):
                mentioned.append(member)

        return mentioned

    def named_members(self, text: str) -> List[str]:
        """Canonical member names appearing in ``text``, deduplicated.

        ``extract_query_persons`` returns matched *aliases*, which is what the
        retrieval post-filter wants (it compares them against chunk metadata).
        Counting who a turn is about needs the canonical name instead, so "gilão"
        and "gil" are one person rather than two.
        """
        lowered = (text or "").lower()
        found = []
        for member in self._members_data:
            name = member.get("name")
            if not name:
                continue
            aliases = {name.lower(), *(a.lower() for a in member.get("aliases", []))}
            if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases):
                found.append(name)
        return found

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 query_embedding: Optional[Any] = None,
                 scope: Optional[str] = None,
                 exclude_from: Optional[str] = None) -> List[Dict[str, Any]]:
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

        # Extract mentioned persons for filtering
        query_persons = self.extract_query_persons(query) if FILTER_BY_PERSON else []

        # Generate query embedding (normalized to match the stored vectors),
        # unless the caller already computed it.
        if query_embedding is None:
            query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0]

        # NOTE: ChromaDB doesn't support $contains, so we retrieve more results and filter post-query
        # Retrieve extra results to account for filtering
        n_results_to_fetch = top_k * 3 if query_persons else top_k

        # Restrict to scopes this chat may read: shared group memory plus its own.
        # A DM can recall the group's history; the group can never recall a DM.
        # Done as a `where` clause so filtered chunks don't eat the top-k budget.
        where = scope_filter(scope) if scope else None

        query_kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=min(n_results_to_fetch, self.collection.count()),  # Don't exceed collection size
            include=['documents', 'metadatas', 'distances'],
        )
        if where:
            query_kwargs['where'] = where
        try:
            results = self.collection.query(**query_kwargs)
        except Exception as exc:  # noqa: BLE001
            # An older store may predate the `scope` metadata. Never fail open on
            # a scope error — that would leak a DM into the group. Fall back to
            # shared-only, which is always safe to show anywhere.
            if not where:
                raise
            print(f"⚠️  scope-filtered query failed ({exc}); falling back to shared-only")
            query_kwargs['where'] = {"scope": SHARED}
            results = self.collection.query(**query_kwargs)

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

            # Recency cutoff: the live session store already holds the last few
            # turns verbatim, so retrieving a chunk covering the same window would
            # inject the same text twice and waste the context budget. The session
            # store owns recent history; the vector DB owns everything older.
            if exclude_from:
                chunk_end = parse_iso(metadata.get('timestamp_end'))
                if chunk_end and chunk_end >= exclude_from:
                    continue

            # Defence in depth: the `where` clause above should already have
            # excluded other chats, but a chunk written before scoping existed has
            # no scope field, so verify rather than trust.
            if scope and not is_readable(metadata.get('scope'), scope):
                continue

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

        return self._prepend_recency_window(
            query, retrieved_chunks, top_k, scope, exclude_from)

    def _date_index(self):
        """``[(timestamp_start, id)]`` for every chunk, sorted, cached.

        ChromaDB's ``$gte``/``$lte`` are numeric-only — it rejects an ISO string
        outright ("Expected operand value to be an int or a float") — and the
        timestamps are stored as ISO strings. Rather than migrate the store to
        epoch metadata, the date filter happens here.

        A full metadata scan of 3,611 chunks measures 158 ms, which is fine
        against a ~9 s turn but not fine on every one, so it is cached and
        invalidated on ``count()``. Ingestion only ever appends, in this same
        process, so a changed count is a sufficient signal.
        """
        count = self.collection.count()
        cached = getattr(self, "_date_index_cache", None)
        if cached is not None and cached[0] == count:
            return cached[1]
        rows = self.collection.get(include=['metadatas'])
        index = sorted(
            (str(metadata.get('timestamp_start') or ''), chunk_id)
            for chunk_id, metadata in zip(rows.get('ids') or [],
                                          rows.get('metadatas') or [])
            if metadata.get('timestamp_start')
        )
        self._date_index_cache = (count, index)
        return index

    def _prepend_recency_window(self, query, chunks, top_k, scope, exclude_from):
        """Put the period the question named at the front, fetched by date.

        Nearest-neighbour cannot find "ontem": the question shares no vocabulary
        with an evening of restaurant chatter, so it returns whatever is
        topically closest from any point in six years. Live, 2026-08-29, "o que é
        que o grupo fez ontem? Fomos jantar fora" came back "Não tenho registos
        claros" after retrieving 6,109 characters — while the dinner sat in
        ChromaDB, correctly chunked, from the previous evening. The top hits were
        from November 2025, December 2025 and July 2026.

        The named window is fetched by date and put first; the semantic hits stay
        behind it. The window is what was asked for, the semantic hits are what
        the words matched, and both are worth having.

        Best-effort throughout: any failure returns the semantic results
        unchanged, which is exactly the previous behaviour.
        """
        window = _recency_window(query)
        if not window or not self.collection:
            return chunks
        start, end, _label = window
        try:
            ids = [chunk_id for timestamp, chunk_id in self._date_index()
                   if start <= timestamp <= end]
            if not ids:
                return chunks
            found = self.collection.get(ids=ids[-max(top_k, 1):],
                                        include=['documents', 'metadatas'])
        except Exception as exc:  # noqa: BLE001 — never lose an answer to this
            print(f"⚠️  recency window fetch failed ({exc}); using semantic results only")
            return chunks

        seen = {chunk['text'] for chunk in chunks}
        dated = []
        for doc, metadata in zip(found.get('documents') or [],
                                 found.get('metadatas') or []):
            if not doc or doc in seen:
                continue
            # The same two guards the semantic path applies. Scope is defence in
            # depth — a chunk written before scoping existed has no scope field,
            # and this path has no `where` clause in front of it at all, so the
            # check is load-bearing here rather than merely belt-and-braces.
            if scope and not is_readable(metadata.get('scope'), scope):
                continue
            if exclude_from:
                chunk_end = parse_iso(metadata.get('timestamp_end'))
                if chunk_end and chunk_end >= exclude_from:
                    continue
            dated.append({
                'rank': 0,
                'text': doc,
                'metadata': metadata,
                # Not a cosine score: this chunk was matched by date, not by
                # similarity. Kept high so nothing downstream drops it on the
                # relevance floor, and below 1.0 so it never displaces a real
                # match in anything that sorts by score.
                'similarity_score': 0.99,
                'distance': 0.01,
                'participants': (metadata.get('participants', '').split(',')
                                 if metadata.get('participants') else []),
                'mentioned': (metadata.get('mentioned', '').split(',')
                              if metadata.get('mentioned') else []),
                'message_count': metadata.get('message_count', 0),
                'token_count': metadata.get('token_count', 0),
                'timestamp_start': metadata.get('timestamp_start'),
                'timestamp_end': metadata.get('timestamp_end'),
            })
            seen.add(doc)

        if not dated:
            return chunks
        dated.sort(key=lambda chunk: str(chunk.get('timestamp_start') or ''))
        merged = dated + chunks
        for index, chunk in enumerate(merged, start=1):
            chunk['rank'] = index
        # The window may legitimately be larger than top_k — a whole evening of
        # chatter is several chunks, and truncating it to top_k would answer
        # "what did we do yesterday" with the first ten minutes of it.
        return merged[:max(top_k, len(dated))]

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

    def retrieve_documents(self, query: str, top_k: Optional[int] = None,
                           scope: Optional[str] = None,
                           query_embedding: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Retrieve chunks from documents the group has shared.

        Scope-filtered exactly like conversations, and for the same reason: a PDF
        sent in a DM must never surface in the group. The `where` clause is
        backed by the same in-Python `is_readable` check, because failing open
        here leaks a private document rather than merely a stale one.

        Every chunk carries the pages it came from, which is what lets an answer
        say "página 51" instead of "o documento diz".
        """
        if not self.documents_collection or not self.encoder:
            return []
        try:
            available = self.documents_collection.count()
        except Exception:  # noqa: BLE001
            return []
        if not available:
            return []

        dcfg = self.config.get('documents', {}) or {}
        if top_k is None:
            top_k = int(dcfg.get('top_k', 4))
        if query_embedding is None:
            query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0]

        query_kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, available),
            include=['documents', 'metadatas', 'distances'],
        )
        where = scope_filter(scope) if scope else None
        if where:
            query_kwargs['where'] = where
        try:
            results = self.documents_collection.query(**query_kwargs)
        except Exception as exc:  # noqa: BLE001
            if not where:
                return []
            print(f"⚠️  scope-filtered document query failed ({exc}); shared-only")
            query_kwargs['where'] = {"scope": SHARED}
            try:
                results = self.documents_collection.query(**query_kwargs)
            except Exception:  # noqa: BLE001
                return []

        min_similarity = float(dcfg.get('min_similarity', 0.25))
        chunks: List[Dict[str, Any]] = []
        for doc, metadata, distance in zip(results['documents'][0],
                                           results['metadatas'][0],
                                           results['distances'][0]):
            similarity = 1 - distance
            if similarity < min_similarity:
                continue
            # Defence in depth, load-bearing: the where clause above is the only
            # other thing standing between a DM's document and the group.
            if scope and not is_readable(metadata.get('scope'), scope):
                continue
            chunks.append({
                'text': doc,
                'filename': metadata.get('filename', 'documento'),
                'sender': metadata.get('sender', ''),
                'page_start': metadata.get('page_start', 0),
                'page_end': metadata.get('page_end', 0),
                'page_count': metadata.get('page_count', 0),
                'synopsis': metadata.get('synopsis', ''),
                'doc_id': metadata.get('doc_id', ''),
                'similarity_score': similarity,
            })
        return chunks

    @staticmethod
    def page_label(chunk: Dict[str, Any]) -> str:
        """"p. 51" or "pp. 51-52" — how a chunk is cited."""
        start, end = chunk.get('page_start') or 0, chunk.get('page_end') or 0
        if not start:
            return ""
        return f"p. {start}" if not end or end == start else f"pp. {start}-{end}"

    def format_documents_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Render document hits with the page numbers that make them citable."""
        if not chunks:
            return ""
        parts = ["=== Documentos partilhados no grupo ==="]
        for chunk in chunks:
            pages = self.page_label(chunk)
            sender = f", enviado por {chunk['sender']}" if chunk.get('sender') else ""
            header = f"--- {chunk.get('filename', 'documento')}"
            header += f", {pages}" if pages else ""
            header += f"{sender} ---"
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n".join(parts)

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

    def retrieve_all(
        self,
        query: str,
        knowledge_approach: str = "both",
        top_k: Optional[int] = None,
        scope: Optional[str] = None,
        exclude_from: Optional[str] = None,
        include_documents: bool = True,
    ) -> str:
        """
        Retrieve context from all active sources and return a combined formatted context block.

        ``include_documents=False`` is for a debate, which retrieves documents
        itself so it can hand them to the model as a NUMBERED source list that
        can be cited and verified. Leaving them here as well would send the same
        pages twice.

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

        # Retrieve conversation history (sorted by similarity descending). ``top_k``
        # is narrowed by the caller for the lighter `mixed` intent, where the reply
        # is conversational and does not need the full evidence pile.
        conv_chunks = self.retrieve(
            query,
            top_k=top_k,
            query_embedding=query_embedding,
            scope=scope,
            exclude_from=exclude_from,
        )

        # Retrieve from knowledge base if approach calls for it
        kb_chunks = []
        if knowledge_approach in ("both", "chromadb_only"):
            kb_config = self.rag_config.get('knowledge_base', {})
            if kb_config.get('enabled', False):
                kb_chunks = self.retrieve_knowledge(query, query_embedding=query_embedding)

        # Documents the group has shared. Retrieved for every mode that retrieves
        # at all, not only for debates: "que docs mandou o Bana?" and "o que dizia
        # aquele paper?" are ordinary questions, and an index nothing reads is an
        # index not worth building.
        doc_chunks = []
        if include_documents and (self.config.get('documents', {}) or {}).get('enabled', True):
            doc_chunks = self.retrieve_documents(
                query, scope=scope, query_embedding=query_embedding)

        # Inject recent summaries for members mentioned in the query
        recent_summaries_text = ""
        if inject_recent_summaries:
            query_persons = self.extract_query_persons(query)
            recent_summaries_text = self._format_recent_summaries(query_persons)

        # Enforce token budget — truncate lowest-priority context first:
        #   1. Conversation chunks (lowest similarity first, i.e. from the end)
        #   2. Document chunks (lowest similarity first)
        #   3. Knowledge facts (all at once)
        #   4. Recent summaries
        #
        # Documents outrank conversation chunks here because a document chunk is
        # the only kind of context that can be cited by page, and a question that
        # pulled documents at all is usually a question about them. They are still
        # dropped before the curated knowledge facts, which are cheap and dense.
        def _total() -> int:
            return (
                self._count_tokens(self.format_context(conv_chunks, show_dates=show_dates))
                + self._count_tokens(self.format_documents_context(doc_chunks))
                + self._count_tokens(self.format_knowledge_context(kb_chunks, show_dates=show_dates))
                + self._count_tokens(recent_summaries_text)
            )

        while conv_chunks and _total() > max_tokens:
            # retrieve() returns chunks sorted by similarity descending, so the last
            # item is the lowest-similarity chunk — remove it first.
            conv_chunks.pop()

        while doc_chunks and _total() > max_tokens:
            doc_chunks.pop()

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
        # Last, so the pages sit closest to the question being answered.
        if doc_chunks:
            context_parts.append(self.format_documents_context(doc_chunks))

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

def peek_retriever() -> Optional["ConversationRetriever"]:
    """The retriever if one is already loaded, else None — never builds one.

    Lets the ingester borrow the embedding model this process has already paid
    for instead of loading a second ~2.2GB copy onto the serving card, without
    forcing a load in the CLI paths where no retriever exists.
    """
    return _retriever_instance


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