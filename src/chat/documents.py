"""Reading the documents people share.

Same shape as `vision.py` and `stt.py`, and for the same reason: a PDF carries no
text of its own, so the message means nothing until its content becomes text.
Until 2026-09-05 it meant less than nothing — the transcribe branch in
`whatsapp_adapter` was gated on "not an image", so a PDF was handed to Whisper,
failed, and left the message with empty text, which the empty-text gate then
dropped. Four papers shared in an argument that day left no trace on disk at all.

Once a document is text it flows through everything else unchanged: the message
log records "[Documento: nome — sinopse]" exactly the way a photo records
"[Imagem: …]", the ingester folds that into ChromaDB, and "aquele doc que o Bana
mandou" is findable a month later. That reuse is the whole design; nothing
downstream needs to know what a PDF is.

What is different from a photo is that the FULL TEXT is worth keeping. A
description of an image is a summary and the image itself is not searchable, but
a paper is 300 pages of prose that answers questions nobody has asked yet. So the
pages are chunked into their own collection, `kaya_documents`, with the page
number on every chunk — because the question this exists to answer is Pedro's:
"da me uma pagina sff".

A separate collection, not `kaya_conversations`: `build_vector_db.initialize()`
DROPS and recreates that one on every full rebuild, and says so in a warning about
the `image`/`live` chunks it destroys. Documents stored there would not survive
the next pipeline run.

The synopsis is written by the LOCAL model over the same llama server the vision
describer uses. Group data does not leave the box.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# pypdf logs a paragraph-long warning per embedded font it cannot fully parse
# ("fontTools is required to fully parse the encoding of a CFF Type1 font..."),
# several times per page. Extraction succeeds anyway, so this is pure noise, and
# a 93-page manual produced 287 KB of it. Errors still come through.
logging.getLogger("pypdf").setLevel(logging.ERROR)

SYNOPSIS_PROMPT = (
    "Isto são excertos de um documento partilhado num grupo de amigos. Escreve uma "
    "sinopse em português europeu, em 2 a 3 frases: de que trata o documento, que "
    "tipo de documento é (artigo académico, relatório, manual, contrato, notícia) e "
    "qual é a conclusão ou o argumento principal, se der para perceber. Não inventes "
    "dados nem números que não estejam no texto. Responde só com a sinopse."
)

# Mimetypes we can turn into text. Anything else is left alone: a document the
# bot cannot read must not be announced as one it can.
DEFAULT_MIMETYPES = (
    "application/pdf",
    "text/plain",
    "text/markdown",
)

_EXT_MIMETYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}

# A page whose extracted text is shorter than this carries nothing worth
# embedding — a scanned image, a divider, a page of page numbers. Kept out of the
# chunks rather than stored as an empty hit that outranks a real one.
_MIN_PAGE_CHARS = 40


def _config(config: Dict[str, Any]) -> Dict[str, Any]:
    return (config.get("documents", {}) or {})


def is_available(config: Dict[str, Any]) -> bool:
    return bool(_config(config).get("enabled", False))


def _server_url(config: Dict[str, Any]) -> str:
    gguf = ((config.get("inference", {}) or {}).get("gguf", {}) or {})
    return os.environ.get("KAYA_LLAMA_URL") or gguf.get("server_url", "http://llama:8080")


def collection_name(config: Dict[str, Any]) -> str:
    return str(_config(config).get("collection_name", "kaya_documents"))


def store_dir(config: Dict[str, Any]) -> str:
    configured = str(_config(config).get("store_dir", "./data/documents"))
    if os.path.isabs(configured):
        return configured
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, configured.lstrip("./"))


def guess_mimetype(mimetype: str, filename: str) -> str:
    """The mimetype to treat this attachment as.

    WAHA does not always supply one, and a bare extension is better evidence than
    an empty string. The declared type wins when there is one.
    """
    declared = (mimetype or "").split(";")[0].strip().lower()
    if declared:
        return declared
    _, ext = os.path.splitext((filename or "").lower())
    return _EXT_MIMETYPES.get(ext, "")


def is_document(mimetype: str, filename: str, config: Dict[str, Any]) -> bool:
    """Whether this attachment is something we can extract text from."""
    allowed = tuple(_config(config).get("mimetypes") or DEFAULT_MIMETYPES)
    return guess_mimetype(mimetype, filename) in allowed


def doc_id(payload: bytes) -> str:
    """Content-addressed id, so the same paper shared twice is stored once."""
    return hashlib.sha256(payload).hexdigest()[:24]


def download(url: str, api_key: str = "", waha_base_url: str = "",
             max_bytes: int = 0) -> Optional[bytes]:
    """Fetch the file from WAHA. None on any failure — never raises.

    ``rewrite_media_url`` is not optional: WAHA reports its media at
    ``http://localhost:3000/...``, which is its own container and not ours.
    """
    if not url:
        return None

    from src.chat.stt import rewrite_media_url

    url = rewrite_media_url(url, waha_base_url)
    try:
        import httpx

        headers = {"X-Api-Key": api_key} if api_key else {}
        with httpx.Client(timeout=120.0, headers=headers, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            payload = response.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not fetch document %s: %s", url, exc)
        return None

    if max_bytes and len(payload) > max_bytes:
        logger.warning("document is %.1f MB, over the %.1f MB cap — skipped",
                       len(payload) / 1e6, max_bytes / 1e6)
        return None
    return payload


def extract_pages(payload: bytes, mimetype: str,
                  max_pages: int = 0) -> List[Tuple[int, str]]:
    """``[(page_number, text)]``, 1-indexed. Empty on any failure.

    The page number is the point of this module: it is what turns "o documento diz
    X" into "página 51 diz X", which is the only form of citation this group
    accepts.

    A plain-text file has no pages, so it is reported as one page — the caller
    renders that as the whole document rather than as "página 1".
    """
    if not payload:
        return []

    if mimetype in ("text/plain", "text/markdown"):
        try:
            return [(1, payload.decode("utf-8", errors="replace").strip())]
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not decode text document: %s", exc)
            return []

    try:
        import io as _io

        from pypdf import PdfReader

        reader = PdfReader(_io.BytesIO(payload))
        if reader.is_encrypted:
            # An empty-password decrypt covers the common "protected but not
            # secret" case; a real password is not something we can ask for.
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001
                logger.warning("document is encrypted and could not be opened")
                return []
        pages: List[Tuple[int, str]] = []
        for number, page in enumerate(reader.pages, start=1):
            if max_pages and number > max_pages:
                logger.warning("document has more than %d pages — truncated", max_pages)
                break
            try:
                text = (page.extract_text() or "").strip()
            except Exception as exc:  # noqa: BLE001 — one bad page is not a bad document
                logger.warning("page %d could not be read: %s", number, exc)
                continue
            if len(text) >= _MIN_PAGE_CHARS:
                pages.append((number, _tidy(text)))
        return pages
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not extract document text: %s", exc)
        return []


def _tidy(text: str) -> str:
    """Collapse the whitespace PDF extraction leaves behind."""
    text = text.replace("­", "")           # soft hyphens
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: Sequence[Tuple[int, str]], chunk_size_tokens: int = 400,
                overlap_tokens: int = 60) -> List[Dict[str, Any]]:
    """Group pages into chunks that each know which pages they came from.

    Chunking is by page rather than by a sliding window over the whole document,
    because a chunk that cannot name its pages cannot be cited, and a citation is
    the entire point. A page longer than the budget is split, and every piece
    keeps that page's number.

    Sizes are in whitespace words scaled the way the retriever estimates tokens
    (`words / 0.60`), so the budget here means the same thing as the budget in
    `retrieve_all`.
    """
    max_words = max(1, int(chunk_size_tokens * 0.60))
    overlap_words = max(0, min(int(overlap_tokens * 0.60), max_words - 1))

    chunks: List[Dict[str, Any]] = []
    buffer: List[str] = []
    first_page = last_page = 0
    # Words added since the last flush. Without this the carried-over overlap
    # alone can trigger a final flush, emitting a tail chunk that is a pure
    # subset of the one before it — which happens whenever the document ends
    # exactly on a chunk boundary.
    fresh = 0

    def flush() -> None:
        nonlocal buffer, first_page, fresh
        if not buffer:
            return
        chunks.append({
            "text": " ".join(buffer).strip(),
            "page_start": first_page,
            "page_end": last_page,
        })
        buffer = buffer[-overlap_words:] if overlap_words else []
        first_page = last_page
        fresh = 0

    for number, text in pages:
        for word in text.split():
            if not buffer:
                first_page = number
            buffer.append(word)
            fresh += 1
            last_page = number
            if len(buffer) >= max_words:
                flush()
    if fresh and len(" ".join(buffer).strip()) >= _MIN_PAGE_CHARS:
        flush()
    return [chunk for chunk in chunks if chunk["text"]]


def synopsis(pages: Sequence[Tuple[int, str]], config: Dict[str, Any]) -> str:
    """Two or three sentences on what the document is. "" on any failure.

    Written by the LOCAL model: a document shared in the group is group data and
    does not leave the box.

    The excerpt deliberately takes the front AND the tail. An academic paper's
    abstract is at the front and its conclusion is at the back, and the group's own
    reading habit is the same — "Tens chapter de conclusao n precisas de ler todo".
    """
    if not pages:
        return ""
    dcfg = _config(config)
    head = " ".join(text for _, text in pages[:3])
    tail = " ".join(text for _, text in pages[-2:]) if len(pages) > 3 else ""
    excerpt = (head[:6000] + ("\n\n[...]\n\n" + tail[:3000] if tail else "")).strip()
    try:
        import requests

        response = requests.post(
            f"{_server_url(config).rstrip('/')}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content":
                              f"{dcfg.get('synopsis_prompt', SYNOPSIS_PROMPT)}\n\n{excerpt}"}],
                "max_tokens": int(dcfg.get("synopsis_max_new_tokens", 160)),
                "temperature": 0.2,
                # Without this Gemma-4 emits its thinking channel and the answer
                # lands in reasoning_content with content left empty.
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=float(dcfg.get("timeout", 180)),
        )
        response.raise_for_status()
        return (response.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as exc:  # noqa: BLE001 — a failed synopsis is not a failed document
        logger.warning("document synopsis failed: %s", exc)
        return ""


def store_file(payload: bytes, identifier: str, scope: str, filename: str,
               config: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> str:
    """Keep the bytes so the document can be re-indexed without being re-sent.

    Chunking will change; asking the group to re-share a 300-page paper because we
    improved the splitter is not an option. Returns the path, or "" on failure —
    the index is still usable when the archive write fails.
    """
    try:
        _, ext = os.path.splitext(filename or "")
        directory = os.path.join(store_dir(config), _safe(scope))
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{identifier}{ext.lower() or '.bin'}")
        if not os.path.exists(path):
            with open(path, "wb") as handle:
                handle.write(payload)
        with open(os.path.join(directory, f"{identifier}.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"doc_id": identifier, "filename": filename, "scope": scope,
                       "bytes": len(payload), "stored_at": time.time(),
                       **(meta or {})}, handle, ensure_ascii=False, indent=2)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not archive document %s: %s", filename, exc)
        return ""


def _safe(value: str) -> str:
    """A scope turned into one path segment, matching MessageLog's filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value or "shared")


# ── indexing ────────────────────────────────────────────────────────────────


def _db_path(config: Dict[str, Any]) -> str:
    path = str((config.get("rag", {}) or {}).get("db_path", "./data/rag_db"))
    if os.path.isabs(path):
        return path
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, path.lstrip("./"))


def get_collection(config: Dict[str, Any], client=None):
    """The `kaya_documents` collection, created on first use.

    Deliberately NOT `kaya_conversations`: `build_vector_db.initialize()` drops
    and recreates that collection on every full pipeline run, which would delete
    every document ever shared.
    """
    import chromadb

    client = client or chromadb.PersistentClient(path=_db_path(config))
    return client.get_or_create_collection(
        name=collection_name(config), metadata={"hnsw:space": "cosine"}
    )


def _encoder(config: Dict[str, Any]):
    """The embedder the serving process already holds, or a private one.

    Same borrow as `src/data/ingest.py`: a second SentenceTransformer is ~2.2 GB
    on the card the model answers from.
    """
    try:
        from src.chat.retriever import peek_retriever

        shared = peek_retriever()
        if shared is not None and getattr(shared, "encoder", None) is not None:
            return shared.encoder
    except Exception as exc:  # noqa: BLE001 — fall back to our own
        logger.debug("could not borrow the retriever's encoder: %s", exc)

    from sentence_transformers import SentenceTransformer

    logger.info("loading a private embedder for documents")
    return SentenceTransformer(
        (config.get("rag", {}) or {}).get("embedding_model", "BAAI/bge-m3"),
        trust_remote_code=True,
    )


def index_document(payload: bytes, filename: str, scope: str, sender: str,
                   config: Dict[str, Any], mimetype: str = "",
                   timestamp: Optional[float] = None,
                   collection=None, encoder=None) -> Dict[str, Any]:
    """Extract, chunk, embed and store one document.

    Returns a report dict; ``{"ok": False, ...}`` when there is nothing to index.
    Idempotent: chunk ids are derived from the content hash, so the same paper
    shared twice upserts rather than duplicating.
    """
    dcfg = _config(config)
    mimetype = guess_mimetype(mimetype, filename) or "application/pdf"
    identifier = doc_id(payload)
    report: Dict[str, Any] = {"ok": False, "doc_id": identifier, "filename": filename,
                              "pages": 0, "chunks": 0, "synopsis": ""}

    pages = extract_pages(payload, mimetype,
                          max_pages=int(dcfg.get("max_pages", 0) or 0))
    if not pages:
        # A scanned PDF extracts to nothing. Saying so beats indexing an empty
        # document that then answers questions with silence.
        report["error"] = "no extractable text"
        return report
    report["pages"] = len(pages)

    chunks = chunk_pages(pages,
                         int(dcfg.get("chunk_size_tokens", 400)),
                         int(dcfg.get("chunk_overlap_tokens", 60)))
    if not chunks:
        report["error"] = "no chunks"
        return report

    text = synopsis(pages, config)
    report["synopsis"] = text

    stamp = float(timestamp or time.time())
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp))
    store_file(payload, identifier, scope, filename, config,
               meta={"sender": sender, "pages": len(pages), "synopsis": text,
                     "mimetype": mimetype})

    collection = collection if collection is not None else get_collection(config)
    encoder = encoder if encoder is not None else _encoder(config)

    ids, texts, metadatas = [], [], []
    for index, chunk in enumerate(chunks):
        # The filename and synopsis ride along in the embedded text. A chunk from
        # page 180 of a labour-market paper shares no vocabulary with "aquele doc
        # que o Bana mandou", so without this the document is only findable by
        # people who already know what is inside it.
        header = f"[{filename}] {text}".strip()
        ids.append(f"doc_{identifier}_{index:04d}")
        texts.append(f"{header}\n\n{chunk['text']}" if header else chunk["text"])
        metadatas.append({
            "doc_id": identifier,
            "filename": filename or "documento",
            "sender": sender or "",
            "scope": scope or "shared",
            "timestamp": iso,
            "page_start": int(chunk["page_start"]),
            "page_end": int(chunk["page_end"]),
            "page_count": len(pages),
            "synopsis": text,
            "source": "document",
        })

    vectors = encoder.encode(texts, normalize_embeddings=True,
                             show_progress_bar=False).tolist()
    collection.upsert(ids=ids, documents=texts, embeddings=vectors,
                      metadatas=metadatas)

    report["ok"] = True
    report["chunks"] = len(chunks)
    return report


def describe_for_log(report: Dict[str, Any]) -> str:
    """The one line that goes into the message text, mirroring "[Imagem: …]".

    Empty when nothing was indexed — the caller then leaves the message alone
    rather than writing a claim about a document it cannot read.
    """
    if not report.get("ok"):
        return ""
    name = report.get("filename") or "documento"
    pages = report.get("pages") or 0
    text = (report.get("synopsis") or "").strip()
    head = f"{name}, {pages} página{'s' if pages != 1 else ''}"
    return f"{head} — {text}" if text else head
