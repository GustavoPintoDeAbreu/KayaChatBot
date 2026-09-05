"""Documents shared in the chat become searchable memory, with page numbers.

Until 2026-09-05 they became nothing at all. The transcribe branch was gated on
"empty text AND not an image", so a PDF was handed to Whisper, which failed, left
the message text empty, and let the empty-text gate drop it. Four papers shared
during an argument that day left no trace anywhere on disk.

The page number is the feature, not a detail: the group's bar for a source is
"da me uma pagina sff" / "Pagina X capítulo x", and a citation that cannot name a
page is worth nothing to them.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import documents

CONFIG = {
    "documents": {"enabled": True, "chunk_size_tokens": 100,
                  "chunk_overlap_tokens": 20},
    "rag": {"embedding_model": "x"},
}


# ── what counts as a document ────────────────────────────────────────────────
@pytest.mark.parametrize("mimetype,filename,expected", [
    ("application/pdf", "paper.pdf", True),
    ("application/pdf; charset=binary", "paper.pdf", True),
    ("text/plain", "notes.txt", True),
    # No declared mimetype: the extension is the only evidence there is, and
    # NOWEB does not always fill the field in.
    ("", "paper.pdf", True),
    ("", "notes.md", True),
    # Not documents — these have their own handlers and must keep them.
    ("image/jpeg", "photo.jpg", False),
    ("audio/ogg; codecs=opus", "voice.oga", False),
    ("video/mp4", "clip.mp4", False),
    ("", "", False),
])
def test_is_document(mimetype, filename, expected):
    assert documents.is_document(mimetype, filename, CONFIG) is expected


# ── chunking keeps the pages ─────────────────────────────────────────────────
def test_every_chunk_knows_its_pages():
    pages = [(1, " ".join(["alpha"] * 80)), (2, " ".join(["beta"] * 80))]
    chunks = documents.chunk_pages(pages, 100, 20)

    assert chunks
    for chunk in chunks:
        assert chunk["page_start"] >= 1
        assert chunk["page_end"] >= chunk["page_start"]
    assert any(c["page_start"] == 1 for c in chunks)
    assert any(c["page_end"] == 2 for c in chunks)


def test_no_chunk_is_a_duplicate_of_the_one_before():
    """A document ending exactly on a boundary used to emit its overlap twice."""
    # 252 words with max=60/overlap=12 lands a flush on the final word.
    pages = [(1, " ".join(f"w{i}" for i in range(252)))]
    chunks = documents.chunk_pages(pages, 100, 20)
    texts = [c["text"] for c in chunks]

    for earlier, later in zip(texts, texts[1:]):
        assert later not in earlier, "tail chunk repeats the previous chunk"


def test_chunking_loses_no_words():
    pages = [(1, " ".join(f"w{i}" for i in range(500)))]
    chunks = documents.chunk_pages(pages, 100, 20)

    covered = set()
    for chunk in chunks:
        covered.update(chunk["text"].split())
    assert len(covered) == 500


def test_a_document_with_no_text_yields_nothing():
    assert documents.chunk_pages([], 100, 20) == []
    assert documents.extract_pages(b"", "application/pdf") == []


def test_a_corrupt_pdf_does_not_raise():
    assert documents.extract_pages(b"not a pdf at all", "application/pdf") == []


def test_plain_text_is_one_page():
    pages = documents.extract_pages("olá mundo, isto é um documento de texto".encode(),
                                    "text/plain")
    assert pages == [(1, "olá mundo, isto é um documento de texto")]


# ── identity and the log line ────────────────────────────────────────────────
def test_the_same_document_gets_the_same_id():
    payload = b"%PDF-1.4 whatever"
    assert documents.doc_id(payload) == documents.doc_id(payload)
    assert documents.doc_id(payload) != documents.doc_id(payload + b" ")


def test_describe_for_log_reads_like_the_image_line():
    line = documents.describe_for_log(
        {"ok": True, "filename": "labour.pdf", "pages": 51,
         "synopsis": "Um artigo sobre sindicatos."})
    assert line == "labour.pdf, 51 páginas — Um artigo sobre sindicatos."


def test_a_document_that_could_not_be_read_is_not_announced():
    """Better silent than claiming to have read a scan it could not open."""
    assert documents.describe_for_log({"ok": False, "filename": "scan.pdf"}) == ""


def test_a_synopsis_failure_still_produces_a_line():
    line = documents.describe_for_log(
        {"ok": True, "filename": "x.pdf", "pages": 2, "synopsis": ""})
    assert line == "x.pdf, 2 páginas"


# ── the archive must not be escapable ────────────────────────────────────────
def test_a_crafted_scope_cannot_escape_the_store(tmp_path):
    config = {**CONFIG, "documents": {**CONFIG["documents"],
                                      "store_dir": str(tmp_path)}}
    path = documents.store_file(b"data", "abc123", "../../etc", "x.pdf", config)

    assert path
    assert Path(path).resolve().is_relative_to(tmp_path.resolve())


# ── the live bugs of 2026-09-05 ──────────────────────────────────────────────
class _FakeClient:
    """A Chroma client where the documents collection does not exist yet."""

    def __init__(self):
        self.created = []

    def get_collection(self, name):
        raise ValueError(f"Collection {name} does not exist.")

    def get_or_create_collection(self, name, metadata=None):
        self.created.append(name)
        return f"collection:{name}"


def test_the_documents_collection_is_created_not_merely_fetched():
    """It is created by the first upload, which is after the process starts.

    Resolving it with `get_collection` left `documents_collection = None` for the
    life of the serving process, so retrieve_documents returned [] on its first
    line and document RAG was dead until the next restart. Live on 2026-09-05:
    162 chunks indexed, `retrieved_chars: 164` on a question about them.
    """
    from src.chat.retriever import open_documents_collection

    client = _FakeClient()
    assert open_documents_collection(client, "kaya_documents") == "collection:kaya_documents"
    assert client.created == ["kaya_documents"]


def test_an_unavailable_documents_collection_does_not_stop_the_boot():
    from src.chat.retriever import open_documents_collection

    class Broken:
        def get_or_create_collection(self, **kwargs):
            raise RuntimeError("disk gone")

    assert open_documents_collection(Broken(), "kaya_documents") is None


class _RecordingCollection:
    """Enough of a Chroma collection to see whether work was repeated."""

    def __init__(self):
        self.rows = {}
        self.upserts = 0

    def get(self, where=None, limit=None, include=None):
        ids = [i for i, meta in self.rows.items()
               if not where or meta.get("doc_id") == where.get("doc_id")]
        if limit:
            ids = ids[:limit]
        return {"ids": ids, "metadatas": [self.rows[i] for i in ids]}

    def upsert(self, ids, documents, embeddings, metadatas):
        self.upserts += 1
        for identifier, meta in zip(ids, metadatas):
            self.rows[identifier] = meta


class _CountingEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        import numpy as np

        return np.zeros((len(texts), 8), dtype="float32")


def _pdf_bytes():
    return ("%PDF-1.4\n" + "x" * 200).encode()


def test_the_same_document_is_not_reindexed(tmp_path):
    """WAHA delivered each message TWICE: six webhook POSTs for three PDFs.

    Upserting by content hash already made that harmless for the store, but the
    expensive half still ran twice — two extractions, two synopsis calls and two
    embedding passes, which produced two different synopses for the same paper.
    """
    config = {"documents": {"store_dir": str(tmp_path), "chunk_size_tokens": 100,
                            "chunk_overlap_tokens": 20},
              "rag": {"db_path": str(tmp_path / "db")}}
    collection, encoder = _RecordingCollection(), _CountingEncoder()
    pages = [(1, "uma frase com conteudo suficiente para sobreviver ao filtro de paginas curtas")]

    import unittest.mock as mock

    with mock.patch.object(documents, "extract_pages", return_value=pages), \
            mock.patch.object(documents, "synopsis", return_value="uma sinopse"):
        first = documents.index_document(
            _pdf_bytes(), "paper.pdf", "shared", "Bernardo", config,
            "application/pdf", collection=collection, encoder=encoder)
        second = documents.index_document(
            _pdf_bytes(), "paper.pdf", "shared", "Bernardo", config,
            "application/pdf", collection=collection, encoder=encoder)

    assert first["ok"] and not first.get("cached")
    assert second["ok"] and second["cached"] is True
    assert encoder.calls == 1, "the second delivery re-embedded the document"
    assert collection.upserts == 1


def test_a_redelivery_still_reports_enough_to_log(tmp_path):
    """The cached report has to produce the same "[Documento: …]" line."""
    config = {"documents": {"store_dir": str(tmp_path), "chunk_size_tokens": 100,
                            "chunk_overlap_tokens": 20},
              "rag": {"db_path": str(tmp_path / "db")}}
    collection, encoder = _RecordingCollection(), _CountingEncoder()
    pages = [(1, "uma frase com conteudo suficiente para sobreviver ao filtro curto")]

    import unittest.mock as mock

    with mock.patch.object(documents, "extract_pages", return_value=pages), \
            mock.patch.object(documents, "synopsis", return_value="uma sinopse"):
        documents.index_document(_pdf_bytes(), "paper.pdf", "shared", "B", config,
                                 "application/pdf", collection=collection, encoder=encoder)
        cached = documents.index_document(_pdf_bytes(), "paper.pdf", "shared", "B", config,
                                          "application/pdf", collection=collection,
                                          encoder=encoder)

    assert documents.describe_for_log(cached) == "paper.pdf, 1 página — uma sinopse"


def test_the_same_bytes_under_a_new_name_are_recognised(tmp_path):
    """Forwarding a paper the group already has must not re-embed it."""
    config = {"documents": {"store_dir": str(tmp_path), "chunk_size_tokens": 100,
                            "chunk_overlap_tokens": 20},
              "rag": {"db_path": str(tmp_path / "db")}}
    collection, encoder = _RecordingCollection(), _CountingEncoder()
    pages = [(1, "uma frase com conteudo suficiente para sobreviver ao filtro curto")]

    import unittest.mock as mock

    with mock.patch.object(documents, "extract_pages", return_value=pages), \
            mock.patch.object(documents, "synopsis", return_value="uma sinopse"):
        documents.index_document(_pdf_bytes(), "original.pdf", "shared", "B", config,
                                 "application/pdf", collection=collection, encoder=encoder)
        again = documents.index_document(_pdf_bytes(), "forwarded.pdf", "shared", "R", config,
                                         "application/pdf", collection=collection, encoder=encoder)

    assert again["cached"] is True
    assert encoder.calls == 1
