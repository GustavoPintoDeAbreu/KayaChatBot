"""A citation the bot did not earn must not survive the reply.

Written against the argument of 2026-09-05, where Pedro took Bernardo's
AI-generated reference list apart claim by claim ("diz-me nas tuas referências
onde é que o custo de comida face à inflação subiu 100%", "Porque me mandaste ai
slop a pensar que era factos"). The bot is held to that bar by code, not only by
a prompt line, because the prompt will eventually lose and the group only has to
catch it once.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import sources

DOCS = [
    {"filename": "labour.pdf", "sender": "Bernardo", "page_start": 51,
     "page_end": 52, "text": "Real wages grew ~55% since 1970."},
    {"filename": "oecd.pdf", "sender": "Bernardo", "page_start": 7,
     "page_end": 7, "text": "Food expenditure share fell."},
]
WEB = [{"answer": "OECD real wage growth 1994-2024: +30.8%.",
        "sources": ["https://www.oecd.org/a", "https://wsj.com/b"]}]


def _allowed(docs=DOCS, web=WEB):
    _, markers, pages = sources.build_source_block(docs, web)
    return markers, pages


# ── the source block ─────────────────────────────────────────────────────────
def test_every_source_is_numbered_and_paged():
    block, markers, pages = sources.build_source_block(DOCS, WEB)

    assert markers == {"D1", "D2", "W1"}
    assert pages == {51, 52, 7}
    assert "[D1] labour.pdf, pp. 51-52" in block
    assert "[D2] oecd.pdf, p. 7" in block
    assert "[W1]" in block


def test_no_sources_means_no_block():
    block, markers, pages = sources.build_source_block([], [])
    assert block == "" and markers == set() and pages == set()


# ── invented markers ─────────────────────────────────────────────────────────
def test_a_real_citation_survives():
    markers, pages = _allowed()
    text, removed = sources.verify_citations("Subiram 55% [D1].", markers, pages)
    assert text == "Subiram 55% [D1]." and removed == []


def test_an_invented_marker_is_removed_and_the_claim_kept():
    markers, pages = _allowed()
    text, removed = sources.verify_citations(
        "A comida subiu 100% [D9].", markers, pages)

    assert "[D9]" not in text
    assert "A comida subiu 100%" in text, "the claim stays; it is now unattributed"
    assert removed == ["[D9]"]


def test_an_invented_web_marker_is_removed():
    markers, pages = _allowed()
    text, _ = sources.verify_citations("Foi assim [W4].", markers, pages)
    assert "[W4]" not in text


# ── invented pages ───────────────────────────────────────────────────────────
def test_a_real_page_survives():
    markers, pages = _allowed()
    text, removed = sources.verify_citations(
        "Está na página 51, como disseste.", markers, pages)
    assert "página 51" in text and removed == []


def test_an_invented_page_is_redacted_grammatically():
    markers, pages = _allowed()
    for given, expected in (
        ("A página 288 diz isso.", "Essa parte do documento diz isso."),
        ("Isso vem da página 288 do estudo.",
         "Isso vem dessa parte do documento do estudo."),
        ("Está na página 288.", "Está nessa parte do documento."),
    ):
        text, removed = sources.verify_citations(given, markers, pages)
        assert text == expected, given
        assert removed


def test_ordinary_numbers_are_not_touched():
    markers, pages = _allowed()
    text, removed = sources.verify_citations(
        "Custa 400 euros e ele tem 51 anos.", markers, pages)
    assert text == "Custa 400 euros e ele tem 51 anos." and removed == []


def test_pages_are_not_policed_when_no_document_was_retrieved():
    """"a página 51 do relatório que mandaste" is ordinary speech."""
    text, removed = sources.verify_citations(
        "Vê a página 51 do relatório que mandaste.", set(), set())
    assert "página 51" in text and removed == []


# ── the citation line ────────────────────────────────────────────────────────
def test_only_cited_sources_are_listed():
    """Carnall: "leste 4 documentos que perfazem mais de 300 páginas?"."""
    line = sources.citation_line(DOCS, WEB[0]["sources"], {"D1"})
    assert "labour.pdf pp. 51-52" in line
    assert "oecd.pdf" not in line
    assert "🌐" not in line, "no web marker was cited"


def test_web_domains_appear_when_a_web_source_is_cited():
    line = sources.citation_line(DOCS, WEB[0]["sources"], {"W1"})
    assert "oecd.org" in line and "wsj.com" in line


def test_citing_nothing_produces_no_line():
    assert sources.citation_line(DOCS, WEB[0]["sources"], set()) == ""


def test_www_is_stripped_without_eating_the_host():
    """`lstrip("www.")` would turn wsj.com into sj.com."""
    assert sources._domain("https://wsj.com/x") == "wsj.com"
    assert sources._domain("https://www.oecd.org/x") == "oecd.org"
    assert sources._domain("https://w3schools.com/x") == "w3schools.com"


def test_cited_markers_reads_what_the_reply_used():
    assert sources.cited_markers("a [D1] b [W2] c [D1]") == {"D1", "W2"}
    assert sources.cited_markers("") == set()


# ── a redaction must not read like a typo ────────────────────────────────────
def test_removing_a_marker_takes_its_dangling_conjunction():
    """Seen live: "os documentos [D1] e [W1] confirmam" -> "[D1] e confirmam",
    where that [W1] had been invented (no web lookup ran on that turn).

    The redaction was correct and the sentence looked broken, which in this group
    is its own kind of failure.
    """
    markers, pages = _allowed()
    for given, expected in (
        # W5 is not among the sources granted by `_allowed()` (D1, D2, W1).
        ("Os documentos [D1] e [W5] confirmam o aumento.",
         "Os documentos [D1] confirmam o aumento."),
        ("Como dizem [W4] e [D1], subiu.", "Como dizem [D1], subiu."),
        ("Vê [D1], [W7] e [W8] para isso.", "Vê [D1] para isso."),
        ("Isso está errado [W9].", "Isso está errado."),
    ):
        text, removed = sources.verify_citations(given, markers, pages)
        assert text == expected, given
        assert removed


def test_a_kept_citation_keeps_its_conjunction():
    markers, pages = _allowed()
    text, removed = sources.verify_citations(
        "Os documentos [D1] e [D2] confirmam.", markers, pages)
    assert text == "Os documentos [D1] e [D2] confirmam."
    assert removed == []
