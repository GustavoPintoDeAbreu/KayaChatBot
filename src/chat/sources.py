"""Sources the bot is allowed to cite, and the check that it only cited those.

This exists because of what the group was arguing about on 2026-09-05. Bernardo
posted AI-generated statistics with a list of references; Pedro went through them
and found the references did not say what the numbers claimed:

    "Será que antes de mandares a tua verborreia jurada por AI, leste as tuas
     referências que o AI ensinou? Viste a data? Abriste os links?"
    "Porque me mandaste ai slop a pensar que era factos"
    "da me uma pagina sff" / "Pagina X capítulo x"

A bot that invents one page number gets treated exactly the way they are treating
him, permanently. So the rule is enforced twice: the prompt tells the model it may
only cite what it was handed, and this module deletes any citation that was not.

The check is deliberately dumb and total. It does not try to judge whether a claim
is supported — it only asks whether the marker the model wrote corresponds to a
source that was actually retrieved this turn. That is a question with a definite
answer, which is the only kind worth enforcing in code.

An unsupported claim is not rewritten or deleted: dropping the marker leaves the
sentence standing as the bot's own assertion, which is honest. Silently deleting
the sentence would make a reply that no longer says what the model meant.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set, Tuple

# The markers the model is told to use. Anything matching this shape that is not
# in the allowed set was invented.
_MARKER_RE = re.compile(r"\[(?P<kind>[DW])(?P<index>\d{1,2})\]")

# Stands in for a removed marker while the sentence around it is tidied.
# A control character, so it cannot collide with anything the model wrote.
_HOLE = "\x00"

# "página 51", "pag. 51", "pp. 51-52", "page 51", "p.51". Matched so a page the
# model names in prose can be checked against the pages actually retrieved.
# The optional leading article is swallowed on purpose: replacing only "página
# 400" inside "a página 400 confirma" leaves "a essa parte do documento".
_PAGE_RE = re.compile(
    r"(?P<lead>\b(?:n[ao]s?|d[ao]s?|[ao]s?|à|em)\s+)?"
    r"\b(?:p{1,2}\.?|p[áa]gs?\.?|p[áa]ginas?|pages?)\s*(?P<first>\d{1,4})(?:\s*[-–]\s*(?P<last>\d{1,4}))?",
    re.IGNORECASE,
)


def build_source_block(doc_chunks: Sequence[Dict[str, Any]],
                       web_results: Sequence[Dict[str, Any]]) -> Tuple[str, Set[str], Set[int]]:
    """Render the numbered sources the model may cite.

    Returns ``(block, allowed_markers, allowed_pages)``. ``allowed_pages`` is
    every page number covered by a retrieved document chunk, used to catch a
    plausible-looking "página 51" that came from nowhere.
    """
    lines: List[str] = []
    allowed: Set[str] = set()
    pages: Set[int] = set()

    for index, chunk in enumerate(doc_chunks, start=1):
        marker = f"D{index}"
        allowed.add(marker)
        start = int(chunk.get("page_start") or 0)
        end = int(chunk.get("page_end") or start)
        if start:
            pages.update(range(start, max(start, end) + 1))
        where = _page_label(start, end)
        name = chunk.get("filename") or "documento"
        sender = chunk.get("sender") or ""
        header = f"[{marker}] {name}"
        if where:
            header += f", {where}"
        if sender:
            header += f" (enviado por {sender})"
        lines.append(f"{header}\n{(chunk.get('text') or '').strip()}")

    for index, result in enumerate(web_results, start=1):
        marker = f"W{index}"
        allowed.add(marker)
        domains = ", ".join(result.get("sources") or []) or "pesquisa web"
        lines.append(f"[{marker}] {domains}\n{(result.get('answer') or '').strip()}")

    if not lines:
        return "", allowed, pages

    block = (
        "Fontes disponíveis. Só podes citar destas, usando a marca entre "
        "parênteses rectos (por exemplo [D1] ou [W1]). Se uma afirmação tua não "
        "estiver em nenhuma delas, diz que não tens fonte para isso em vez de "
        "inventares uma:\n\n" + "\n\n".join(lines)
    )
    return block, allowed, pages


def _page_label(start: int, end: int) -> str:
    if not start:
        return ""
    return f"p. {start}" if not end or end == start else f"pp. {start}-{end}"


def verify_citations(reply: str, allowed: Set[str],
                     allowed_pages: Set[int]) -> Tuple[str, List[str]]:
    """Strip citations the model was not given. ``(cleaned, removed)``.

    Two kinds of invention are caught:

    * a marker like ``[D3]`` when only two documents were retrieved;
    * a page number in prose that no retrieved chunk covers, but ONLY when
      documents were actually retrieved. With no documents in hand the number is
      just a number in a sentence ("página 51 do relatório que mandaste"), and
      deleting it would mangle ordinary text.
    """
    if not reply:
        return reply, []

    removed: List[str] = []

    def _marker(match: re.Match) -> str:
        token = f"{match.group('kind')}{int(match.group('index'))}"
        if token in allowed:
            return match.group(0)
        removed.append(match.group(0))
        return ""

    # Removed markers become a sentinel first, so a conjunction left dangling by
    # the removal can be cleaned up knowing WHERE the hole is. Deleting "[W1]"
    # from "os documentos [D1] e [W1] confirmam" otherwise leaves "[D1] e
    # confirmam", which is how a correct redaction ends up looking like a typo.
    cleaned = _MARKER_RE.sub(lambda m: _HOLE if _marker(m) == "" else m.group(0), reply)
    cleaned = re.sub(rf"\s*(?:,|\be\b|\band\b)\s*{_HOLE}", "", cleaned)
    cleaned = re.sub(rf"{_HOLE}\s*(?:,|\be\b|\band\b)\s*", "", cleaned)
    cleaned = cleaned.replace(_HOLE, "")

    if allowed_pages:
        cleaned_so_far = cleaned

        def _page(match: re.Match) -> str:
            numbers = [int(n) for n in (match.group("first"), match.group("last")) if n]
            if any(number in allowed_pages for number in numbers):
                return match.group(0)
            removed.append(match.group(0))
            # The claim survives without the fabricated locator. The replacement
            # has to agree with whatever article or preposition was swallowed,
            # or "vem da página 999" becomes "vem essa parte do documento".
            lead = (match.group("lead") or "").strip().lower()
            if lead.startswith("d"):
                phrase = "dessa parte do documento"
            elif lead.startswith("n") or lead == "em":
                phrase = "nessa parte do documento"
            else:
                phrase = "essa parte do documento"
            start = match.start()
            prefix = cleaned_so_far[:start].rstrip()
            if not prefix or prefix[-1] in ".!?\n":
                phrase = phrase[0].upper() + phrase[1:]
            return phrase

        cleaned = _PAGE_RE.sub(_page, cleaned)

    # Tidy the holes a removed marker leaves behind.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    return cleaned.strip(), removed


def citation_line(doc_chunks: Sequence[Dict[str, Any]],
                  web_sources: Sequence[str], used: Set[str]) -> str:
    """The "📄 …  🌐 …" line appended to a written reply.

    Only the sources the reply actually cited are listed. Listing everything
    retrieved is how a bot ends up looking like it read four papers it merely
    received — the exact thing Carnall called out ("leste 4 documentos que no seu
    total perfazem mais de 300 páginas last night?").
    """
    parts: List[str] = []
    seen: Set[str] = set()
    for index, chunk in enumerate(doc_chunks, start=1):
        if f"D{index}" not in used:
            continue
        name = chunk.get("filename") or "documento"
        label = _page_label(int(chunk.get("page_start") or 0),
                            int(chunk.get("page_end") or 0))
        entry = f"{name} {label}".strip()
        if entry not in seen:
            seen.add(entry)
            parts.append(entry)
    doc_part = f"📄 {' · '.join(parts)}" if parts else ""

    web_part = ""
    if any(marker.startswith("W") for marker in used) and web_sources:
        domains = []
        for url in web_sources:
            domain = _domain(url)
            if domain and domain not in domains:
                domains.append(domain)
        if domains:
            web_part = f"🌐 {', '.join(domains[:3])}"

    return "  ".join(part for part in (doc_part, web_part) if part)


def cited_markers(reply: str) -> Set[str]:
    """Which sources the reply actually referenced."""
    return {f"{m.group('kind')}{int(m.group('index'))}"
            for m in _MARKER_RE.finditer(reply or "")}


def _domain(url: str) -> str:
    """Host without a leading ``www.``.

    Written out rather than using ``lstrip("www.")``, which takes a character set
    and turns ``wsj.com`` into ``sj.com``.
    """
    from urllib.parse import urlparse

    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:  # noqa: BLE001
        return ""
    return netloc[4:] if netloc.startswith("www.") else netloc
