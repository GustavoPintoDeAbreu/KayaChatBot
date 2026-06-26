"""Optional web-search fallback for out-of-group / general-knowledge questions.

The local model has no live internet and the LLM providers expose no tool-use, so
for questions that aren't about the Kaya group we fetch a few web snippets from an
external search API (Tavily) and inject them as extra context. The already-loaded
Gemma then summarizes them in its normal generation pass — no second model, no extra
VRAM.

Trigger (the query must name no group member, then any of):
  * **explicit request** — the user asks the bot to search ("procura na net", "search
    the web", "google it", …) → always search;
  * **current-events / recency intent** — scores, prices, release dates, news, "hoje",
    "último jogo", … → search when RAG isn't strongly anchored to the group;
  * **low RAG relevance** — ``best_similarity < trigger_similarity`` (nothing close).

Web-grounded answers carry a ``CITATION_PREFIX`` line ("🌐 Fontes: …") so the user can
see it searched and verify the source. Everything degrades to an unused result on any
failure, so a search outage never breaks a reply.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

_TAVILY_ENDPOINT = "https://api.tavily.com/search"

# Visible marker prepended to the sources line of a web-grounded reply. Also used by
# the WhatsApp server to detect web usage for metrics (stateless, no signature change).
CITATION_PREFIX = "🌐 Fontes:"

# The user explicitly asks the bot to look something up online → always search.
_EXPLICIT_SEARCH_RE = re.compile(
    r"\bprocura(?:r|s)?\b.*\b(net|internet|google|web|online)\b"
    r"|\bpesquisa(?:r|s)?\b"
    r"|\bv[êe]\s+(?:na\s+)?(?:net|internet|google)\b"
    r"|\bconfirma(?:r)?\b.*\bonline\b"
    r"|\b[úu]ltimas?\s+not[íi]cias\b"
    r"|\bsearch\s+(?:the\s+)?(?:web|online|net|it up|for)\b"
    r"|\bgoogle\s+(?:it|this|that|isto|isso)\b"
    r"|\blook\s+(?:it|this)\s+up\b",
    re.IGNORECASE,
)

# Current-events / recency cues: questions whose answer changes over time and that the
# group's static memory can't answer reliably (sports results, prices, releases, news).
_CURRENT_EVENTS_RE = re.compile(
    r"\bquem\s+ganhou\b|\bresultado\b|\b[úu]ltimo\s+jogo\b|\bplacar\b|\bmarcou\b"
    r"|\bpre[çc]o\b|\bquanto\s+custa\b|\bcusta\s+quanto\b"
    r"|\bdata\s+de\s+lan[çc]amento\b|\bquando\s+(?:sai|lan[çc]a|estreia)\b"
    r"|\bnot[íi]cia\b|\baconteceu\s+(?:no\s+mundo|hoje|ontem|recentemente|esta\s+semana)\b"
    r"|\bo\s+que\s+se\s+passa\s+(?:no\s+mundo|com)\b"
    r"|\bhoje\b|\bontem\b|\besta\s+semana\b|\beste\s+ano\b|\bagora\b|\batual(?:mente)?\b"
    r"|\bneste\s+momento\b|\bprevis[ãa]o\s+do\s+tempo\b|\bcota[çc][ãa]o\b"
    r"|\bscore\b|\bwho\s+won\b|\bprice\b|\brelease\s+date\b|\bweather\b|\bnews\b|\blatest\b",
    re.IGNORECASE,
)

# Questions explicitly about the group's own history/memory ("a melhor cena que aconteceu no
# grupo", "o que se passou nas conversas"). These are answered from RAG, never the live web,
# even when they carry an incidental recency cue like "aconteceu" — otherwise they false-trigger.
_GROUP_HISTORY_RE = re.compile(
    r"\b(?:n[oa]|d[oa]|neste|deste)\s+grupo\b|\bn[oa]\s+kaya\b|\bd[oa]\s+kaya\b"
    r"|\bnas?\s+conversas?\b|\bno\s+chat\b",
    re.IGNORECASE,
)


@dataclass
class WebSearchResult:
    """Outcome of a (possibly skipped) web search."""

    used: bool = False
    context: str = ""           # formatted block to inject into the prompt
    sources: List[str] = field(default_factory=list)  # result URLs

    def citation_line(self) -> str:
        """A compact '🌐 Fontes: domain1, domain2' line for the reply, or ""."""
        return citation_line(self.sources)


def citation_line(sources: List[str]) -> str:
    """Render up to 2 distinct source domains as a citation line (or "")."""
    domains: List[str] = []
    for url in sources:
        if not url:
            continue
        try:
            host = urlparse(url).netloc.lower().lstrip("www.")
        except Exception:  # noqa: BLE001
            host = ""
        if host and host not in domains:
            domains.append(host)
        if len(domains) >= 2:
            break
    return f"{CITATION_PREFIX} {', '.join(domains)}" if domains else ""


class TavilyClient:
    """Minimal Tavily search client (lazy ``httpx``), with simple backoff retry."""

    def __init__(self, api_key: str, max_results: int = 3, search_depth: str = "advanced",
                 timeout: float = 10.0, exclude_domains: Optional[List[str]] = None):
        self.api_key = api_key
        self.max_results = max_results
        self.search_depth = search_depth
        self.timeout = timeout
        self.exclude_domains = [d for d in (exclude_domains or []) if d]

    def search(self, query: str) -> List[Dict[str, str]]:
        """Return a list of ``{title, url, content}`` results (or [] on failure)."""
        import httpx

        body = {
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
            "search_depth": self.search_depth,
            "include_answer": True,
        }
        if self.exclude_domains:
            body["exclude_domains"] = self.exclude_domains
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = httpx.post(_TAVILY_ENDPOINT, json=body, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                results: List[Dict[str, str]] = []
                answer = data.get("answer")
                if answer:
                    results.append({"title": "Resumo", "url": "", "content": answer})
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", ""),
                    })
                return results[: self.max_results + 1]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(0.5 * (2 ** attempt))
        print(f"⚠️  Tavily search failed: {last_exc}")
        return []


def _truncate_clean(text: str, limit: int = 400) -> str:
    """Trim ``text`` to ``limit`` chars at the last word boundary (no mid-word cuts)."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:  # only back off to a space if it isn't pathologically early
        cut = cut[:space]
    return cut.rstrip() + "…"


def _format_results(results: List[Dict[str, str]]) -> str:
    if not results:
        return ""
    lines = ["=== Resultados de pesquisa web (informação externa e atual) ==="]
    for r in results:
        snippet = (r.get("content") or "").strip().replace("\n", " ")
        if not snippet:
            continue
        src = f" ({r['url']})" if r.get("url") else ""
        lines.append(f"- {_truncate_clean(snippet, 400)}{src}")
    lines.append("=== Fim dos resultados web ===")
    return "\n".join(lines) if len(lines) > 2 else ""


_client: Optional[TavilyClient] = None


def _get_client(config: Dict[str, Any]) -> Optional[TavilyClient]:
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return None
    ws_cfg = config.get("web_search", {}) or {}
    _client = TavilyClient(
        api_key=api_key,
        max_results=int(ws_cfg.get("max_results", 3)),
        search_depth=str(ws_cfg.get("search_depth", "advanced")),
        exclude_domains=list(ws_cfg.get("exclude_domains", []) or []),
    )
    return _client


def is_explicit_request(query: str) -> bool:
    return bool(query) and bool(_EXPLICIT_SEARCH_RE.search(query))


def is_current_events(query: str) -> bool:
    return bool(query) and bool(_CURRENT_EVENTS_RE.search(query))


def should_search(query: str, retriever, config: Dict[str, Any], query_embedding=None) -> bool:
    """Decide whether to web-search for ``query`` (see module docstring)."""
    ws_cfg = config.get("web_search", {}) or {}
    if not ws_cfg.get("enabled", False) or retriever is None:
        return False
    if not os.environ.get("TAVILY_API_KEY", "").strip():
        return False
    # A named group member ⇒ group question ⇒ never search.
    try:
        if retriever.extract_query_persons(query):
            return False
    except Exception:  # noqa: BLE001
        pass
    # Explicit user request always wins.
    if is_explicit_request(query):
        return True
    # Group-history questions are about the group's own memory, not the live web.
    if _GROUP_HISTORY_RE.search(query):
        return False
    # Relevance probe (shared embedding when provided).
    try:
        score = retriever.best_similarity(query, query_embedding=query_embedding)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  web-search relevance probe failed: {exc}")
        return False
    # Current-events intent: search unless the query is strongly anchored to the group.
    if is_current_events(query) and score < float(ws_cfg.get("current_events_max", 0.60)):
        return True
    # Low-RAG fallback: nothing close in the group's memory.
    return score < float(ws_cfg.get("trigger_similarity", 0.40))


def maybe_web_search(query: str, retriever, config: Dict[str, Any], query_embedding=None) -> WebSearchResult:
    """Return a ``WebSearchResult`` (``used``/``context``/``sources``); never raises."""
    try:
        if not should_search(query, retriever, config, query_embedding=query_embedding):
            return WebSearchResult()
        client = _get_client(config)
        if client is None:
            return WebSearchResult()
        results = client.search(query)
        context = _format_results(results)
        if not context:
            return WebSearchResult()
        sources = [r["url"] for r in results if r.get("url")]
        return WebSearchResult(used=True, context=context, sources=sources)
    except Exception as exc:  # noqa: BLE001 — never break a reply on search
        print(f"⚠️  web search failed: {exc}")
        return WebSearchResult()
