"""Web-search answers for out-of-group / current-events questions.

The local fine-tuned model has no live internet and garbles factual data when it
tries to summarize raw web snippets (corrupted scores, dates, prices). So for
questions that aren't about the Kaya group we ask **xAI Grok with its web_search
Agent Tool**, which performs a server-side live search and returns a finished,
factually-grounded answer in European Portuguese with citations. That answer is
used *directly* — the local model never re-synthesizes it — which is what fixed
the previous garbling (a spike scored Grok 5/5 vs the local path 0/3).

Trigger (the query must name no group member, then any of):
  * **explicit request** — "procura na net", "search the web", "google it", … → always;
  * **current-events / recency intent** — scores, prices, release dates, news, "hoje",
    "último jogo", … → when RAG isn't strongly anchored to the group;
  * **low RAG relevance** — ``best_similarity < trigger_similarity`` (nothing close).

Web-grounded answers carry a ``CITATION_PREFIX`` line ("🌐 Fontes: …"). Everything
degrades to an unused result on any failure, so a search outage never breaks a reply.

Privacy: only the user's (off-topic, no-member-named) question goes to Grok — never
the group's private RAG context.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Visible marker prepended to the sources line of a web-grounded reply. Also used by
# the WhatsApp server to detect web usage for metrics (stateless, no signature change).
CITATION_PREFIX = "🌐 Fontes:"

# Instruction given to Grok for web-grounded answers: factual, current, European-PT,
# short. The group's private memory is never included here.
_GROK_SYSTEM = (
    "És o assistente do grupo de amigos 'Kaya'. Respondes a esta pergunta com base em "
    "informação atual e verdadeira da web, em português europeu, de forma directa e "
    "factual, em 1 a 3 frases. Não inventes números, datas ou nomes; se a informação não "
    "for clara, di-lo. Não uses emojis."
)

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
    """Outcome of a (possibly skipped) web search.

    For the Grok path ``answer`` holds the finished reply to send to the user
    directly. ``context`` is retained for backward compatibility (always "" now).
    """

    used: bool = False
    answer: str = ""            # finished web-grounded reply (Grok), used as-is
    context: str = ""           # legacy: prompt-injection block (unused with Grok)
    sources: List[str] = field(default_factory=list)  # citation URLs

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


class GrokSearchClient:
    """Thin wrapper over the xAI SDK using the server-side web_search Agent Tool."""

    def __init__(self, api_key: str, model: str, excluded_domains: Optional[List[str]] = None):
        self.api_key = api_key
        self.model = model
        # xAI's web_search tool allows at most 5 excluded domains.
        self.excluded_domains = [d for d in (excluded_domains or []) if d][:5]

    def search(self, query: str) -> tuple:
        """Return ``(answer_text, [citation_urls])`` for ``query`` (or ("", []) on failure)."""
        from xai_sdk import Client
        from xai_sdk.chat import system, user
        from xai_sdk.tools import web_search

        client = Client(api_key=self.api_key)
        chat = client.chat.create(
            model=self.model,
            messages=[system(_GROK_SYSTEM), user(query)],
            tools=[web_search(excluded_domains=self.excluded_domains or None)],
        )
        resp = chat.sample()
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        try:
            cites = list(resp.citations)
        except Exception:  # noqa: BLE001
            cites = []
        return text, cites


_client: Optional[GrokSearchClient] = None


def _get_client(config: Dict[str, Any]) -> Optional[GrokSearchClient]:
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        return None
    ws_cfg = config.get("web_search", {}) or {}
    model = ws_cfg.get("model") or config.get("generation", {}).get("xai", {}).get(
        "model", "grok-4-1-fast-reasoning"
    )
    _client = GrokSearchClient(
        api_key=api_key, model=str(model),
        excluded_domains=list(ws_cfg.get("exclude_domains", []) or []),
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
    if not os.environ.get("XAI_API_KEY", "").strip():
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
    """Return a ``WebSearchResult`` with a finished ``answer``; never raises."""
    try:
        if not should_search(query, retriever, config, query_embedding=query_embedding):
            return WebSearchResult()
        client = _get_client(config)
        if client is None:
            return WebSearchResult()
        answer, sources = client.search(query)
        if not answer:
            return WebSearchResult()
        return WebSearchResult(used=True, answer=answer, sources=sources)
    except Exception as exc:  # noqa: BLE001 — never break a reply on search
        print(f"⚠️  web search failed: {exc}")
        return WebSearchResult()
