"""Optional web-search fallback for out-of-group / general-knowledge questions.

The local model has no live internet and the LLM providers expose no tool-use, so
when a question clearly isn't about the Kaya group we fetch a few web snippets from
an external search API (Tavily) and inject them as extra context. The already-loaded
Gemma then summarizes them in its normal generation pass — no second model, no extra
VRAM.

Trigger (all must hold): the feature is enabled, a ``TAVILY_API_KEY`` is present, the
query mentions no known group member, and the retriever's ``best_similarity`` is below
``web_search.trigger_similarity`` (RAG has nothing close → out-of-group). Everything
degrades to ``""`` (no augmentation) on any failure, so a search outage never breaks a
reply.
"""

import os
import time
from typing import Any, Dict, List, Optional

_TAVILY_ENDPOINT = "https://api.tavily.com/search"


class TavilyClient:
    """Minimal Tavily search client (lazy ``httpx``), with simple backoff retry."""

    def __init__(self, api_key: str, max_results: int = 3, timeout: float = 8.0):
        self.api_key = api_key
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> List[Dict[str, str]]:
        """Return a list of ``{title, url, content}`` results (or [] on failure)."""
        import httpx

        body = {
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
            "search_depth": "basic",
            "include_answer": True,
        }
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = httpx.post(_TAVILY_ENDPOINT, json=body, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                results = []
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


def _format_results(query: str, results: List[Dict[str, str]]) -> str:
    if not results:
        return ""
    lines = ["=== Resultados de pesquisa web (informação externa, atual) ==="]
    for r in results:
        snippet = (r.get("content") or "").strip().replace("\n", " ")
        if not snippet:
            continue
        src = f" ({r['url']})" if r.get("url") else ""
        lines.append(f"- {snippet[:400]}{src}")
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
    _client = TavilyClient(api_key=api_key, max_results=int(ws_cfg.get("max_results", 3)))
    return _client


def should_search(query: str, retriever, config: Dict[str, Any], query_embedding=None) -> bool:
    """Decide whether ``query`` is an out-of-group question worth searching."""
    ws_cfg = config.get("web_search", {}) or {}
    if not ws_cfg.get("enabled", False) or retriever is None:
        return False
    if not os.environ.get("TAVILY_API_KEY", "").strip():
        return False
    # If a known group member is named, treat it as a group question — never search.
    try:
        if retriever.extract_query_persons(query):
            return False
    except Exception:  # noqa: BLE001
        pass
    threshold = float(ws_cfg.get("trigger_similarity", 0.40))
    try:
        return retriever.best_similarity(query, query_embedding=query_embedding) < threshold
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  web-search relevance probe failed: {exc}")
        return False


def maybe_web_search(query: str, retriever, config: Dict[str, Any], query_embedding=None) -> str:
    """Return a formatted web-results context block, or "" if not triggered/failed."""
    try:
        if not should_search(query, retriever, config, query_embedding=query_embedding):
            return ""
        client = _get_client(config)
        if client is None:
            return ""
        return _format_results(query, client.search(query))
    except Exception as exc:  # noqa: BLE001 — never break a reply on search
        print(f"⚠️  web search failed: {exc}")
        return ""
