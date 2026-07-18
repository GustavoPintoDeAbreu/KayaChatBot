"""Retrieval-quality evaluation for the Kaya RAG system.

Measures the *retrieval layer* directly (recall@k / hit-rate@k / MRR) against a
labelled query set, independent of the generator. This is the missing piece the
end-to-end golden suite could not give us: a before/after number for every
retrieval change (chunking, hybrid search, person-filter, recency).

Gold labels are **content-based**, not chunk-id based: chunk ids (``chunk_N``)
are reassigned on every rebuild, so a gold label is a set of substrings that must
co-occur inside a single retrieved chunk. A query may have several gold specs
(several relevant chunks) — recall@k is then the fraction of gold specs covered
by the top-k chunks.

No GPU required: only the bge-m3 embedding model runs (CPU is fine).

Usage:
    kaya_chatbot_env/bin/python -m src.testing.retrieval_eval
    kaya_chatbot_env/bin/python -m src.testing.retrieval_eval --gold data/golden_retrieval.json
    kaya_chatbot_env/bin/python -m src.testing.retrieval_eval --compare-person-filter
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).parent.parent.parent
import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config_loader import load_config
from src.chat.retriever import ConversationRetriever

DEFAULT_GOLD = _REPO_ROOT / "data" / "golden_retrieval.json"
REPORTS_DIR = _REPO_ROOT / "reports" / "benchmarks"
K_VALUES = (1, 3, 5, 10)


def _norm(text: str) -> str:
    """Casefold + strip accents so gold substrings match code-switched PT/EN."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.casefold()


def _spec_matches_chunk(spec: List[str], chunk_text: str) -> bool:
    """A gold spec matches a chunk when *all* its substrings co-occur in it."""
    haystack = _norm(chunk_text)
    return all(_norm(term) in haystack for term in spec)


def _load_gold(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data.get("queries", data if isinstance(data, list) else [])
    normalised = []
    for q in queries:
        gold = q.get("gold", [])
        # Accept a flat list of strings as a single AND-spec for convenience.
        if gold and isinstance(gold[0], str):
            gold = [gold]
        normalised.append({**q, "gold": gold})
    return normalised


def _first_hit_rank(chunks: List[Dict[str, Any]], gold: List[List[str]]) -> Optional[int]:
    for rank, chunk in enumerate(chunks, 1):
        if any(_spec_matches_chunk(spec, chunk["text"]) for spec in gold):
            return rank
    return None


def _specs_covered(chunks: List[Dict[str, Any]], gold: List[List[str]]) -> int:
    covered = 0
    for spec in gold:
        if any(_spec_matches_chunk(spec, chunk["text"]) for chunk in chunks):
            covered += 1
    return covered


def evaluate(
    retriever: ConversationRetriever,
    queries: List[Dict[str, Any]],
    max_k: int = max(K_VALUES),
    disable_person_filter: bool = False,
) -> Dict[str, Any]:
    """Run every query and aggregate recall@k / hit-rate@k / MRR."""
    orig_extract = retriever.extract_query_persons
    if disable_person_filter:
        retriever.extract_query_persons = lambda _q: []  # type: ignore

    per_query = []
    try:
        for q in queries:
            gold = q["gold"]
            if not gold:
                continue
            t0 = time.perf_counter()
            chunks = retriever.retrieve(q["query"], top_k=max_k)
            latency = time.perf_counter() - t0

            first_rank = _first_hit_rank(chunks, gold)
            row = {
                "id": q.get("id", q["query"][:40]),
                "category": q.get("category", "uncategorised"),
                "num_gold": len(gold),
                "first_hit_rank": first_rank,
                "mrr": (1.0 / first_rank) if first_rank else 0.0,
                "latency_ms": round(latency * 1000, 1),
            }
            for k in K_VALUES:
                topk = chunks[:k]
                row[f"hit@{k}"] = 1.0 if (first_rank and first_rank <= k) else 0.0
                row[f"recall@{k}"] = _specs_covered(topk, gold) / len(gold)
            per_query.append(row)
    finally:
        retriever.extract_query_persons = orig_extract  # type: ignore

    n = len(per_query) or 1
    agg: Dict[str, Any] = {"num_queries": len(per_query)}
    agg["mrr"] = round(sum(r["mrr"] for r in per_query) / n, 4)
    for k in K_VALUES:
        agg[f"hit@{k}"] = round(sum(r[f"hit@{k}"] for r in per_query) / n, 4)
        agg[f"recall@{k}"] = round(sum(r[f"recall@{k}"] for r in per_query) / n, 4)

    # Per-category recall@5 (headline diagnostic).
    cats: Dict[str, List[float]] = {}
    for r in per_query:
        cats.setdefault(r["category"], []).append(r["recall@5"])
    agg["by_category_recall@5"] = {
        c: round(sum(v) / len(v), 4) for c, v in sorted(cats.items())
    }
    return {"aggregate": agg, "per_query": per_query}


def main() -> None:
    ap = argparse.ArgumentParser(description="Kaya retrieval-quality eval")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD), help="Path to golden_retrieval.json")
    ap.add_argument("--label", default="", help="Label written into the report (e.g. 'baseline')")
    ap.add_argument("--compare-person-filter", action="store_true",
                    help="Also run with the person-filter disabled and print the delta")
    ap.add_argument("--no-save", action="store_true", help="Do not write a report file")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        raise SystemExit(f"Gold file not found: {gold_path}")
    queries = _load_gold(gold_path)
    print(f"📋 Loaded {len(queries)} labelled queries from {gold_path.name}")

    config = load_config(str(_REPO_ROOT / "config.yaml"))
    retriever = ConversationRetriever(config)
    retriever.initialize()

    result = evaluate(retriever, queries)
    agg = result["aggregate"]
    print("\n=== Retrieval quality" + (f" [{args.label}]" if args.label else "") + " ===")
    print(f"queries={agg['num_queries']}  MRR={agg['mrr']}")
    for k in K_VALUES:
        print(f"  k={k:<2}  hit@k={agg[f'hit@{k}']:.3f}  recall@k={agg[f'recall@{k}']:.3f}")
    print("  recall@5 by category:")
    for c, v in agg["by_category_recall@5"].items():
        print(f"    {c:<14} {v:.3f}")

    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "gold_file": gold_path.name,
        "config": {
            "embedding_model": config["rag"]["embedding_model"],
            "top_k": config["rag"]["top_k"],
            "min_similarity": config["rag"]["min_similarity"],
            "chunk_size_tokens": config["rag"]["chunk_size_tokens"],
            "filter_by_person": config["rag"]["filter_by_person"],
        },
        "default": result,
    }

    if args.compare_person_filter:
        nofilter = evaluate(retriever, queries, disable_person_filter=True)
        payload["no_person_filter"] = nofilter
        print("\n=== Person-filter impact (default − no_filter), recall@5 ===")
        d = agg["recall@5"]
        nf = nofilter["aggregate"]["recall@5"]
        print(f"  default={d:.3f}  no_filter={nf:.3f}  delta={d - nf:+.3f}")

    if not args.no_save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = REPORTS_DIR / f"retrieval_{ts}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n💾 Report written to {out}")


if __name__ == "__main__":
    main()
