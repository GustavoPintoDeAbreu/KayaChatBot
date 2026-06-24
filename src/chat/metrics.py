"""Interaction metrics: one structured JSONL sink shared by every surface.

Every answered turn — web UI, WhatsApp bridge, CLI — appends one record here so
the dashboard (and any later analysis) has a single source of truth. The schema
extends the original 4-field interaction log with ``source``, response length,
latency and any extra fields a caller passes (e.g. ``web_search_used``).

Invariant: logging must never raise into the caller. A metrics failure can never
drop or delay a user's reply, so everything is wrapped defensively.
"""

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default sink: the historical interaction log, so existing data + tooling keep working.
_DEFAULT_LOG = Path(__file__).resolve().parent.parent.parent / "data" / "feedback" / "live_interactions.jsonl"


def log_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """Resolve the JSONL sink path, honouring ``metrics.log_file`` if configured."""
    if config:
        cfg = config.get("metrics", {}) or {}
        custom = cfg.get("log_file")
        if custom:
            p = Path(custom)
            return p if p.is_absolute() else (_DEFAULT_LOG.parent.parent.parent / custom)
    return _DEFAULT_LOG


def log_interaction(
    *,
    source: str,
    user_message: str,
    assistant_response: str,
    latency_ms: Optional[float] = None,
    path: Optional[Path] = None,
    **extra: Any,
) -> None:
    """Append one interaction record. Never raises.

    ``source`` is the surface (``web`` / ``whatsapp`` / ``cli``). ``extra`` is
    merged verbatim, so callers can add ``rag_relevance``, ``web_search_used`` etc.
    """
    try:
        sink = Path(path) if path else _DEFAULT_LOG
        sink.parent.mkdir(parents=True, exist_ok=True)
        response = assistant_response or ""
        entry = {
            "interaction_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "user_message": user_message or "",
            "assistant_response": response,
            "response_chars": len(response),
            "response_words": len(response.split()),
            "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
        }
        entry.update(extra)
        with open(sink, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 — metrics must never break a reply
        print(f"⚠️  metrics log failed: {exc}")


def load_interactions(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read all interaction records, skipping any malformed lines."""
    sink = Path(path) if path else _DEFAULT_LOG
    rows: List[Dict[str, Any]] = []
    if not sink.exists():
        return rows
    for line in sink.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return rows


def _avg(values: List[Any]) -> float:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 1) if nums else 0.0


def aggregate(path: Optional[Path] = None) -> Dict[str, Any]:
    """Compute dashboard metrics from the log.

    Returns total volume, per-source counts, average response length (words) and
    latency (ms), web-search usage rate, and per-day volume for plotting. Records
    predating the enriched schema are handled gracefully (length is recomputed).
    """
    rows = load_interactions(path)
    total = len(rows)
    by_source: Counter = Counter(r.get("source", "unknown") for r in rows)
    words = [
        r.get("response_words")
        if isinstance(r.get("response_words"), (int, float))
        else len((r.get("assistant_response") or "").split())
        for r in rows
    ]
    latencies = [r.get("latency_ms") for r in rows]
    web_searches = sum(1 for r in rows if r.get("web_search_used"))
    per_day = Counter((r.get("timestamp") or "")[:10] for r in rows if r.get("timestamp"))
    return {
        "total": total,
        "by_source": dict(by_source),
        "avg_response_words": _avg(words),
        "avg_response_chars": _avg([r.get("response_chars") for r in rows]),
        "avg_latency_ms": _avg(latencies),
        "web_search_rate": round(web_searches / total, 3) if total else 0.0,
        "per_day": dict(sorted(per_day.items())),
    }
