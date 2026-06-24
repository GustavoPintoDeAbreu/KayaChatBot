#!/usr/bin/env python3
"""Audit logged chat interactions for recurring model weaknesses.

Reads data/feedback/live_interactions.jsonl (written by src/chat/metrics.py) and
flags answers that exhibit known failure modes, so we can track whether prompt /
training changes actually fix them over time. Read-only; prints a summary plus a
few flagged examples per category.

    kaya_chatbot_env/bin/python scripts/audit_interactions.py            # last 7 days
    kaya_chatbot_env/bin/python scripts/audit_interactions.py --days 30
    kaya_chatbot_env/bin/python scripts/audit_interactions.py --all --examples 5
"""

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import metrics
from src.chat.web_search import is_current_events, CITATION_PREFIX

# Failure heuristics: (label, compiled regex over the assistant response).
_CHECKS = {
    "ai_disclaimer": re.compile(
        r"não tenho acesso|modelo de (?:linguagem|ia)|sou (?:um|uma) (?:ia|modelo)"
        r"|knowledge cutoff|limite de conhecimento|corte em janeiro|não posso (?:navegar|aceder)"
        r"|não consigo (?:navegar|aceder|verificar)|tempo real",
        re.IGNORECASE,
    ),
    "sycophancy_or_blame": re.compile(
        r"deve ter confundido|tens toda a razão|peço imensa desculpa"
        r"|o \w+ deve ter|enganaste-te|confundiste",
        re.IGNORECASE,
    ),
    "very_short": None,  # handled separately (length-based)
}


def _flag(row: dict) -> list:
    """Return the list of failure labels triggered by one interaction record."""
    resp = row.get("assistant_response") or ""
    if not isinstance(resp, str):
        resp = str(resp)
    user_msg = row.get("user_message") or ""
    if not isinstance(user_msg, str):
        user_msg = str(user_msg)
    labels = []
    if _CHECKS["ai_disclaimer"].search(resp):
        labels.append("ai_disclaimer")
    if _CHECKS["sycophancy_or_blame"].search(resp):
        labels.append("sycophancy_or_blame")
    if len(resp.split()) < 4:
        labels.append("very_short")
    # current-events question that was NOT web-grounded → likely stale/hallucinated
    if is_current_events(user_msg) and not row.get("web_search_used") \
            and CITATION_PREFIX not in resp:
        labels.append("stale_current_events")
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit logged interactions for model weaknesses.")
    ap.add_argument("--days", type=int, default=7, help="Only audit the last N days (default 7).")
    ap.add_argument("--all", action="store_true", help="Audit all records (ignore --days).")
    ap.add_argument("--examples", type=int, default=3, help="Flagged examples to print per category.")
    ap.add_argument("--log", type=str, default=None, help="Override the interactions log path.")
    args = ap.parse_args()

    rows = metrics.load_interactions(Path(args.log) if args.log else None)
    if not args.all:
        cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=args.days)).isoformat()
        rows = [r for r in rows if (r.get("timestamp") or "") >= cutoff]

    scope = "all time" if args.all else f"last {args.days} days"
    print("=" * 64)
    print(f"📋 Interaction audit — {scope} — {len(rows)} records")
    print("=" * 64)
    if not rows:
        print("No interactions in range. Nothing to audit.")
        return

    # Summary stats (reuse the dashboard aggregator's spirit).
    from collections import Counter
    by_source = Counter(r.get("source", "unknown") for r in rows)
    web_used = sum(1 for r in rows if r.get("web_search_used"))
    lat = [r.get("latency_ms") for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    words = [len((r.get("assistant_response") or "").split()) for r in rows]
    print(f"by source: {dict(by_source)}")
    print(f"web-search used: {web_used}/{len(rows)}")
    if lat:
        print(f"avg latency: {sum(lat)/len(lat):.0f} ms")
    if words:
        print(f"avg response length: {sum(words)/len(words):.0f} words")

    # Flag failures.
    flagged: dict = {}
    for r in rows:
        for label in _flag(r):
            flagged.setdefault(label, []).append(r)

    print("\n--- flagged failure modes ---")
    if not flagged:
        print("✅ none of the tracked failure modes were detected.")
    for label, items in sorted(flagged.items(), key=lambda kv: -len(kv[1])):
        print(f"\n⚠️  {label}: {len(items)} ({100*len(items)/len(rows):.0f}% of records)")
        for r in items[: args.examples]:
            q = (r.get("user_message") or "").replace("\n", " ")[:90]
            a = (r.get("assistant_response") or "").replace("\n", " ")[:140]
            print(f"    Q: {q}")
            print(f"    A: {a}")

    print("\n--- remediation pointers ---")
    print("• ai_disclaimer / stale_current_events → web_search triggers + system-prompt 'use web results' clause")
    print("• sycophancy_or_blame → system-prompt 'acknowledge corrections without blaming' clause (+ next retrain)")
    print("• very_short → check brevity_hint / max_new_tokens_default")


if __name__ == "__main__":
    main()
