"""User feedback: thumbs up/down, written reasons, and bug reports.

This is the quality-signal counterpart to ``src/chat/metrics.py``. Where metrics
records *every* answered turn, this module records the *explicit* signal a user
chooses to give: a 👍/👎 on an answer (web thumbs or a WhatsApp emoji reaction), an
optional written reason for a 👎, and "the app is broken" bug reports from the web UI.

Two append-only JSONL sinks, kept separate from the interaction log so the existing
dashboard/tooling is untouched:

  * ``message_feedback.jsonl`` — ratings (``rating: up|down``) and reason comments
    (``type: comment``), joined on ``feedback_id``.
  * ``bug_reports.jsonl`` — web bug reports.

Invariant (same as metrics): logging must never raise into the caller. A feedback
failure can never drop or delay a user's reply, so everything is wrapped defensively.
"""

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "feedback"
_DEFAULT_FEEDBACK_LOG = _DATA_DIR / "message_feedback.jsonl"
_DEFAULT_BUG_LOG = _DATA_DIR / "bug_reports.jsonl"


def _resolve(custom: Optional[str], default: Path) -> Path:
    """Resolve a configured log path; relative paths hang off the repo root."""
    if custom:
        p = Path(custom)
        return p if p.is_absolute() else (_DATA_DIR.parent.parent / custom)
    return default


def feedback_log_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """JSONL sink for ratings/reasons, honouring ``chat.feedback.log_file``."""
    cfg = (config or {}).get("chat", {}).get("feedback", {}) or {}
    return _resolve(cfg.get("log_file"), _DEFAULT_FEEDBACK_LOG)


def bug_log_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """JSONL sink for bug reports, honouring ``chat.bug_report.log_file``."""
    cfg = (config or {}).get("chat", {}).get("bug_report", {}) or {}
    return _resolve(cfg.get("log_file"), _DEFAULT_BUG_LOG)


def _append(sink: Path, entry: Dict[str, Any]) -> None:
    """Append one record to a JSONL sink. Never raises."""
    try:
        sink.parent.mkdir(parents=True, exist_ok=True)
        with open(sink, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 — feedback must never break a reply
        print(f"⚠️  feedback log failed: {exc}")


def log_rating(
    *,
    source: str,
    rating: str,
    user_message: str = "",
    assistant_response: str = "",
    interaction_id: Optional[str] = None,
    comment: Optional[str] = None,
    path: Optional[Path] = None,
    **extra: Any,
) -> str:
    """Record a 👍/👎 on an answer. Returns the new ``feedback_id``.

    ``source`` is the surface (``web`` / ``whatsapp``); ``rating`` is ``up`` or
    ``down``. ``extra`` is merged verbatim (e.g. ``is_group`` for WhatsApp). The
    returned id lets the web reason-box attach a follow-up comment via ``log_comment``.
    """
    feedback_id = str(uuid.uuid4())
    entry = {
        "feedback_id": feedback_id,
        "type": "rating",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "rating": rating,
        "user_message": user_message or "",
        "assistant_response": assistant_response or "",
        "interaction_id": interaction_id,
        "comment": comment or None,
    }
    entry.update(extra)
    _append(Path(path) if path else _DEFAULT_FEEDBACK_LOG, entry)
    return feedback_id


def log_comment(
    *,
    feedback_id: str,
    source: str,
    comment: str,
    path: Optional[Path] = None,
) -> None:
    """Attach a written reason to an earlier rating (joined on ``feedback_id``)."""
    if not feedback_id or not (comment or "").strip():
        return
    entry = {
        "feedback_id": feedback_id,
        "type": "comment",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "comment": comment.strip(),
    }
    _append(Path(path) if path else _DEFAULT_FEEDBACK_LOG, entry)


def log_bug_report(
    *,
    source: str,
    description: str,
    contact: Optional[str] = None,
    env: Optional[str] = None,
    version: Optional[str] = None,
    recent_turns: Optional[List[str]] = None,
    path: Optional[Path] = None,
) -> str:
    """Record a bug report. Returns the new ``report_id``.

    ``recent_turns`` is a few of the latest chat lines kept for repro context.
    ``_notify_bug_report`` is invoked as a side-channel (no-op today; the seam where
    email/SMTP lands later — see the module/plan notes).
    """
    report_id = str(uuid.uuid4())
    entry = {
        "report_id": report_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "description": (description or "").strip(),
        "contact": (contact or "").strip() or None,
        "env": env or None,
        "version": version or None,
        "recent_turns": recent_turns or [],
    }
    _append(Path(path) if path else _DEFAULT_BUG_LOG, entry)
    _notify_bug_report(entry)
    return report_id


def _notify_bug_report(record: Dict[str, Any]) -> None:
    """Side-channel notification hook for a new bug report. No-op for now.

    Email delivery is intentionally deferred (current choice: file log only). This is
    the single place to later add SMTP (``smtplib``) driven by env vars, without
    changing the on-disk schema. Must never raise.
    """
    return None


def _load(sink: Path) -> List[Dict[str, Any]]:
    """Read all records from a JSONL sink, skipping malformed lines."""
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


def load_feedback(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    return _load(Path(path) if path else _DEFAULT_FEEDBACK_LOG)


def load_bug_reports(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    return _load(Path(path) if path else _DEFAULT_BUG_LOG)


def aggregate_feedback(
    feedback_path: Optional[Path] = None,
    bug_path: Optional[Path] = None,
    recent: int = 10,
) -> Dict[str, Any]:
    """Summarise ratings + bug reports for the dashboard.

    Returns up/down totals, per-source rating counts, the most recent down-votes (with
    their reason comments stitched in by ``feedback_id``), and recent bug reports.
    """
    rows = load_feedback(feedback_path)
    ratings = [r for r in rows if r.get("type", "rating") == "rating"]
    comments = {
        r.get("feedback_id"): r.get("comment", "")
        for r in rows
        if r.get("type") == "comment"
    }

    up = sum(1 for r in ratings if r.get("rating") == "up")
    down = sum(1 for r in ratings if r.get("rating") == "down")
    by_source: Counter = Counter()
    for r in ratings:
        by_source[(r.get("source", "unknown"), r.get("rating", "?"))] += 1

    recent_down = []
    for r in reversed(ratings):
        if r.get("rating") != "down":
            continue
        reason = r.get("comment") or comments.get(r.get("feedback_id"), "")
        recent_down.append(
            {
                "timestamp": r.get("timestamp", ""),
                "source": r.get("source", ""),
                "user_message": r.get("user_message", ""),
                "reason": reason or "",
            }
        )
        if len(recent_down) >= recent:
            break

    bugs = load_bug_reports(bug_path)
    recent_bugs = [
        {
            "timestamp": b.get("timestamp", ""),
            "description": b.get("description", ""),
            "contact": b.get("contact") or "",
            "version": b.get("version") or "",
        }
        for b in list(reversed(bugs))[:recent]
    ]

    return {
        "total_ratings": len(ratings),
        "up": up,
        "down": down,
        "by_source": {f"{src}:{rating}": count for (src, rating), count in by_source.items()},
        "recent_down": recent_down,
        "bug_total": len(bugs),
        "recent_bugs": recent_bugs,
    }
