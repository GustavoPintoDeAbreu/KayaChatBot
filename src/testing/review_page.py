"""A local page for rating what the model made of the group's content.

The gates for links and stickers are human judgements: does this synopsis say
what the page says, does this reading of a sticker match what the sender meant.
No cloud judge can make them, because the items are group data (the privacy
invariant), so a person does it here, on this PC, bound to 127.0.0.1.

    items.json  →  review_page.serve(run_dir)  →  ratings.json

``items.json`` is a list of::

    {"id": "...", "title": "...", "link": "https://… (optional)",
     "image": "relative/path.webp (optional)", "context": ["line", ...],
     "candidates": [{"key": "A", "text": "..."}, ...]}

Candidates are shown in a per-item shuffled order with their keys hidden, so a
variant cannot be favoured for being the one you expect to win. Every click is
written to ``ratings.json`` at once; closing the tab loses nothing.
"""
from __future__ import annotations

import html
import json
import mimetypes
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

RATINGS = ("right", "partly", "wrong")


def load_ratings(run_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{item_id: {candidate_key: {"rating": ..., "invented": bool}}}``."""
    try:
        return json.loads((Path(run_dir) / "ratings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def score(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Per candidate key: how many rated, and the share right / partly / wrong / invented."""
    totals: Dict[str, Dict[str, int]] = {}
    for by_key in load_ratings(run_dir).values():
        for key, verdict in by_key.items():
            bucket = totals.setdefault(key, {"n": 0, "right": 0, "partly": 0,
                                             "wrong": 0, "invented": 0})
            if verdict.get("rating") in RATINGS:
                bucket["n"] += 1
                bucket[verdict["rating"]] += 1
            if verdict.get("invented"):
                bucket["invented"] += 1
    return {key: {**counts, "right_rate": counts["right"] / counts["n"] if counts["n"] else 0.0}
            for key, counts in totals.items()}


_STYLE = """
:root { --bg:#fafaf7; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e3e2dc;
        --right:#2f7d4f; --partly:#b07d12; --wrong:#b23b3b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161615; --fg:#ecebe6; --muted:#9a9993; --card:#1f1f1d; --line:#33332f; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.5 system-ui, sans-serif; }
main { max-width:860px; margin:0 auto; padding:24px 16px 80px; }
h1 { font-size:20px; margin:0 0 4px; }
.progress { color:var(--muted); margin-bottom:24px; }
.item { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:16px; margin-bottom:18px; }
.item h2 { font-size:15px; margin:0 0 8px; word-break:break-all; }
.item img { max-width:220px; max-height:220px; display:block; margin:8px 0;
            background:repeating-conic-gradient(#ccc 0 25%, #eee 0 50%) 0 0/16px 16px; }
.context { font-size:13px; color:var(--muted); white-space:pre-wrap; margin:8px 0;
           border-left:3px solid var(--line); padding-left:10px; }
.cand { border-top:1px solid var(--line); padding:10px 0 4px; }
.cand p { margin:0 0 8px; white-space:pre-wrap; }
button { font:inherit; border:1px solid var(--line); background:transparent; color:var(--fg);
         border-radius:6px; padding:4px 12px; margin-right:6px; cursor:pointer; }
button.on.right { background:var(--right); color:#fff; border-color:var(--right); }
button.on.partly { background:var(--partly); color:#fff; border-color:var(--partly); }
button.on.wrong { background:var(--wrong); color:#fff; border-color:var(--wrong); }
label { font-size:13px; color:var(--muted); margin-left:8px; }
a { color:inherit; }
"""

_SCRIPT = """
async function rate(item, key, field, value) {
  const response = await fetch('/rate', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({item, key, field, value})});
  const data = await response.json();
  document.querySelector('.progress').textContent = data.progress;
}
document.addEventListener('click', event => {
  const button = event.target.closest('button[data-rating]');
  if (!button) return;
  const box = button.closest('.cand');
  box.querySelectorAll('button').forEach(other => other.classList.remove('on'));
  button.classList.add('on');
  rate(box.dataset.item, box.dataset.key, 'rating', button.dataset.rating);
});
document.addEventListener('change', event => {
  if (!event.target.matches('input[data-invented]')) return;
  const box = event.target.closest('.cand');
  rate(box.dataset.item, box.dataset.key, 'invented', event.target.checked);
});
"""


def _progress(items: List[Dict[str, Any]], ratings: Dict[str, Any]) -> str:
    total = sum(len(item.get("candidates", [])) for item in items)
    done = sum(1 for by_key in ratings.values() for verdict in by_key.values()
               if verdict.get("rating"))
    return f"{done} of {total} rated"


def render(items: List[Dict[str, Any]], ratings: Dict[str, Any], title: str,
           question: str) -> str:
    blocks = []
    for item in items:
        item_id = str(item["id"])
        candidates = list(item.get("candidates", []))
        random.Random(item_id).shuffle(candidates)
        parts = [f'<section class="item"><h2>{html.escape(item.get("title", item_id))}</h2>']
        if item.get("link"):
            safe = html.escape(item["link"], quote=True)
            parts.append(f'<a href="{safe}" target="_blank" rel="noreferrer">abrir o link</a>')
        if item.get("image"):
            parts.append(f'<img src="/file/{html.escape(item["image"], quote=True)}" alt="">')
        if item.get("context"):
            parts.append(f'<div class="context">{html.escape(chr(10).join(item["context"]))}</div>')
        for number, candidate in enumerate(candidates, start=1):
            verdict = (ratings.get(item_id) or {}).get(candidate["key"], {})
            buttons = "".join(
                f'<button data-rating="{rating}" class="{rating}'
                f'{" on" if verdict.get("rating") == rating else ""}">{rating}</button>'
                for rating in RATINGS)
            checked = " checked" if verdict.get("invented") else ""
            parts.append(
                f'<div class="cand" data-item="{html.escape(item_id, quote=True)}" '
                f'data-key="{html.escape(candidate["key"], quote=True)}">'
                f'<p><b>{number}.</b> {html.escape(candidate.get("text") or "(vazio)")}</p>'
                f'{buttons}<label><input type="checkbox" data-invented{checked}> '
                f'inventa algo (um nome, um facto)</label></div>')
        parts.append("</section>")
        blocks.append("".join(parts))
    return (f"<!doctype html><html lang=pt><head><meta charset=utf-8>"
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body><main>"
            f"<h1>{html.escape(title)}</h1><p>{html.escape(question)}</p>"
            f'<div class="progress">{_progress(items, ratings)}</div>'
            + "".join(blocks) + f"</main><script>{_SCRIPT}</script></body></html>")


def serve(run_dir: Path, title: str, question: str, port: int = 8765,
          files_root: Path | None = None) -> None:
    """Serve the page on 127.0.0.1 until interrupted. Ratings land in ``run_dir``."""
    run_dir = Path(run_dir)
    items = json.loads((run_dir / "items.json").read_text(encoding="utf-8"))
    root = Path(files_root or run_dir).resolve()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                page = render(items, load_ratings(run_dir), title, question)
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path.startswith("/file/"):
                target = (root / self.path[len("/file/"):]).resolve()
                if root in target.parents and target.is_file():
                    kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                    self._send(200, target.read_bytes(), kind)
                    return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/rate":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length))
                item, key, field = str(payload["item"]), str(payload["key"]), payload["field"]
            except (ValueError, KeyError):
                self._send(400, b"bad request", "text/plain")
                return
            if field not in ("rating", "invented"):
                self._send(400, b"bad field", "text/plain")
                return
            with lock:
                ratings = load_ratings(run_dir)
                ratings.setdefault(item, {}).setdefault(key, {})[field] = payload.get("value")
                temporary = run_dir / "ratings.json.tmp"
                temporary.write_text(json.dumps(ratings, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
                temporary.replace(run_dir / "ratings.json")
            body = json.dumps({"progress": _progress(items, ratings)}).encode("utf-8")
            self._send(200, body, "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Review page: http://127.0.0.1:{port}/  (Ctrl+C to stop; ratings in {run_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
