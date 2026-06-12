"""Concurrency smoke test against the REAL model + RAG + GPU lock.

Imports src.chat.web_app (which loads the fine-tuned model and retriever exactly
like the live app) and fires several simultaneous `bot_stream` calls — the same
handler a browser submit triggers. Verifies that:

  * every concurrent request gets a coherent, non-empty answer,
  * GPU work is serialized (active generation windows never overlap), and
  * overflowing the lock timeout yields the friendly busy message.

Run: kaya_chatbot_env/bin/python scripts/concurrency_smoketest.py
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.chat.web_app as app  # noqa: E402  (module load triggers model load)

PROMPTS = [
    "Quem é o Gil?",
    "O que faz o grupo ao fim de semana?",
    "Conta-me sobre o Manuel.",
    "Qual é a história mais marcante do grupo?",
]


def drive(prompt, results, lock):
    """Consume bot_stream like Gradio would; record the active-generation window."""
    history = [{"role": "user", "content": prompt}]
    first_token_at = None
    last_yield_at = None
    final = ""
    for hist, _context in app.bot_stream(list(history)):
        now = time.monotonic()
        if first_token_at is None:
            first_token_at = now
        last_yield_at = now
        final = hist[-1]["content"]
    with lock:
        results.append({
            "prompt": prompt,
            "start": first_token_at,
            "end": last_yield_at,
            "len": len(final),
            "text": final,
        })


def windows_overlap(results):
    spans = sorted((r["start"], r["end"]) for r in results)
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        if next_start < prev_end - 1e-3:
            return True
    return False


def main():
    print("\n=== Concurrent requests (simulating multiple users) ===")
    results = []
    lock = threading.Lock()
    threads = [threading.Thread(target=drive, args=(p, results, lock)) for p in PROMPTS]

    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0

    results.sort(key=lambda r: r["start"])
    for r in results:
        active = r["end"] - r["start"]
        snippet = r["text"][:80].replace("\n", " ")
        print(f"\n[{r['prompt']}]  chars={r['len']}  active={active:.1f}s")
        print(f"   -> {snippet}…")

    all_answered = all(r["len"] > 0 for r in results)
    no_overlap = not windows_overlap(results)

    print("\n=== Results ===")
    print(f"wall time for {len(PROMPTS)} concurrent requests: {wall:.1f}s")
    print(f"all requests answered (non-empty): {all_answered}")
    print(f"GPU windows serialized (no overlap): {no_overlap}")

    # Overflow path: tiny timeout while the lock is held must hit the busy message.
    print("\n=== Overload / busy-message path ===")
    from src.chat.gpu_lock import gpu_section
    busy_seen = {"hit": False}
    held = threading.Event()
    release = threading.Event()

    def holder():
        with gpu_section(app.config):
            held.set()
            release.wait(timeout=10)

    h = threading.Thread(target=holder)
    h.start()
    held.wait(timeout=5)
    orig_timeout = app.config["chat"]["concurrency"]["acquire_timeout"]
    app.config["chat"]["concurrency"]["acquire_timeout"] = 0.2
    try:
        for hist, _ctx in app.bot_stream([{"role": "user", "content": "olá"}]):
            if app._BUSY_MESSAGE in hist[-1]["content"]:
                busy_seen["hit"] = True
    finally:
        app.config["chat"]["concurrency"]["acquire_timeout"] = orig_timeout
        release.set()
        h.join()
    print(f"busy message shown when GPU is occupied: {busy_seen['hit']}")

    ok = all_answered and no_overlap and busy_seen["hit"]
    print("\n" + ("✅ ALL CHECKS PASSED" if ok else "❌ CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
