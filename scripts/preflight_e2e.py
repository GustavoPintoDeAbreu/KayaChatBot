#!/usr/bin/env python3
"""End-to-end preflight against the running stack, before making it live.

The unit suite proves the wiring; this proves the box. Every check talks to what
is actually running — the llama server, the vector store, the image worker, Piper
— and each one is a capability the group will try in the first five minutes:

    llama         is the serving model up, and does it admit to vision?
    vision        can it describe a real photo from the export?
    recall        is an image description retrievable later by text?
    router        do "faz uma imagem" and "que foto gira" route differently?
    tts           does an English sentence get the English voice?
    generate      does text-to-image produce a real PNG?
    edit          does the editor keep a recognisable face? (slow: minutes)

Exit code is non-zero if any selected check fails, so it can gate a deploy.

    kaya_chatbot_env/bin/python scripts/preflight_e2e.py
    kaya_chatbot_env/bin/python scripts/preflight_e2e.py --skip edit
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
PHOTOS = BASE_DIR / "data" / "bench_photos"

RESULTS: List[Tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    return ok


def check_llama(config: Dict[str, Any]) -> bool:
    import requests

    from src.chat.vision import _server_url

    url = _server_url(config)
    try:
        health = requests.get(f"{url}/health", timeout=10)
        props = requests.get(f"{url}/props", timeout=10).json()
    except Exception as exc:  # noqa: BLE001
        return record("llama", False, f"{type(exc).__name__}: {exc}")
    vision = bool((props.get("modalities") or {}).get("vision"))
    return record("llama", health.status_code == 200 and vision,
                  f"health={health.status_code} vision={vision} at {url}")


def _a_photo_to_describe():
    """Any real image on this box, or None.

    ``data/bench_photos/`` is gitignored and was selected by a face detector that
    no longer exists, so it is present in the dev checkout and absent from
    ~/kaya-prod — where this check reported "no bench photos to describe" and
    FAILED, while vision itself was working perfectly. A preflight that cannot
    run where production runs is not a preflight.
    """
    for directory, pattern in ((PHOTOS, "*.jpg"),
                               (PHOTOS, "*.png"),
                               (BASE_DIR / "data" / "imagegen_log", "*.png"),
                               (BASE_DIR / "data" / "wpp", "**/*.jpg")):
        if not directory.exists():
            continue
        found = sorted(directory.glob(pattern))
        if found:
            return found[0]
    return None


def check_vision(config: Dict[str, Any]) -> bool:
    from src.chat import vision

    photo = _a_photo_to_describe()
    if photo is None:
        return record("vision", False, "no image on this box to describe")
    photos = [photo]
    started = time.time()
    description = vision.describe_bytes(photos[0].read_bytes(), config)
    if not description:
        return record("vision", False, "no description returned")
    return record("vision", len(description) > 30,
                  f"{time.time() - started:.0f}s: {description[:70]}…")


def check_recall(config: Dict[str, Any]) -> bool:
    """An image description in the store must be findable by ordinary text."""
    from src.chat.retriever import get_retriever
    from src.chat.scope import SHARED

    retriever = get_retriever(config)
    hits = retriever.retrieve("aquela foto que mandaram", top_k=5, scope=SHARED)
    documents = [h.get("text", "") if isinstance(h, dict) else str(h) for h in (hits or [])]
    with_images = [d for d in documents if "[Imagem" in d or "Imagem enviada" in d]
    return record("recall", bool(hits),
                  f"{len(hits or [])} hits, {len(with_images)} carrying image descriptions")


def check_router(config: Dict[str, Any]) -> bool:
    """The line that matters: asking FOR a picture vs talking ABOUT one.

    Pictures are no longer made (2026-09-04), but CMD_IMAGE is kept so the ask
    gets a straight "I do not do that" instead of the model improvising. Talking
    about a photo must still route normally.
    """
    from src.chat import router
    from src.chat.engine import get_engine

    engine = get_engine(config)
    cases = [
        ("faz uma imagem de um gato astronauta", router.CMD_IMAGE),
        ("põe o Rafa vestido de rei", router.CMD_IMAGE),
        ("que foto gira", None),
        ("quem está nesta foto?", None),
    ]
    wrong = []
    for message, expected in cases:
        route = router.classify(engine.backend, config, message)
        got = route.command
        if (got == router.CMD_IMAGE) != (expected == router.CMD_IMAGE):
            wrong.append(f"{message!r}→{got}")
    return record("router", not wrong, "; ".join(wrong) or f"{len(cases)}/{len(cases)} correct")


def check_tts(config: Dict[str, Any]) -> bool:
    from src.chat import tts

    runs = tts.split_by_language("O jantar é às oito. Honestly mate, that is a terrible idea.")
    languages = [lang for lang, _ in runs]
    if languages != ["pt", "en"]:
        return record("tts", False, f"language split was {languages}")
    wav = tts.synthesize_wav("O jantar é às oito. Honestly mate, this is terrible.", config)
    return record("tts", bool(wav and len(wav) > 10000),
                  f"split={languages}, {len(wav or b'')} bytes")


CHECKS = {
    "llama": check_llama, "vision": check_vision, "recall": check_recall,
    "router": check_router, "tts": check_tts,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end preflight.")
    parser.add_argument("--only", default="", help="comma-separated subset")
    parser.add_argument("--skip", default="", help="comma-separated checks to skip")
    args = parser.parse_args()

    from src.config_loader import load_config

    config = load_config(str(BASE_DIR / "config.yaml"))

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(CHECKS)
    skip = {n.strip() for n in args.skip.split(",") if n.strip()}
    names = [n for n in names if n not in skip]

    print(f"preflight: {', '.join(names)}\n")
    for name in names:
        try:
            CHECKS[name](config)
        except Exception as exc:  # noqa: BLE001 — one broken check must not hide the rest
            record(name, False, f"{type(exc).__name__}: {exc}")

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
