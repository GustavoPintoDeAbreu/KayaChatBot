"""The group's stickers, kept and (once it is proven) understood.

A sticker arrives as ``image/webp``, exactly like a photo, so until now it took
the photo path and was logged as ``[Imagem: …]``: the frog the group sends to say
"que seca" went into memory as *"uma figura tridimensional de um sapo verde"*, a
picture somebody apparently shared. What it MEANS was never read.

Two parts, deliberately separate:

* **The archive** is always on. Every copy of the same sticker carries the same
  ``fileSha256``, so the bytes are kept once under that hash. The Pi purges media
  after 7 days; the archive is what a later "send stickers" feature would choose
  from, and what the evaluation reads.
* **Understanding** is off until ``scripts/eval_stickers.py`` shows the model
  reads the group's stickers right at least 80% of the time. Nothing about it
  goes live on the strength of the code working.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

_SAFE_SHA = re.compile(r"^[A-Za-z0-9_\-]{16,128}$")


def _settings(config: Dict[str, Any]) -> Dict[str, Any]:
    return ((config.get("chat", {}) or {}).get("stickers", {}) or {})


def archive_dir(config: Dict[str, Any]) -> Path:
    """Where sticker bytes are kept, resolved against the repo root."""
    path = Path(_settings(config).get("archive_dir", "data/stickers"))
    return path if path.is_absolute() else BASE_DIR / path


def archive_path(config: Dict[str, Any], sha: str) -> Optional[Path]:
    """The file for one sticker, or None for a hash that is not a safe filename."""
    if not _SAFE_SHA.match(sha or ""):
        return None
    return archive_dir(config) / f"{sha}.webp"


def archive(payload: bytes, sha: str, config: Dict[str, Any]) -> Optional[Path]:
    """Keep one sticker's bytes under its hash. Written once, atomically."""
    path = archive_path(config, sha)
    if path is None or not payload:
        return None
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".webp.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return path


def is_archived(config: Dict[str, Any], sha: str) -> bool:
    path = archive_path(config, sha)
    return bool(path and path.exists())


# ── understanding (behind the gate) ────────────────────────────────────────
STICKER_PROMPT = (
    "Isto é um sticker de WhatsApp que alguém mandou num grupo de amigos "
    "portugueses. Responde em português europeu, numa só linha, no formato "
    "'<o que mostra> — reação: <o que exprime>'. Em 'o que mostra' transcreve "
    "palavra por palavra qualquer texto escrito no sticker e diz o que se vê; se "
    "for uma figura pública ou um meme conhecido e tiveres a certeza, diz qual, "
    "senão não adivinhes. Em 'reação' diz a emoção ou intenção de quem o manda "
    "(por exemplo: tédio, gozo, choque, aprovação, nojo, riso). Nunca digas o "
    "nome de ninguém do grupo."
)

CONTEXT_PROMPT = (
    "\n\nMensagens do grupo imediatamente antes do sticker:\n{context}\n\n"
    "Usa-as só para perceber a reação: o que é que a pessoa quis dizer ao mandar "
    "este sticker aqui. Não descrevas as mensagens."
)


def frames(image: bytes, count: int = 3) -> list:
    """PNG stills from a sticker: first, middle and last frame of an animation.

    One frame misses the point of most animated stickers, whose punchline is at
    the end. A static sticker gives one frame. ``[]`` if it cannot be decoded.
    """
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(image)) as sticker:
            total = getattr(sticker, "n_frames", 1) or 1
            picks = sorted({0, total // 2, total - 1}) if total > 1 else [0]
            stills = []
            for index in picks[:count]:
                sticker.seek(index)
                buffer = io.BytesIO()
                sticker.convert("RGBA").save(buffer, format="PNG")
                stills.append(buffer.getvalue())
            return stills
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not decode a sticker (%s)", exc)
        return []


def describe(image: bytes, config: Dict[str, Any], context_lines: Optional[list] = None,
             prompt: str = STICKER_PROMPT, max_frames: int = 3) -> Optional[str]:
    """What a sticker shows and what it expresses, in one line. None on failure."""
    import base64

    from src.chat.inference_backend import openai_chat_fields, resolve_server_url
    from src.chat.vision import looks_like_a_refusal

    stills = frames(image, max_frames)
    if not stills:
        return None
    text = prompt
    if context_lines:
        text += CONTEXT_PROMPT.format(context="\n".join(context_lines))
    content = [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(still).decode("ascii")}}
        for still in stills]
    try:
        import requests

        response = requests.post(
            f"{resolve_server_url(config).rstrip('/')}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": content}],
                  "max_tokens": int(_settings(config).get("max_tokens", 120)),
                  "temperature": 0.2, **openai_chat_fields(config)},
            timeout=float(_settings(config).get("timeout", 120)),
        )
        response.raise_for_status()
        answer = (response.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("sticker description failed: %s", exc)
        return None
    answer = " ".join(answer.split())
    return None if looks_like_a_refusal(answer) else answer
