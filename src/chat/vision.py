"""Reading the photos people send.

Same shape as `stt.py`, and for the same reason: an inbound photo carries no text,
so without this step the message is either dropped or answered as though nothing
were attached — "não recebi nenhuma imagem" while the picture sits right there in
the chat.

Describing it turns the photo into an ordinary message. From that point it flows
through everything else unchanged: it is logged as memory, ingested into the
vector store, routed by intent, and answerable a week later ("aquela foto do
barco"). The description is written in the third person and in Portuguese because
that is what the rest of the group's memory looks like.

The describer is the model already serving prod — the same gemma-4-12b, which is
multimodal, with `--mmproj` loaded on the llama service. No second model, ~180MB
of extra VRAM.
"""
from __future__ import annotations

import base64
import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DESCRIBE_PROMPT = (
    "Descreve esta imagem em português europeu, em 2-3 frases. Diz o que se vê: "
    "pessoas (quantas, o que estão a fazer), o local, objectos relevantes e o "
    "ambiente. Se houver texto legível na imagem, transcreve-o. Não inventes "
    "nomes de pessoas. Responde só com a descrição."
)


# The describer answering that it cannot see anything. This is a successful HTTP
# 200 carrying a refusal, so nothing downstream treated it as a failure: the group
# log ended up holding "[Imagem: Por favor, fornece o vídeo ou a imagem para que
# eu possa descrevê-la]" twice, as something a member had apparently said — and
# from there it is ingested into the vector store as group memory.
#
# Animated WebP stickers are the usual cause: the mimetype is image/*, so the
# photo path takes them, and the projector cannot decode them.
_NO_IMAGE = re.compile(
    r"forne[çc]e|fornecer|partilha o ficheiro|envia (a imagem|o v[íi]deo)"
    r"|n[ãa]o (consigo|posso) (ver|descrever|analisar)"
    r"|sem (o ficheiro|a imagem|acesso)"
    r"|(no|without an?) image (was )?(provided|attached)"
    r"|i (cannot|can't) (see|view|describe)",
    re.IGNORECASE,
)

# A real description of a photo runs to a couple of sentences. Anything this
# short is the model shrugging, not describing.
_MIN_DESCRIPTION_CHARS = 25


def looks_like_a_refusal(text: str) -> bool:
    """Whether the describer said it could not see the image rather than describing it."""
    stripped = (text or "").strip()
    return len(stripped) < _MIN_DESCRIPTION_CHARS or bool(_NO_IMAGE.search(stripped))


def _config(config: Dict[str, Any]) -> Dict[str, Any]:
    return (config.get("chat", {}) or {}).get("vision", {}) or {}


def is_available(config: Dict[str, Any]) -> bool:
    return bool(_config(config).get("enabled", False))


def _server_url(config: Dict[str, Any]) -> str:
    gguf = ((config.get("inference", {}) or {}).get("gguf", {}) or {})
    return os.environ.get("KAYA_LLAMA_URL") or gguf.get("server_url", "http://llama:8080")


def flatten_animation(image: bytes, mimetype: str) -> tuple:
    """An animated sticker as a single still frame. ``(bytes, mimetype)``.

    WhatsApp stickers arrive as ``image/webp`` and are often animated, which the
    projector cannot decode — so the model was handed something it could not see
    and answered that it had received no image. One frame is all a description
    needs, and a static sticker passes through untouched.

    Returns the input unchanged if anything goes wrong: the describer failing is
    better than the message being dropped.
    """
    if "webp" not in (mimetype or "").lower():
        return image, mimetype
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(image)) as sticker:
            if not getattr(sticker, "is_animated", False):
                return image, mimetype
            sticker.seek(0)
            buffer = io.BytesIO()
            sticker.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue(), "image/png"
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not flatten an animated sticker (%s)", exc)
        return image, mimetype


def describe_bytes(image: bytes, config: Dict[str, Any],
                   mimetype: str = "image/jpeg") -> Optional[str]:
    """Describe one image. None on any failure — never raises."""
    if not image:
        return None
    image, mimetype = flatten_animation(image, mimetype)
    vcfg = _config(config)
    try:
        import requests

        response = requests.post(
            f"{_server_url(config).rstrip('/')}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": vcfg.get("prompt", DESCRIBE_PROMPT)},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mimetype};base64,"
                               f"{base64.b64encode(image).decode('ascii')}"}},
                ]}],
                "max_tokens": int(vcfg.get("max_tokens", 160)),
                "temperature": 0.2,
                # Without this Gemma-4 emits its thinking channel and the answer
                # lands in reasoning_content with content left empty.
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=float(vcfg.get("timeout", 120)),
        )
        response.raise_for_status()
        content = (response.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as exc:  # noqa: BLE001 — a failed description must not drop the message
        logger.warning("image description failed: %s", exc)
        return None
    if looks_like_a_refusal(content):
        # None, not the text: the caller writes whatever comes back into the
        # message log as "[Imagem: …]", and a refusal stored there becomes a
        # searchable thing the group said.
        logger.warning("describer could not see the image: %r", content[:120])
        return None
    return content


def describe_url(url: str, mimetype: str, config: Dict[str, Any],
                 api_key: str = "", waha_base_url: str = "") -> Optional[str]:
    """Fetch a photo from WAHA and describe it."""
    if not url:
        return None

    from src.chat.stt import rewrite_media_url

    url = rewrite_media_url(url, waha_base_url)
    try:
        import httpx

        headers = {"X-Api-Key": api_key} if api_key else {}
        with httpx.Client(timeout=60.0, headers=headers, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            payload = response.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not fetch image %s: %s", url, exc)
        return None

    return describe_bytes(payload, config, mimetype or "image/jpeg")
