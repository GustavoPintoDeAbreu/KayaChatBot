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
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DESCRIBE_PROMPT = (
    "Descreve esta imagem em português europeu, em 2-3 frases. Diz o que se vê: "
    "pessoas (quantas, o que estão a fazer), o local, objectos relevantes e o "
    "ambiente. Se houver texto legível na imagem, transcreve-o. Não inventes "
    "nomes de pessoas. Responde só com a descrição."
)


def _config(config: Dict[str, Any]) -> Dict[str, Any]:
    return (config.get("chat", {}) or {}).get("vision", {}) or {}


def is_available(config: Dict[str, Any]) -> bool:
    return bool(_config(config).get("enabled", False))


def _server_url(config: Dict[str, Any]) -> str:
    gguf = ((config.get("inference", {}) or {}).get("gguf", {}) or {})
    return os.environ.get("KAYA_LLAMA_URL") or gguf.get("server_url", "http://llama:8080")


def describe_bytes(image: bytes, config: Dict[str, Any],
                   mimetype: str = "image/jpeg") -> Optional[str]:
    """Describe one image. None on any failure — never raises."""
    if not image:
        return None
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
    return content or None


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
