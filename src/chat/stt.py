"""Speech to text for incoming WhatsApp voice notes.

A voice note arrives with empty text — the words are in an audio file WAHA has
downloaded. Transcribing it turns the message into an ordinary one, so it flows
through the same path as typed text: logged as memory, routed by intent, answered
in whatever medium that chat prefers.

faster-whisper (CTranslate2) rather than openai-whisper: ~4x the throughput on
NVIDIA at int8, and measured here at ~24x realtime on real group voice notes.

The model loads lazily and once — most messages are text, and holding ~1.5GB of
VRAM for a capability that may never be used in a session is wasteful.
"""
from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()

# WhatsApp voice notes are opus; the others appear when someone forwards a file.
_EXT = {
    "audio/ogg": ".ogg", "audio/opus": ".opus", "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a", "audio/aac": ".aac", "audio/wav": ".wav",
    "audio/x-wav": ".wav", "audio/webm": ".webm",
}


def is_available(config: Dict[str, Any]) -> bool:
    acfg = (config.get("chat", {}) or {}).get("audio", {}) or {}
    if not acfg.get("transcribe_enabled", False):
        return False
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _load(config: Dict[str, Any]):
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel

                acfg = (config.get("chat", {}) or {}).get("audio", {}) or {}
                name = acfg.get("whisper_model", "large-v3")
                device = acfg.get("whisper_device", "cuda")
                compute = acfg.get("whisper_compute", "int8_float16")
                logger.info("Loading Whisper %s (%s/%s)", name, device, compute)
                _model = WhisperModel(name, device=device, compute_type=compute)
    return _model


def transcribe_file(path: str, config: Dict[str, Any]) -> Optional[str]:
    """Transcribe a local audio file. None on failure; never raises."""
    acfg = (config.get("chat", {}) or {}).get("audio", {}) or {}
    try:
        model = _load(config)
        segments, _info = model.transcribe(
            str(path),
            language=acfg.get("language", "pt"),
            vad_filter=True,   # drops the silence that ends most voice notes
            beam_size=1,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 — a failed transcript must not drop the message
        logger.warning("Transcription failed for %s: %s", path, exc)
        return None


def transcribe_url(url: str, mimetype: str, config: Dict[str, Any],
                   api_key: str = "") -> Optional[str]:
    """Fetch a voice note from WAHA and transcribe it."""
    if not url:
        return None
    try:
        import httpx

        headers = {"X-Api-Key": api_key} if api_key else {}
        with httpx.Client(timeout=60.0, headers=headers, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            audio = resp.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch voice note %s: %s", url, exc)
        return None

    suffix = _EXT.get((mimetype or "").split(";")[0].strip(), ".ogg")
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(audio)
            tmp = fh.name
        return transcribe_file(tmp, config)
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)
