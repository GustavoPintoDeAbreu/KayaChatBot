"""Text to speech for WhatsApp voice replies.

Piper (`pt_PT-tugão-medium`) synthesises European Portuguese locally — ~28x
realtime on CPU, so it never competes with the GPU that is answering. Kokoro, the
usual 2026 default, only ships Brazilian Portuguese and would sound wrong to this
group.

Piper emits WAV; WhatsApp voice notes must be OGG/Opus. The conversion goes
through PyAV (already present as a faster-whisper dependency) rather than
shelling out to ffmpeg, which is not installed on this box and would need root.

Loading is lazy and cached: the voice is ~61MB and most replies are text.
"""
from __future__ import annotations

import io
import logging
import threading
import wave
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "models/piper/pt/pt_PT/tugão/medium/pt_PT-tugão-medium.onnx"

_voice = None
_voice_lock = threading.Lock()


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).parent.parent.parent / path
    return p


def is_available(config: Dict[str, Any]) -> bool:
    """Whether voice replies can actually be produced right now.

    Checked before the bot ever offers to speak: confirming "passo a responder por
    áudio" and then silently answering in text is worse than declining.
    """
    acfg = (config.get("chat", {}) or {}).get("audio", {}) or {}
    if not acfg.get("reply_enabled", False):
        return False
    return _resolve(acfg.get("voice_model", DEFAULT_VOICE)).exists()


def _load(config: Dict[str, Any]):
    global _voice
    if _voice is None:
        with _voice_lock:
            if _voice is None:
                from piper import PiperVoice

                acfg = (config.get("chat", {}) or {}).get("audio", {}) or {}
                model = _resolve(acfg.get("voice_model", DEFAULT_VOICE))
                logger.info("Loading Piper voice from %s", model)
                _voice = PiperVoice.load(str(model))
    return _voice


def synthesize_wav(text: str, config: Dict[str, Any]) -> Optional[bytes]:
    """Text -> WAV bytes. None on failure; never raises."""
    if not text or not text.strip():
        return None
    try:
        voice = _load(config)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(text, wf)
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001 — a failed voice reply must fall back to text
        logger.warning("TTS failed: %s", exc)
        return None


def wav_to_ogg_opus(wav_bytes: bytes) -> Optional[bytes]:
    """WAV -> OGG/Opus, the format WhatsApp expects for a voice note."""
    try:
        import av

        inp = av.open(io.BytesIO(wav_bytes), "r")
        out_buf = io.BytesIO()
        out = av.open(out_buf, "w", format="ogg")
        # 24kHz mono is what voice notes use; Opus resamples from Piper's 22.05kHz.
        stream = out.add_stream("libopus", rate=24000)
        stream.layout = "mono"

        resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
        for frame in inp.decode(audio=0):
            for resampled in resampler.resample(frame):
                for packet in stream.encode(resampled):
                    out.mux(packet)
        for packet in stream.encode(None):   # flush
            out.mux(packet)
        out.close()
        inp.close()
        return out_buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.warning("WAV->Opus conversion failed: %s", exc)
        return None


def synthesize_voice_note(text: str, config: Dict[str, Any]) -> Optional[bytes]:
    """Text -> OGG/Opus bytes ready to send as a WhatsApp voice note."""
    wav = synthesize_wav(text, config)
    if not wav:
        return None
    return wav_to_ogg_opus(wav)
