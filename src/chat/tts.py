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
import re
import threading
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Piper voices are single-language: the pt_PT model applies Portuguese phonetics
# to whatever it is given, so English text read by it does not sound like English.
# One voice per language, chosen per sentence.
DEFAULT_VOICES = {
    "pt": "models/piper/pt/pt_PT/tugão/medium/pt_PT-tugão-medium.onnx",
    "en": "models/piper/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
}
DEFAULT_VOICE = DEFAULT_VOICES["pt"]

_voices: Dict[str, Any] = {}
_voice_lock = threading.Lock()

# Sentence boundaries, keeping the terminator with its sentence.
_SENTENCE = re.compile(r"[^.!?…\n]+[.!?…]*\s*", re.UNICODE)


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).parent.parent.parent / path
    return p


def _voice_paths(config: Dict[str, Any]) -> Dict[str, str]:
    acfg = (config.get("chat", {}) or {}).get("audio", {}) or {}
    voices = dict(DEFAULT_VOICES)
    voices.update(acfg.get("voices", {}) or {})
    # Back-compat with the single-voice setting.
    if acfg.get("voice_model"):
        voices["pt"] = acfg["voice_model"]
    return voices


def is_available(config: Dict[str, Any]) -> bool:
    """Whether voice replies can actually be produced right now.

    Checked before the bot ever offers to speak: confirming "passo a responder por
    áudio" and then silently answering in text is worse than declining. Only the
    Portuguese voice is required — English falls back to it if missing.
    """
    acfg = (config.get("chat", {}) or {}).get("audio", {}) or {}
    if not acfg.get("reply_enabled", False):
        return False
    return _resolve(_voice_paths(config)["pt"]).exists()


def split_by_language(text: str) -> List[Tuple[str, str]]:
    """Group the reply into consecutive same-language runs of sentences.

    Sentence granularity rather than word: this group code-switches by clause
    ("bora ao meeting" stays one sentence), and swapping voice mid-sentence sounds
    far worse than one language colouring a stray loanword.

    A sentence carrying no language marker ("Absolutely brutal.") inherits the
    sentence before it, falling back to the whole reply's language — guessing
    Portuguese per sentence would break exactly the English replies this exists
    to fix.
    """
    from src.chat.response_utils import language_signal

    sentences = [m.group(0) for m in _SENTENCE.finditer(text or "") if m.group(0).strip()]
    if not sentences:
        return [("pt", text)] if text and text.strip() else []

    overall = language_signal(text) or "pt"
    runs: List[Tuple[str, str]] = []
    for sentence in sentences:
        lang = language_signal(sentence) or (runs[-1][0] if runs else overall)
        if runs and runs[-1][0] == lang:
            runs[-1] = (lang, runs[-1][1] + sentence)
        else:
            runs.append((lang, sentence))
    return runs


def _load(config: Dict[str, Any], lang: str = "pt"):
    """Load (and cache) the voice for one language, falling back to Portuguese."""
    paths = _voice_paths(config)
    path = _resolve(paths.get(lang, paths["pt"]))
    if not path.exists():
        path = _resolve(paths["pt"])
        lang = "pt"
    key = str(path)
    if key not in _voices:
        with _voice_lock:
            if key not in _voices:
                from piper import PiperVoice

                logger.info("Loading Piper voice (%s) from %s", lang, path)
                _voices[key] = PiperVoice.load(str(path))
    return _voices[key]


def synthesize_wav(text: str, config: Dict[str, Any]) -> Optional[bytes]:
    """Text -> WAV bytes, each sentence spoken by its own language's voice."""
    if not text or not text.strip():
        return None
    try:
        runs = split_by_language(text)
        if not runs:
            return None

        frames, params = [], None
        for lang, chunk in runs:
            voice = _load(config, lang)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                voice.synthesize_wav(chunk, wf)
            buf.seek(0)
            with wave.open(buf, "rb") as wf:
                if params is None:
                    params = wf.getparams()
                elif wf.getframerate() != params.framerate:
                    # Voices should share 22.05kHz mono; if one differs, skipping
                    # it is better than emitting a chipmunk.
                    logger.warning("Voice sample-rate mismatch for %s; skipping run", lang)
                    continue
                frames.append(wf.readframes(wf.getnframes()))

        if not frames or params is None:
            return None

        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setparams(params)
            for f in frames:
                wf.writeframes(f)
        return out.getvalue()
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
