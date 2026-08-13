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

# ── what a reply sounds like is not what it looks like ──────────────────────────
# A written reply can carry things that only make sense on screen. Spoken, they
# are noise: the group heard a voice note end with "🌐 Fontes: x.com,
# play.google.com" read out as two bare domains, because the citation line was
# glued onto the reply before delivery ever chose text or speech.
_URL_RE = re.compile(r"https?://\S+|\bwww\.\S+", re.IGNORECASE)
# A bare domain ("espn.com.br", "play.google.com") — Piper spells these out.
_BARE_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|pt|br|uk|io|ai|tv|es|fr|de|co)\b(?:\.[a-z]{2})?",
    re.IGNORECASE,
)
_MARKDOWN_RE = re.compile(r"\*\*|__|`+|^#{1,6}\s*|^\s*[-*+]\s+", re.MULTILINE)
# Anywhere in the line, not just at the start: by the time a reply reaches
# speech it has already been cleaned, so anything left is a leak worth removing
# wherever it sits rather than a deliberate aside.
_SCAFFOLD_RE = re.compile(
    r"\[\s*(?:(?:á|a)udio|imagem|foto|v[íi]deo|a\s+responder\s+a)\b[^\]]*\]",
    re.IGNORECASE,
)
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U0000FE0F\U000020E3\U00002B00-\U00002BFF"
    "]+",
    flags=re.UNICODE,
)


def sanitize_for_speech(text: str, citation_prefix: str = "🌐 Fontes:") -> str:
    """Strip everything that reads fine but does not *speak* fine.

    Drops the sources line entirely (a whole line starting with the citation
    marker), then emoji, markdown syntax, URLs and bare domains. Applied at the
    single point where a reply becomes audio, so every surface gets it.
    """
    if not text:
        return ""
    kept = [
        line for line in text.split("\n")
        if not line.strip().startswith(citation_prefix)
        and not line.strip().lstrip("🌐 ").lower().startswith(("fontes:", "sources:"))
    ]
    out = "\n".join(kept)
    # A retrieval wrapper the model copied out of its context — "[Áudio enviado
    # por Kaya Bot]", "[Imagem enviada por X em DATA]". clean_response strips it
    # from the written reply; this is the second layer, because the one time it
    # got through, Piper read it aloud and the group filed a bug about the audio
    # being "badly formatted".
    out = _SCAFFOLD_RE.sub(" ", out)
    out = _URL_RE.sub(" ", out)
    out = _BARE_DOMAIN_RE.sub(" ", out)
    out = _MARKDOWN_RE.sub("", out)
    out = _EMOJI_RE.sub(" ", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


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


def _synthesis_config(config: Dict[str, Any]):
    """Piper's per-request knobs, or None when nothing is configured.

    ``length_scale`` is speaking rate: >1 slower, <1 faster. Exposed because the
    group's first reaction to a voice note was that the bot "parece que se está a
    atropelar enquanto fala", and pace should be tunable without a code change.
    """
    acfg = (config.get("chat", {}) or {}).get("audio", {}) or {}
    length_scale = acfg.get("length_scale")
    if length_scale is None:
        return None
    try:
        from piper import SynthesisConfig

        return SynthesisConfig(length_scale=float(length_scale))
    except Exception as exc:  # noqa: BLE001 — an unusable knob must not lose the audio
        logger.warning("could not apply length_scale (%s); using the voice default", exc)
        return None


def _silence(params, seconds: float) -> bytes:
    """A gap of digital silence in the same format as the surrounding audio."""
    frames = int(params.framerate * max(0.0, seconds))
    return b"\x00" * frames * params.nchannels * params.sampwidth


def synthesize_wav(text: str, config: Dict[str, Any]) -> Optional[bytes]:
    """Text -> WAV bytes, each sentence spoken by its own language's voice."""
    if not text or not text.strip():
        return None
    try:
        runs = split_by_language(text)
        if not runs:
            return None

        syn_config = _synthesis_config(config)
        acfg = (config.get("chat", {}) or {}).get("audio", {}) or {}
        gap = float(acfg.get("run_gap_seconds", 0.18))

        frames, params = [], None
        for lang, chunk in runs:
            voice = _load(config, lang)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                voice.synthesize_wav(chunk, wf, syn_config=syn_config)
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
            # A gap at every run boundary. The runs were being written
            # frame-to-frame, so a Portuguese sentence and the English one after
            # it collided with no pause at all — one voice stops mid-breath and
            # another starts on the next sample, which is what "parece que se
            # está a atropelar" describes.
            pause = _silence(params, gap)
            for index, frame in enumerate(frames):
                if index and pause:
                    wf.writeframes(pause)
                wf.writeframes(frame)
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
