"""PyAV 19 removed av.open(metadata_errors=...), which faster-whisper's own decoder passes;
src.chat.stt.decode_audio does the decode itself (2026-11-23)."""
import io
import wave

import numpy as np

from src.chat.stt import decode_audio
from src.chat.tts import wav_to_ogg_opus


def _wav(seconds=1.0, rate=22050):
    t = np.arange(int(seconds * rate)) / rate
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.sin(2 * np.pi * 440 * t) * 16000).astype("<i2").tobytes())
    return buf.getvalue()


def test_decode_audio_gives_whisper_16k_mono_float(tmp_path):
    note = tmp_path / "note.ogg"
    note.write_bytes(wav_to_ogg_opus(_wav()))
    audio = decode_audio(str(note))
    assert audio.dtype == np.float32 and abs(len(audio) - 16000) < 1600
    assert 0.3 < float(np.abs(audio).max()) <= 1.0
