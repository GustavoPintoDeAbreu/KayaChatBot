"""Voice runs must not collide.

A Piper voice speaks one language, so a mixed PT/EN reply is split into runs and
each is spoken by its own voice. Those runs were concatenated frame-to-frame:
one voice stopped mid-breath and the next started on the very next sample. The
group's first reaction to a voice note was that the bot "parece que se está a
atropelar enquanto fala".
"""
import io
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import tts

RATE = 22050
RUN_SECONDS = 0.1


class FakeVoice:
    """Emits a fixed run of non-silent audio and records the synthesis config."""

    last_syn_config = "unset"

    def synthesize_wav(self, text, wav_file, syn_config=None, **kwargs):
        FakeVoice.last_syn_config = syn_config
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(RATE)
        wav_file.writeframes(b"\x11\x22" * int(RATE * RUN_SECONDS))


@pytest.fixture(autouse=True)
def _fake_piper(monkeypatch):
    monkeypatch.setattr(tts, "_load", lambda config, lang="pt": FakeVoice())
    FakeVoice.last_syn_config = "unset"


def duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def config(**audio):
    return {"chat": {"audio": audio}}


# "Olá malta, tudo bem?" | "This is English now." | "E outra vez português."
MIXED = "Olá malta, tudo bem? This is English now. E outra vez português."


def test_a_gap_is_written_between_language_runs():
    data = tts.synthesize_wav(MIXED, config(run_gap_seconds=0.18))

    # three runs of audio, two boundaries between them
    assert duration(data) == pytest.approx(3 * RUN_SECONDS + 2 * 0.18, abs=0.01)


def test_a_single_run_gets_no_leading_or_trailing_silence():
    data = tts.synthesize_wav("Olá malta tudo bem.", config(run_gap_seconds=0.18))

    assert duration(data) == pytest.approx(RUN_SECONDS, abs=0.01)


def test_the_gap_can_be_turned_off():
    data = tts.synthesize_wav("Olá malta. This is English.", config(run_gap_seconds=0))

    assert duration(data) == pytest.approx(2 * RUN_SECONDS, abs=0.01)


def test_length_scale_reaches_piper():
    tts.synthesize_wav("Olá malta tudo bem.", config(length_scale=1.15))

    assert FakeVoice.last_syn_config.length_scale == pytest.approx(1.15)


def test_no_length_scale_leaves_the_voice_default():
    tts.synthesize_wav("Olá malta tudo bem.", config())

    assert FakeVoice.last_syn_config is None


def test_empty_text_still_produces_nothing():
    assert tts.synthesize_wav("", config()) is None
    assert tts.synthesize_wav("   ", config()) is None
