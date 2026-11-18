"""A reply-mode instruction at the end of a message (2026-11-18)."""
import pytest

from src.chat.mode_directive import AUDIO, AUDIO_ONCE, TEXT, find


@pytest.mark.parametrize("text, rest, mode", [
    ("answer in audio", "", AUDIO),
    ("Answer in audio.", "", AUDIO),
    ("answer in text", "", TEXT),
    ("Answer in text, please", "", TEXT),
    ("reply by voice", "", AUDIO),
    ("respond with a voice note", "", AUDIO),
    ("from now on answer in audio", "", AUDIO),
    ("answer in audio from now on", "", AUDIO),
    ("back to text", "", TEXT),
    ("go back to text please", "", TEXT),
    ("text only", "", TEXT),
    ("responde em áudio", "", AUDIO),
    ("Responde em texto.", "", TEXT),
    ("responde-me por voz", "", AUDIO),
    ("fala comigo por áudio", "", AUDIO),
    ("volta ao texto", "", TEXT),
    ("volta a escrever", "", TEXT),
    ("a partir de agora responde em áudio", "", AUDIO),
    ("só áudio", "", AUDIO),
    ("who's coming to dinner tonight? answer in audio", "who's coming to dinner tonight?", AUDIO),
    ("quem vem ao jantar logo? responde em áudio", "quem vem ao jantar logo?", AUDIO),
    ("o que achas do Benfica, responde em texto", "o que achas do Benfica", TEXT),
    ("explain inflation, answer in audio, just this one", "explain inflation", AUDIO_ONCE),
    ("conta-me a história do jantar, responde em áudio só desta vez", "conta-me a história do jantar", AUDIO_ONCE),
    ("Okay thanks. Answer in text, just this message, nothing else. Answer in text.", "", TEXT),
    ("ok obrigado, responde em texto", "", TEXT),
    ("Answer in text. Answer in text.", "", TEXT),
    ("thanks, who scored? answer in text", "thanks, who scored?", TEXT),
])
def test_instructions_at_the_end_are_found(text, rest, mode):
    found = find(text)
    assert found is not None and found.rest == rest and found.mode == mode


@pytest.mark.parametrize("text", [
    "o Rafa respondeu em áudio",
    "what did he say in the audio?",
    "o áudio estava mau",
    "não gosto de áudios",
    "ouvi o teu áudio ontem",
    "o Rafa manda por áudio",
    "ele fala por voz",
    "I only text",
    "manda um áudio a explicar isso",
    "answer in audio and then tell me about Gil",
    "audio",
    "",
])
def test_talk_about_audio_is_not_an_instruction(text):
    assert find(text) is None
