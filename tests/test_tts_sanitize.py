"""What a reply sounds like is not what it looks like.

A written reply can carry a sources line, emoji, markdown and URLs. Piper reads
all of it out loud, which is how a voice note ended with "🌐 Fontes: x.com,
play.google.com" spoken domain by domain. No test covered the string handed to
the synthesiser, only the byte count that came back.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.tts import sanitize_for_speech, split_by_language


class TestSanitizeForSpeech:
    def test_citation_line_is_dropped(self):
        text = "O Benfica ganhou 6-1.\n\n🌐 Fontes: espn.com.br, pt.uefa.com"
        assert sanitize_for_speech(text) == "O Benfica ganhou 6-1."

    def test_english_sources_label_dropped_too(self):
        text = "Benfica won 6-1.\n\nSources: espn.com"
        assert sanitize_for_speech(text) == "Benfica won 6-1."

    def test_urls_and_bare_domains_removed(self):
        spoken = sanitize_for_speech("Vê em https://record.pt e também record.pt.")
        assert "record.pt" not in spoken
        assert "https" not in spoken

    def test_markdown_emphasis_removed(self):
        assert sanitize_for_speech("escolheria **Kaya**") == "escolheria Kaya"

    def test_emoji_removed(self):
        assert sanitize_for_speech("boa 😂😂 noite") == "boa noite"

    def test_ordinary_reply_survives_untouched(self):
        text = "Dá-me uns minutos que isto demora, já mando."
        assert sanitize_for_speech(text) == text

    def test_portuguese_punctuation_and_accents_kept(self):
        text = "Não sei, mas o Gil já cá anda há anos."
        assert sanitize_for_speech(text) == text

    def test_empty_is_empty(self):
        assert sanitize_for_speech("") == ""

    def test_citation_only_reply_becomes_empty(self):
        """A reply that is nothing but sources has nothing to say out loud."""
        assert sanitize_for_speech("🌐 Fontes: espn.com") == ""

    def test_still_splits_into_language_runs_afterwards(self):
        """Sanitising must not break the PT/EN voice split downstream."""
        spoken = sanitize_for_speech(
            "Isto é português. This part is English.\n\n🌐 Fontes: bbc.com"
        )
        runs = split_by_language(spoken)
        assert runs
        assert "bbc" not in " ".join(text for _, text in runs)


def test_a_retrieval_wrapper_is_never_spoken():
    """Second layer. clean_response strips it from the written reply; this is
    what stands between a leak and Piper reading "Áudio enviado por Kaya Bot"
    out loud, which is exactly what happened."""
    from src.chat.tts import sanitize_for_speech

    spoken = sanitize_for_speech(
        "[Áudio enviado por Kaya Bot] Nem preciso de teoria para saber isso.")

    assert "Áudio enviado" not in spoken
    assert spoken.strip() == "Nem preciso de teoria para saber isso."
