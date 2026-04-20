"""
Pipeline Content Preservation Audit
=====================================
These tests verify that the data pipeline does NOT over-filter training data.
KayaChatBot is intentionally unrestricted (banter, dark humor, profanity are
features, not bugs), so this audit confirms that:

  1. extract_all_messages.py — is_valid_message() and clean_text() keep edgy
     content intact and only remove structural noise (URLs, media tags, etc.).

  2. format_direct_training.py — _IDENTITY_LEAK_RE only strips first-person
     identity impersonation claims, NOT banter, roasts, or profanity.

Cloud LLM Self-Censorship Investigation
-----------------------------------------
Azure OpenAI and xAI Grok do NOT expose a simple "disable safety" API flag:

  - Azure OpenAI: The `safety_identifier` param in chat.completions.create()
    is for audit/attribution tracking only (not a filter toggle). Content
    filters are configured at the Azure resource level in the Azure Portal
    (Deployments → Content Filtering). There is no SDK-level way to bypass
    them; the deployment itself must have relaxed filters.

  - xAI Grok: The GetCompletionsRequest proto has no safety filter field.
    Grok is less restricted than most models by design. No API-level toggle
    exists, but the uncensored system prompt preamble (config.yaml:
    chat.uncensored_system_prompt) can be prepended to generation prompts
    to further reduce self-censorship effects.

  - For both providers: framing synthetic-data prompts as factual extraction
    ("extract biographical facts from chat logs") rather than opinion
    generation substantially reduces self-censoring responses. The current
    get_generation_prompt() already does this for the RAG format.

Local finetuned models (Gemma4, Qwen3) fully bypass any external filter
since they run offline and the uncensored preamble is injected at runtime.
"""

import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.extract_all_messages import MessageExtractor
from src.data.format_direct_training import _IDENTITY_LEAK_RE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def extractor():
    return MessageExtractor()


# ===========================================================================
# 1. is_valid_message — must keep edgy / banter content
# ===========================================================================

class TestIsValidMessage:
    """is_valid_message() should only reject structural noise (< 3 chars with
    no semantic value), never content-based filtering."""

    def test_keeps_wtf(self, extractor):
        assert extractor.is_valid_message("wtf") is True

    def test_keeps_lmao(self, extractor):
        assert extractor.is_valid_message("lmao") is True

    def test_keeps_portuguese_profanity(self, extractor):
        assert extractor.is_valid_message("fdp") is True

    def test_keeps_roast(self, extractor):
        assert extractor.is_valid_message("és um tremendo idiota") is True

    def test_keeps_dark_humor(self, extractor):
        msg = "isso foi tão mau que eu quase morri a rir"
        assert extractor.is_valid_message(msg) is True

    def test_keeps_banter_insult(self, extractor):
        assert extractor.is_valid_message("que parvo és") is True

    def test_keeps_short_ok(self, extractor):
        # "ok" is in common_short allow-list
        assert extractor.is_valid_message("ok") is True

    def test_keeps_sim(self, extractor):
        assert extractor.is_valid_message("sim") is True

    def test_keeps_nao(self, extractor):
        assert extractor.is_valid_message("não") is True

    def test_rejects_single_char_noise(self, extractor):
        # A lone symbol with no semantic value should be filtered
        assert extractor.is_valid_message("x") is False

    def test_rejects_empty(self, extractor):
        assert extractor.is_valid_message("") is False

    def test_rejects_two_char_noise(self, extractor):
        # 2-char random noise not in allow-list should be rejected
        assert extractor.is_valid_message("zz") is False


# ===========================================================================
# 2. clean_text — must NOT strip message content
# ===========================================================================

class TestCleanText:
    """clean_text() only removes structural noise: URLs and Unicode mention
    markers. ALL actual message content must survive unchanged."""

    def test_strips_https_url(self, extractor):
        out = extractor.clean_text("olha este site https://example.com baza")
        assert "https://" not in out
        assert "olha este site" in out
        assert "baza" in out

    def test_strips_http_url(self, extractor):
        out = extractor.clean_text("http://t.co/abc123 vai ver")
        assert "http://" not in out
        assert "vai ver" in out

    def test_keeps_profanity_intact(self, extractor):
        msg = "és mesmo uma besta total"
        assert extractor.clean_text(msg) == msg

    def test_keeps_dark_humor_intact(self, extractor):
        msg = "que plano de merda, mas gostei"
        assert extractor.clean_text(msg) == msg

    def test_keeps_roast_intact(self, extractor):
        msg = "o peter não sabe sequer amarrar os sapatos ahaha"
        assert extractor.clean_text(msg) == msg

    def test_keeps_sexual_innuendo_intact(self, extractor):
        msg = "sabes mesmo mexer no pau de bilhar"
        assert extractor.clean_text(msg) == msg

    def test_strips_unicode_mention(self, extractor):
        # WhatsApp Unicode mentions format: @\u2068NAME\u2069
        msg = "olha @\u2068\u200bGustavo\u2069 tens razão"
        out = extractor.clean_text(msg)
        assert "@\u2068" not in out
        assert "\u2069" not in out
        assert "olha" in out
        assert "tens razão" in out

    def test_keeps_portuguese_slang_intact(self, extractor):
        msg = "fds que seca, vou-me embora"
        assert extractor.clean_text(msg) == msg

    def test_keeps_emoji_intact(self, extractor):
        msg = "😂😂 que palhaço"
        assert extractor.clean_text(msg) == msg


# ===========================================================================
# 3. _IDENTITY_LEAK_RE — must NOT match banter/roasts/profanity
# ===========================================================================

class TestIdentityLeakFilter:
    """_IDENTITY_LEAK_RE should ONLY match first-person identity impersonation
    claims, so the bot never learns to say it IS a group member. It must NOT
    match normal banter, roasts, or profanity."""

    # --- Strings that SHOULD be filtered (identity impersonation) -----------

    def test_matches_meu_amigo(self):
        assert _IDENTITY_LEAK_RE.search("o Peter é meu amigo desde sempre")

    def test_matches_vivemos_juntos(self):
        assert _IDENTITY_LEAK_RE.search("nós vivemos juntos em Lisboa")

    def test_matches_ja_vivemos(self):
        assert _IDENTITY_LEAK_RE.search("já vivemos naquela casa")

    def test_matches_conheco_desde(self):
        assert _IDENTITY_LEAK_RE.search("conheço-o desde criança")

    def test_matches_somos_amigos_desde(self):
        assert _IDENTITY_LEAK_RE.search("somos amigos desde 2010")

    def test_matches_nos_conhecemos_ha(self):
        assert _IDENTITY_LEAK_RE.search("nos conhecemos há dez anos")

    def test_matches_fui_com_ele(self):
        assert _IDENTITY_LEAK_RE.search("fui com ele ao jogo")

    # --- Strings that MUST NOT be filtered ---------------------------------

    def test_does_not_match_roast(self):
        assert not _IDENTITY_LEAK_RE.search("que parvo és, não sabes nada")

    def test_does_not_match_dark_humor(self):
        assert not _IDENTITY_LEAK_RE.search(
            "isso foi tão mau que quase morri a rir"
        )

    def test_does_not_match_profanity(self):
        assert not _IDENTITY_LEAK_RE.search("fdp, isso foi brutal")

    def test_does_not_match_banter_insult(self):
        assert not _IDENTITY_LEAK_RE.search("o gil é mesmo um idiota as vezes")

    def test_does_not_match_third_person_fact(self):
        # Third-person factual statement is fine — no identity leak
        assert not _IDENTITY_LEAK_RE.search(
            "o Peter parece viver em Paço de Arcos segundo as conversas"
        )

    def test_does_not_match_portuguese_slang(self):
        assert not _IDENTITY_LEAK_RE.search("baza fixe, vamo-nos embora")

    def test_does_not_match_sexual_banter(self):
        assert not _IDENTITY_LEAK_RE.search("és um tremendo gajo, a falar a sério")

    def test_does_not_match_third_person_amigo(self):
        # "amigo" in third-person context must not be caught
        assert not _IDENTITY_LEAK_RE.search(
            "o peter é amigo do gustavo há anos"
        )

    def test_does_not_match_event_reference(self):
        assert not _IDENTITY_LEAK_RE.search(
            "foram ao concerto juntos o ano passado"
        )
