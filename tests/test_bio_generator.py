"""
Unit tests for generate_bios_offline.py

Tests are fully offline — no model loading, no API calls.
All heavy dependencies (torch, unsloth) are mocked where needed.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generate_bios_offline import (
    BIO_SYSTEM_PROMPT,
    BIO_USER_TEMPLATE,
    LOCAL_MODELS,
    collect_member_context,
    is_gemma4_model,
    parse_json_response,
    build_comparison_md,
)


# ---------------------------------------------------------------------------
# is_gemma4_model
# ---------------------------------------------------------------------------

class TestIsGemma4Model:
    def test_detects_gemma4_hyphen(self):
        assert is_gemma4_model("unsloth/gemma-4-E4B-it-unsloth-bnb-4bit") is True

    def test_detects_gemma4_no_hyphen(self):
        assert is_gemma4_model("Gemma4ForConditionalGeneration") is True

    def test_rejects_qwen3(self):
        assert is_gemma4_model("unsloth/Qwen3-14B-bnb-4bit") is False

    def test_rejects_empty(self):
        assert is_gemma4_model("") is False

    def test_rejects_llama(self):
        assert is_gemma4_model("meta-llama/Llama-3-8B") is False


# ---------------------------------------------------------------------------
# collect_member_context
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES = [
    {"timestamp": "2024-01-01T10:00:00", "sender": "Peter", "content": "let's grab pizza tonight"},
    {"timestamp": "2024-01-01T10:01:00", "sender": "Gil",   "content": "sure, Peter I'm in"},
    {"timestamp": "2024-01-01T10:02:00", "sender": "Gustavo","content": "count me out today"},
    {"timestamp": "2024-01-01T10:03:00", "sender": "Peter", "content": "works at DAZN now"},
    {"timestamp": "2024-01-01T10:04:00", "sender": "Rafa",  "content": "Peter's got a dog named Kaya"},
    {"timestamp": "2024-01-01T10:05:00", "sender": "Gil",   "content": "Gustavo has the YouTube Premium link"},
]


class TestCollectMemberContext:
    def test_collects_messages_by_sender(self):
        ctx = collect_member_context(SAMPLE_MESSAGES, "Peter", max_tokens=2000)
        assert "let's grab pizza tonight" in ctx
        assert "works at DAZN now" in ctx

    def test_collects_messages_mentioning_name(self):
        ctx = collect_member_context(SAMPLE_MESSAGES, "Peter", max_tokens=2000)
        # "sure, Peter I'm in" mentions Peter
        assert "sure, Peter I'm in" in ctx

    def test_respects_token_limit(self):
        # Very small limit — should truncate
        ctx = collect_member_context(SAMPLE_MESSAGES, "Peter", max_tokens=5)
        char_limit = 5 * 4  # 20 chars
        assert len(ctx) <= char_limit + 200  # some tolerance for line lengths

    def test_returns_empty_for_unknown_member(self):
        ctx = collect_member_context(SAMPLE_MESSAGES, "Nonexistent", max_tokens=2000)
        assert ctx.strip() == ""

    def test_alias_matching(self):
        ctx = collect_member_context(
            SAMPLE_MESSAGES, "Gil", max_tokens=2000, aliases=["gilao"]
        )
        assert "sure, Peter I'm in" in ctx  # Gil sent this

    def test_includes_timestamp_and_sender(self):
        ctx = collect_member_context(SAMPLE_MESSAGES, "Peter", max_tokens=2000)
        assert "Peter" in ctx
        assert "2024-01-01" in ctx


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    VALID_BIO = {
        "name": "Peter",
        "age": "early 30s",
        "occupation": "Editor at DAZN",
        "living_place": "Paço de Arcos",
        "marital_status": "single",
        "state_of_mind": "social and humorous",
        "interests": ["football", "dogs"],
        "frequently_discussed_topics": ["social events", "food"],
        "biography_summary": "Peter is an active group member.",
        "free_text_bio": "Peter lives in Paço de Arcos. He enjoys football.",
    }

    def test_parses_clean_json(self):
        result = parse_json_response(json.dumps(self.VALID_BIO))
        assert result is not None
        assert result["name"] == "Peter"
        assert result["occupation"] == "Editor at DAZN"

    def test_strips_markdown_fences(self):
        fenced = "```json\n" + json.dumps(self.VALID_BIO) + "\n```"
        result = parse_json_response(fenced)
        assert result is not None
        assert result["name"] == "Peter"

    def test_strips_plain_fences(self):
        fenced = "```\n" + json.dumps(self.VALID_BIO) + "\n```"
        result = parse_json_response(fenced)
        assert result is not None

    def test_extracts_json_from_prose(self):
        prose = "Here is the biography:\n" + json.dumps(self.VALID_BIO) + "\nEnd."
        result = parse_json_response(prose)
        assert result is not None
        assert result["name"] == "Peter"

    def test_returns_none_for_empty(self):
        assert parse_json_response("") is None

    def test_returns_none_for_garbage(self):
        assert parse_json_response("I cannot help with that request.") is None

    def test_returns_none_for_malformed_json(self):
        assert parse_json_response('{"name": "Peter", "age":') is None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_bio_user_template_contains_name(self):
        rendered = BIO_USER_TEMPLATE.format(name="Gil", context="[context]")
        assert "Gil" in rendered
        assert "[context]" in rendered

    def test_bio_system_prompt_mentions_factual(self):
        assert "invent" in BIO_SYSTEM_PROMPT.lower() or "not invent" in BIO_SYSTEM_PROMPT.lower()

    def test_bio_system_prompt_no_self_censor_rule(self):
        # The prompt explicitly asks not to self-censor
        assert "self-censor" in BIO_SYSTEM_PROMPT.lower() or "do not self-censor" in BIO_SYSTEM_PROMPT.lower()

    def test_local_models_set(self):
        assert "gemma4" in LOCAL_MODELS
        assert "qwen3" in LOCAL_MODELS
        assert "grok" not in LOCAL_MODELS
        assert "azure" not in LOCAL_MODELS


# ---------------------------------------------------------------------------
# build_comparison_md
# ---------------------------------------------------------------------------

class TestBuildComparisonMd:
    def test_produces_markdown_with_member_sections(self, tmp_path):
        # Create mock bio JSON files
        bios = {
            "model": "grok",
            "members": {
                "Peter": {
                    "name": "Peter",
                    "age": "early 30s",
                    "occupation": "Editor",
                    "living_place": "Paço de Arcos",
                    "marital_status": "single",
                    "state_of_mind": "energetic",
                    "interests": ["football"],
                    "frequently_discussed_topics": ["events"],
                    "biography_summary": "Peter is social.",
                    "free_text_bio": "Peter enjoys football and dinners.",
                }
            },
        }
        (tmp_path / "bios_grok.json").write_text(json.dumps(bios), encoding="utf-8")

        md = build_comparison_md(tmp_path, ["Peter"])
        assert "## Peter" in md
        assert "### grok" in md
        assert "Paço de Arcos" in md

    def test_handles_error_entries(self, tmp_path):
        bios = {
            "model": "azure",
            "members": {
                "Peter": {"name": "Peter", "_error": "API quota exceeded"},
            },
        }
        (tmp_path / "bios_azure.json").write_text(json.dumps(bios), encoding="utf-8")

        md = build_comparison_md(tmp_path, ["Peter"])
        assert "API quota exceeded" in md or "Error" in md

    def test_returns_message_when_no_files(self, tmp_path):
        md = build_comparison_md(tmp_path, ["Peter"])
        assert "No bio JSON files" in md or len(md) > 0

    def test_multiple_models_listed(self, tmp_path):
        for model in ("gemma4", "qwen3"):
            bios = {
                "model": model,
                "members": {
                    "Gil": {
                        "name": "Gil",
                        "biography_summary": f"Gil bio from {model}.",
                        "free_text_bio": f"Gil free text from {model}.",
                    }
                },
            }
            (tmp_path / f"bios_{model}.json").write_text(
                json.dumps(bios), encoding="utf-8"
            )
        md = build_comparison_md(tmp_path, ["Gil"])
        assert "### gemma4" in md
        assert "### qwen3" in md
