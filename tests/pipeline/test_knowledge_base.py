"""
Tests for enhanced knowledge base generation (structured member profiles).

These tests validate the helper functions in generate_knowledge_base.py and
the new Pydantic models in src/models.py without requiring any API keys or
external services.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Model imports
# ---------------------------------------------------------------------------
from models import MemberProfile, TopicMapping  # noqa: E402

# ---------------------------------------------------------------------------
# generate_knowledge_base helpers
# ---------------------------------------------------------------------------
from data.generate_knowledge_base import (  # noqa: E402
    chunk_messages,
    format_chunk_for_prompt,
    get_mentioned_members,
    build_extraction_prompt,
    merge_profiles,
    merge_list_field,
    strip_markdown_fences,
    save_group_members,
    save_group_knowledge,
    get_profile_fields,
    DEFAULT_PROFILE_FIELDS,
    SENSITIVE_FIELDS,
)


# ===========================================================================
# Fixtures
# ===========================================================================

SAMPLE_MESSAGES = [
    {"timestamp": "2023-06-01T10:00:00", "sender": "Peter", "text": "Hello everyone!"},
    {"timestamp": "2023-06-01T10:01:00", "sender": "Gil", "text": "Hey Peter, how are you?"},
    {"timestamp": "2023-06-01T10:02:00", "sender": "Gustavo", "text": "What's up guys?"},
    {"timestamp": "2023-06-01T10:03:00", "sender": "Peter", "text": "All good, heading to the gym."},
    {"timestamp": "2023-06-01T10:04:00", "sender": "Gil", "text": "Nice! See you later."},
]

MEMBER_ALIASES = {
    "Peter": ["peter"],
    "Gil": ["gil", "gilao"],
    "Gustavo": ["gustavo"],
}

SAMPLE_MEMBERS_DATA = {
    "group_name": "Kaya",
    "members": [
        {"name": "Peter", "aliases": ["peter"], "notes": ""},
        {"name": "Gil", "aliases": ["gil", "gilao"], "notes": ""},
        {"name": "Gustavo", "aliases": ["gustavo"], "notes": ""},
    ],
}

SAMPLE_KNOWLEDGE_DATA = {
    "description": "Test knowledge base",
    "facts": [
        {
            "id": "member_peter",
            "category": "member",
            "subject": "Peter",
            "text": "",
        },
        {
            "id": "member_gil",
            "category": "member",
            "subject": "Gil",
            "text": "",
        },
    ],
}


# ===========================================================================
# MemberProfile Pydantic model tests
# ===========================================================================

class TestMemberProfile:
    def test_minimal_profile(self):
        profile = MemberProfile(name="Peter")
        assert profile.name == "Peter"
        assert profile.age is None
        assert profile.interests is None
        assert profile.political_preference is None

    def test_full_profile(self):
        profile = MemberProfile(
            name="Peter",
            age="late 20s",
            interests=["music", "gym", "Marvel"],
            occupation="software engineer",
            living_place="Lisbon",
            marital_status="single",
            political_preference="social democrat",
            state_of_mind="positive and energetic",
            biography_summary="Peter is a software engineer from Lisbon.",
            frequently_discussed_topics=["music", "tech", "sports"],
            notes="Peter is a software engineer from Lisbon.",
        )
        assert profile.name == "Peter"
        assert "music" in profile.interests
        assert profile.political_preference == "social democrat"

    def test_to_dict_excludes_none(self):
        profile = MemberProfile(name="Peter", age="late 20s")
        d = profile.to_dict()
        assert "name" in d
        assert "age" in d
        assert "occupation" not in d  # None fields excluded

    def test_to_public_dict_excludes_sensitive(self):
        profile = MemberProfile(
            name="Peter",
            political_preference="social democrat",
            biography_summary="Peter is a software engineer.",
        )
        pub = profile.to_public_dict()
        assert "political_preference" not in pub
        assert "biography_summary" in pub
        assert "name" in pub

    def test_sensitive_fields_constant(self):
        """SENSITIVE_FIELDS must contain political_preference."""
        assert "political_preference" in SENSITIVE_FIELDS


class TestTopicMapping:
    def test_creation(self):
        tm = TopicMapping(
            id="topics_peter",
            subject="Peter",
            text="Peter frequently discusses: music, tech, sports.",
        )
        assert tm.id == "topics_peter"
        assert tm.category == "topic_mapping"
        assert tm.subject == "Peter"

    def test_to_dict(self):
        tm = TopicMapping(
            id="topics_peter",
            subject="Peter",
            text="Peter frequently discusses: music.",
        )
        d = tm.to_dict()
        assert d["id"] == "topics_peter"
        assert d["category"] == "topic_mapping"


# ===========================================================================
# chunk_messages tests
# ===========================================================================

class TestChunkMessages:
    def test_single_chunk_small_messages(self):
        messages = [{"text": "hello world"} for _ in range(5)]
        chunks = chunk_messages(messages, chunk_size_words=100)
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_splits_into_multiple_chunks(self):
        # Each message is 10 words; chunk_size_words=25 → new chunk after 2 msgs
        messages = [{"text": " ".join([f"word{j}" for j in range(10)])} for _ in range(6)]
        chunks = chunk_messages(messages, chunk_size_words=25)
        assert len(chunks) >= 2

    def test_empty_messages(self):
        assert chunk_messages([], chunk_size_words=100) == []

    def test_respects_chunk_size(self):
        messages = [{"text": " ".join(["x"] * 100)} for _ in range(10)]
        chunks = chunk_messages(messages, chunk_size_words=150)
        for chunk in chunks:
            total_words = sum(len(m["text"].split()) for m in chunk)
            # Each chunk should not vastly exceed the limit (one overflow message allowed)
            assert total_words <= 250


# ===========================================================================
# format_chunk_for_prompt tests
# ===========================================================================

class TestFormatChunkForPrompt:
    def test_basic_format(self):
        msgs = [
            {"timestamp": "2023-06-01T10:00:00", "sender": "Peter", "text": "Hello"},
            {"timestamp": "2023-06-01T10:01:00", "sender": "Gil", "text": "Hi"},
        ]
        result = format_chunk_for_prompt(msgs)
        assert "Peter" in result
        assert "Gil" in result
        assert "Hello" in result
        assert "2023-06-01T10:00" in result

    def test_skips_empty_text(self):
        msgs = [
            {"timestamp": "2023-06-01T10:00:00", "sender": "Peter", "text": ""},
            {"timestamp": "2023-06-01T10:01:00", "sender": "Gil", "text": "Hi"},
        ]
        result = format_chunk_for_prompt(msgs)
        assert "Peter" not in result
        assert "Gil: Hi" in result


# ===========================================================================
# get_mentioned_members tests
# ===========================================================================

class TestGetMentionedMembers:
    def test_detects_senders(self):
        mentioned = get_mentioned_members(SAMPLE_MESSAGES, MEMBER_ALIASES)
        assert "Peter" in mentioned
        assert "Gil" in mentioned
        assert "Gustavo" in mentioned

    def test_detects_alias_mentions(self):
        msgs = [{"sender": "Other", "text": "gilao is coming tonight"}]
        mentioned = get_mentioned_members(msgs, MEMBER_ALIASES)
        assert "Gil" in mentioned

    def test_empty_messages(self):
        assert get_mentioned_members([], MEMBER_ALIASES) == []

    def test_returns_sorted(self):
        mentioned = get_mentioned_members(SAMPLE_MESSAGES, MEMBER_ALIASES)
        assert mentioned == sorted(mentioned)


# ===========================================================================
# build_extraction_prompt tests
# ===========================================================================

class TestBuildExtractionPrompt:
    def test_includes_member_names(self):
        profiles = {"Peter": {"biography_summary": "Peter is a dev."}, "Gil": {}}
        prompt = build_extraction_prompt("chunk text", profiles, ["Peter", "Gil"], DEFAULT_PROFILE_FIELDS)
        assert "Peter" in prompt
        assert "Gil" in prompt

    def test_includes_chunk_text(self):
        profiles = {"Peter": {}}
        prompt = build_extraction_prompt("unique_chunk_content_xyz", profiles, ["Peter"], DEFAULT_PROFILE_FIELDS)
        assert "unique_chunk_content_xyz" in prompt

    def test_includes_existing_bio(self):
        profiles = {"Peter": {"biography_summary": "Peter is a drummer."}}
        prompt = build_extraction_prompt("...", profiles, ["Peter"], DEFAULT_PROFILE_FIELDS)
        assert "Peter is a drummer." in prompt

    def test_subset_fields_note(self):
        subset_fields = ["biography_summary", "interests"]
        prompt = build_extraction_prompt("...", {"Peter": {}}, ["Peter"], subset_fields)
        assert "biography_summary" in prompt or "interests" in prompt


# ===========================================================================
# merge_list_field tests
# ===========================================================================

class TestMergeListField:
    def test_none_existing(self):
        assert merge_list_field(None, ["a", "b"]) == ["a", "b"]

    def test_none_new(self):
        assert merge_list_field(["a"], None) == ["a"]

    def test_deduplication(self):
        merged = merge_list_field(["a", "b"], ["b", "c"])
        assert merged == ["a", "b", "c"]

    def test_order_preserved(self):
        merged = merge_list_field(["x", "y"], ["z"])
        assert merged[0] == "x"
        assert merged[-1] == "z"


# ===========================================================================
# merge_profiles tests
# ===========================================================================

class TestMergeProfiles:
    def test_empty_existing_profile(self):
        new_data = {
            "age": "late 20s",
            "interests": ["music"],
            "biography_summary": "Peter is a developer.",
        }
        merged = merge_profiles({}, new_data)
        assert merged["age"] == "late 20s"
        assert "music" in merged["interests"]
        assert "Peter is a developer." in merged["biography_summary"]
        assert merged["notes"] == merged["biography_summary"]  # legacy alias

    def test_merges_list_fields(self):
        existing = {"interests": ["music", "gym"]}
        new_data = {"interests": ["gym", "travel"]}
        merged = merge_profiles(existing, new_data)
        assert set(merged["interests"]) == {"music", "gym", "travel"}

    def test_appends_new_bio_sentences(self):
        existing = {"biography_summary": "Peter is a developer."}
        new_data = {"biography_summary": "He lives in Lisbon."}
        merged = merge_profiles(existing, new_data)
        assert "developer" in merged["biography_summary"]
        assert "Lisbon" in merged["biography_summary"]
        assert merged["biography_summary"].endswith(".")

    def test_no_duplicate_bio_sentences(self):
        bio = "Peter is a developer."
        existing = {"biography_summary": bio}
        new_data = {"biography_summary": bio}  # Same sentence
        merged = merge_profiles(existing, new_data)
        # Should not duplicate the sentence
        assert merged["biography_summary"].count("developer") == 1

    def test_null_new_values_ignored(self):
        existing = {"age": "late 20s"}
        new_data = {"age": None, "occupation": None}
        merged = merge_profiles(existing, new_data)
        assert merged["age"] == "late 20s"
        assert "occupation" not in merged

    def test_scalar_field_overwritten(self):
        existing = {"living_place": "Porto"}
        new_data = {"living_place": "Lisbon"}
        merged = merge_profiles(existing, new_data)
        assert merged["living_place"] == "Lisbon"

    def test_notes_synced_with_biography_summary(self):
        existing = {}
        new_data = {"biography_summary": "Peter is a developer."}
        merged = merge_profiles(existing, new_data)
        assert merged["notes"] == merged["biography_summary"]


# ===========================================================================
# strip_markdown_fences tests
# ===========================================================================

class TestStripMarkdownFences:
    def test_strips_fences(self):
        content = "```json\n{\"members\": {}}\n```"
        result = strip_markdown_fences(content)
        assert "```" not in result
        assert "{" in result

    def test_no_fences_unchanged(self):
        content = '{"members": {}}'
        assert strip_markdown_fences(content) == content


# ===========================================================================
# get_profile_fields tests
# ===========================================================================

class TestGetProfileFields:
    def test_returns_config_fields(self):
        config = {"knowledge_base": {"profile_fields": ["name", "age", "interests"]}}
        fields = get_profile_fields(config)
        assert fields == ["name", "age", "interests"]

    def test_returns_defaults_when_missing(self):
        config = {}
        fields = get_profile_fields(config)
        assert fields == DEFAULT_PROFILE_FIELDS

    def test_all_default_fields_present(self):
        expected_fields = [
            "name", "age", "interests", "occupation", "living_place",
            "marital_status", "political_preference", "state_of_mind",
            "biography_summary", "frequently_discussed_topics",
        ]
        for field in expected_fields:
            assert field in DEFAULT_PROFILE_FIELDS, (
                f"'{field}' missing from DEFAULT_PROFILE_FIELDS"
            )


# ===========================================================================
# save_group_members tests
# ===========================================================================

class TestSaveGroupMembers:
    def test_writes_profile_fields(self, tmp_path):
        members_data = {
            "group_name": "Kaya",
            "members": [
                {"name": "Peter", "aliases": ["peter"], "notes": ""},
            ],
        }
        profiles = {
            "Peter": {
                "age": "late 20s",
                "interests": ["music"],
                "biography_summary": "Peter is a dev.",
                "notes": "Peter is a dev.",
            }
        }
        out_file = tmp_path / "group_members.json"
        # Temporarily redirect the constant
        import data.generate_knowledge_base as kb_module
        original = kb_module.GROUP_MEMBERS_FILE
        kb_module.GROUP_MEMBERS_FILE = out_file
        try:
            save_group_members(members_data, profiles, DEFAULT_PROFILE_FIELDS)
            with open(out_file, "r") as f:
                saved = json.load(f)
            peter = saved["members"][0]
            assert peter["age"] == "late 20s"
            assert peter["interests"] == ["music"]
            assert peter["notes"] == "Peter is a dev."
        finally:
            kb_module.GROUP_MEMBERS_FILE = original

    def test_sensitive_field_stays_local(self, tmp_path):
        """political_preference is written to group_members.json (local storage)."""
        members_data = {
            "group_name": "Kaya",
            "members": [{"name": "Peter", "aliases": ["peter"], "notes": ""}],
        }
        profiles = {
            "Peter": {
                "political_preference": "social democrat",
                "biography_summary": "Peter is a dev.",
            }
        }
        out_file = tmp_path / "group_members.json"
        import data.generate_knowledge_base as kb_module
        original = kb_module.GROUP_MEMBERS_FILE
        kb_module.GROUP_MEMBERS_FILE = out_file
        try:
            save_group_members(members_data, profiles, DEFAULT_PROFILE_FIELDS)
            with open(out_file, "r") as f:
                saved = json.load(f)
            peter = saved["members"][0]
            # Sensitive field is stored locally
            assert peter.get("political_preference") == "social democrat"
        finally:
            kb_module.GROUP_MEMBERS_FILE = original


# ===========================================================================
# save_group_knowledge tests
# ===========================================================================

class TestSaveGroupKnowledge:
    def test_updates_member_bio(self, tmp_path):
        knowledge_data = {
            "description": "test",
            "facts": [
                {"id": "member_peter", "category": "member", "subject": "Peter", "text": "old bio"},
            ],
        }
        profiles = {"Peter": {"biography_summary": "new bio"}}
        out_file = tmp_path / "group_knowledge.json"
        import data.generate_knowledge_base as kb_module
        original = kb_module.GROUP_KNOWLEDGE_FILE
        kb_module.GROUP_KNOWLEDGE_FILE = out_file
        try:
            save_group_knowledge(knowledge_data, profiles)
            with open(out_file, "r") as f:
                saved = json.load(f)
            peter_fact = next(f for f in saved["facts"] if f["id"] == "member_peter")
            assert peter_fact["text"] == "new bio"
        finally:
            kb_module.GROUP_KNOWLEDGE_FILE = original

    def test_adds_topic_mapping_facts(self, tmp_path):
        knowledge_data = {
            "description": "test",
            "facts": [],
        }
        profiles = {
            "Peter": {
                "frequently_discussed_topics": ["music", "tech", "sports"],
                "biography_summary": "Peter is a dev.",
            }
        }
        out_file = tmp_path / "group_knowledge.json"
        import data.generate_knowledge_base as kb_module
        original = kb_module.GROUP_KNOWLEDGE_FILE
        kb_module.GROUP_KNOWLEDGE_FILE = out_file
        try:
            save_group_knowledge(knowledge_data, profiles)
            with open(out_file, "r") as f:
                saved = json.load(f)
            topic_facts = [f for f in saved["facts"] if f["category"] == "topic_mapping"]
            assert len(topic_facts) == 1
            assert topic_facts[0]["subject"] == "Peter"
            assert "music" in topic_facts[0]["text"]
        finally:
            kb_module.GROUP_KNOWLEDGE_FILE = original

    def test_topic_mapping_excludes_sensitive_fields(self, tmp_path):
        """political_preference must not appear in group_knowledge.json topic entries."""
        knowledge_data = {"description": "test", "facts": []}
        profiles = {
            "Peter": {
                "frequently_discussed_topics": ["music"],
                "political_preference": "social democrat",
            }
        }
        out_file = tmp_path / "group_knowledge.json"
        import data.generate_knowledge_base as kb_module
        original = kb_module.GROUP_KNOWLEDGE_FILE
        kb_module.GROUP_KNOWLEDGE_FILE = out_file
        try:
            save_group_knowledge(knowledge_data, profiles)
            with open(out_file, "r") as f:
                saved = json.load(f)
            # political_preference should not appear in any fact text
            for fact in saved["facts"]:
                assert "political" not in fact.get("text", "").lower()
        finally:
            kb_module.GROUP_KNOWLEDGE_FILE = original

    def test_updates_existing_topic_mapping(self, tmp_path):
        knowledge_data = {
            "description": "test",
            "facts": [
                {
                    "id": "topics_peter",
                    "category": "topic_mapping",
                    "subject": "Peter",
                    "text": "Peter frequently discusses: old_topic.",
                }
            ],
        }
        profiles = {
            "Peter": {"frequently_discussed_topics": ["music", "tech"]}
        }
        out_file = tmp_path / "group_knowledge.json"
        import data.generate_knowledge_base as kb_module
        original = kb_module.GROUP_KNOWLEDGE_FILE
        kb_module.GROUP_KNOWLEDGE_FILE = out_file
        try:
            save_group_knowledge(knowledge_data, profiles)
            with open(out_file, "r") as f:
                saved = json.load(f)
            topic_facts = [f for f in saved["facts"] if f["id"] == "topics_peter"]
            assert len(topic_facts) == 1  # No duplicates
            assert "music" in topic_facts[0]["text"]
        finally:
            kb_module.GROUP_KNOWLEDGE_FILE = original


# ===========================================================================
# Integration-style: mock LLM provider
# ===========================================================================

class TestCallLlmForProfiles:
    def test_parses_valid_response(self):
        from data.generate_knowledge_base import call_llm_for_profiles

        mock_provider = MagicMock()
        mock_provider.generate_text.return_value = json.dumps({
            "members": {
                "Peter": {
                    "age": "late 20s",
                    "interests": ["music", "gym"],
                    "biography_summary": "Peter is a developer from Lisbon.",
                    "frequently_discussed_topics": ["music", "tech"],
                }
            }
        })

        result = call_llm_for_profiles(mock_provider, "test prompt")
        assert result is not None
        assert "Peter" in result["members"]
        assert result["members"]["Peter"]["age"] == "late 20s"

    def test_handles_markdown_fences(self):
        from data.generate_knowledge_base import call_llm_for_profiles

        mock_provider = MagicMock()
        mock_provider.generate_text.return_value = (
            "```json\n"
            + json.dumps({"members": {"Gil": {"biography_summary": "Gil is the organizer."}}})
            + "\n```"
        )

        result = call_llm_for_profiles(mock_provider, "prompt")
        assert result is not None
        assert "Gil" in result["members"]

    def test_returns_none_on_invalid_json(self):
        from data.generate_knowledge_base import call_llm_for_profiles

        mock_provider = MagicMock()
        mock_provider.generate_text.return_value = "this is not json"

        result = call_llm_for_profiles(mock_provider, "prompt", max_retries=1)
        assert result is None

    def test_returns_none_on_repeated_api_error(self):
        from data.generate_knowledge_base import call_llm_for_profiles

        mock_provider = MagicMock()
        mock_provider.generate_text.side_effect = Exception("API error")

        result = call_llm_for_profiles(mock_provider, "prompt", max_retries=2, retry_delay=0)
        assert result is None
