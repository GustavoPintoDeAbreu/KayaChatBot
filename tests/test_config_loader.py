"""
Unit tests for src/config_loader.py.

Validates load_config's profile merging, backward compatibility,
and unknown-profile graceful fallback.
"""

import sys
import tempfile
import textwrap
from pathlib import Path

import pytest
import yaml

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config, deep_merge


# ---------------------------------------------------------------------------
# Tests for _deep_merge helper
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self):
        base = {"model": {"id": "base_model", "lora_r": 32, "extra": "keep"}}
        override = {"model": {"id": "new_model", "lora_r": 16}}
        result = deep_merge(base, override)
        assert result["model"]["id"] == "new_model"
        assert result["model"]["lora_r"] == 16
        assert result["model"]["extra"] == "keep"  # unaffected key preserved

    def test_no_mutation_of_base(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 2}}
        deep_merge(base, override)
        assert base["a"]["x"] == 1  # base must not be modified

    def test_list_replacement(self):
        base = {"modules": ["a", "b"]}
        override = {"modules": ["c", "d"]}
        result = deep_merge(base, override)
        assert result["modules"] == ["c", "d"]


# ---------------------------------------------------------------------------
# Helper — write a minimal config YAML to a tmp file
# ---------------------------------------------------------------------------

_BASE_CONFIG = textwrap.dedent("""\
    model:
      model_id: "base_model"
      lora_r: 32
      lora_alpha: 32
    training:
      output_dir: "./models/base"
      learning_rate: 0.0001
    active_model_profile: null
    model_profiles:
      profile-a:
        model:
          model_id: "model_a"
          lora_r: 16
        training:
          output_dir: "./models/a"
          learning_rate: 0.00005
      profile-b:
        model:
          model_id: "model_b"
        training:
          output_dir: "./models/b"
""")


@pytest.fixture
def tmp_config(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(_BASE_CONFIG, encoding="utf-8")
    return str(cfg_file)


# ---------------------------------------------------------------------------
# Tests for load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_no_profile_returns_top_level(self, tmp_config):
        """When active_model_profile is null and no override, top-level values are used."""
        config = load_config(tmp_config)
        assert config["model"]["model_id"] == "base_model"
        assert config["model"]["lora_r"] == 32
        assert config["training"]["output_dir"] == "./models/base"

    def test_profile_override_merges_model(self, tmp_config):
        """profile_override applies the named profile over top-level values."""
        config = load_config(tmp_config, profile_override="profile-a")
        assert config["model"]["model_id"] == "model_a"
        assert config["model"]["lora_r"] == 16
        # Keys not in profile are preserved from top-level
        assert config["model"]["lora_alpha"] == 32

    def test_profile_override_merges_training(self, tmp_config):
        config = load_config(tmp_config, profile_override="profile-a")
        assert config["training"]["output_dir"] == "./models/a"
        assert config["training"]["learning_rate"] == 0.00005

    def test_active_model_profile_in_yaml(self, tmp_path):
        """active_model_profile set in YAML is used when no override provided."""
        cfg_text = _BASE_CONFIG.replace(
            "active_model_profile: null", "active_model_profile: profile-b"
        )
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(cfg_text, encoding="utf-8")
        config = load_config(str(cfg_file))
        assert config["model"]["model_id"] == "model_b"
        assert config["training"]["output_dir"] == "./models/b"

    def test_profile_override_empty_string_uses_top_level(self, tmp_config):
        """Passing profile_override='' disables profile merging."""
        config = load_config(tmp_config, profile_override="")
        assert config["model"]["model_id"] == "base_model"

    def test_unknown_profile_raises_value_error(self, tmp_config):
        with pytest.raises(ValueError, match="nonexistent"):
            load_config(tmp_config, profile_override="nonexistent")

    def test_default_path_resolves_to_project_root(self):
        """load_config() with no arguments should resolve to the real config.yaml."""
        config = load_config()
        # The real config always has a model section
        assert "model" in config
        assert "training" in config

    def test_profile_does_not_mutate_profiles_section(self, tmp_config):
        """Applying a profile must not alter model_profiles in the returned dict."""
        config = load_config(tmp_config, profile_override="profile-a")
        assert config["model_profiles"]["profile-a"]["model"]["model_id"] == "model_a"

    def test_qwen3_profile_present_in_real_config(self):
        """The real config.yaml must contain the qwen3-14b and gemma4-e4b profiles."""
        config = load_config()
        profiles = config.get("model_profiles", {})
        assert "qwen3-14b" in profiles, "qwen3-14b profile missing from config.yaml"
        assert "gemma4-e4b" in profiles, "gemma4-e4b profile missing from config.yaml"

    def test_gemma4_model_id_correct(self):
        """gemma4-e4b profile must use the real HuggingFace model ID."""
        config = load_config()
        gemma_id = config["model_profiles"]["gemma4-e4b"]["model"]["model_id"]
        assert gemma_id == "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

    def test_apply_gemma4_profile(self):
        """Applying gemma4-e4b profile overrides model and training sections."""
        config = load_config(profile_override="gemma4-e4b")
        assert config["model"]["model_id"] == "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
        assert config["model"]["lora_r"] == 16
        assert config["model"]["lora_alpha"] == 32
        assert config["model"]["lora_dropout"] == 0.0
        assert "gemma4" in config["training"]["output_dir"]

    def test_apply_qwen3_profile(self):
        """Applying qwen3-14b profile overrides model and training sections."""
        config = load_config(profile_override="qwen3-14b")
        assert config["model"]["model_id"] == "unsloth/Qwen3-14B-bnb-4bit"
        assert config["model"]["lora_r"] == 32
        assert "kaya_qwen3_14b" in config["training"]["output_dir"]
