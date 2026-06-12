"""
Tests for src/config_loader.py — model profile resolution and config merging.

Validates:
- Profile merge: active_model_profile='gemma4-e4b' overrides model.model_id,
  training.output_dir, and lora_r correctly.
- Backward compat: configs without model_profiles / active_model_profile are
  returned unchanged.
- Invalid profile: load_config with a non-existent profile raises ValueError
  (not KeyError).
- profile_override kwarg: explicitly passing profile_override='qwen3-14b'
  takes precedence over active_model_profile in the file.
- Deep merge: profile values override only the keys they declare; unspecified
  keys (e.g. lora_dropout) fall through from the base config unchanged.
"""

import sys
from pathlib import Path

import pytest
import yaml

# Make src/ importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config_loader import load_config, _deep_merge

# ---------------------------------------------------------------------------
# Fixtures — temporary YAML config files
# ---------------------------------------------------------------------------

#: Base config sections shared across several tests
_BASE_MODEL = {
    "model_id": "unsloth/Qwen3-14B-bnb-4bit",
    "max_seq_length": 4096,
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}

_BASE_TRAINING = {
    "output_dir": "./models/kaya_v2_synthetic",
    "max_steps": 1500,
    "learning_rate": 0.0001,
}

_QWEN3_PROFILE = {
    "model": {
        "model_id": "unsloth/Qwen3-14B-bnb-4bit",
        "max_seq_length": 4096,
        "lora_r": 32,
    },
    "training": {
        "output_dir": "./models/kaya_qwen3_14b",
    },
}

_GEMMA4_PROFILE = {
    "model": {
        "model_id": "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
        "max_seq_length": 4096,
        "lora_r": 16,
    },
    "training": {
        "output_dir": "./models/kaya_gemma4_e4b",
    },
}


def _write_yaml(path: Path, data: dict) -> Path:
    """Serialise *data* to *path* as YAML and return the path."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, allow_unicode=True)
    return path


@pytest.fixture()
def full_config(tmp_path: Path) -> Path:
    """Config file with two profiles and active_model_profile='qwen3-14b'."""
    data = {
        "active_model_profile": "qwen3-14b",
        "model": dict(_BASE_MODEL),
        "training": dict(_BASE_TRAINING),
        "model_profiles": {
            "qwen3-14b": dict(_QWEN3_PROFILE),
            "gemma4-e4b": dict(_GEMMA4_PROFILE),
        },
    }
    return _write_yaml(tmp_path / "config_full.yaml", data)


@pytest.fixture()
def legacy_config(tmp_path: Path) -> Path:
    """Config file with NO model_profiles / active_model_profile (legacy format)."""
    data = {
        "model": dict(_BASE_MODEL),
        "training": dict(_BASE_TRAINING),
    }
    return _write_yaml(tmp_path / "config_legacy.yaml", data)


@pytest.fixture()
def gemma_active_config(tmp_path: Path) -> Path:
    """Config file with active_model_profile set to 'gemma4-e4b'."""
    data = {
        "active_model_profile": "gemma4-e4b",
        "model": dict(_BASE_MODEL),
        "training": dict(_BASE_TRAINING),
        "model_profiles": {
            "qwen3-14b": dict(_QWEN3_PROFILE),
            "gemma4-e4b": dict(_GEMMA4_PROFILE),
        },
    }
    return _write_yaml(tmp_path / "config_gemma.yaml", data)


# ---------------------------------------------------------------------------
# TestDeepMerge — unit tests for the internal helper
# ---------------------------------------------------------------------------

class TestDeepMerge:
    """Unit tests for the _deep_merge helper."""

    def test_top_level_override(self):
        result = _deep_merge({"a": 1, "b": 2}, {"b": 99})
        assert result["a"] == 1
        assert result["b"] == 99

    def test_nested_override(self):
        base = {"model": {"id": "old", "lr": 0.01}}
        override = {"model": {"id": "new"}}
        result = _deep_merge(base, override)
        assert result["model"]["id"] == "new"
        assert result["model"]["lr"] == 0.01  # untouched

    def test_base_not_mutated(self):
        base = {"model": {"id": "base"}}
        _deep_merge(base, {"model": {"id": "new"}})
        assert base["model"]["id"] == "base"

    def test_override_not_mutated(self):
        override = {"model": {"id": "new"}}
        _deep_merge({"model": {"id": "base"}}, override)
        assert override["model"]["id"] == "new"

    def test_new_key_added(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result["a"] == 1
        assert result["b"] == 2


# ---------------------------------------------------------------------------
# TestProfileMerge — loading a config activates the named profile
# ---------------------------------------------------------------------------

class TestProfileMerge:
    """Profile merge: active_model_profile overrides model and training keys."""

    def test_gemma4_model_id_overridden(self, gemma_active_config):
        cfg = load_config(str(gemma_active_config))
        assert cfg["model"]["model_id"] == "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

    def test_gemma4_lora_r_overridden(self, gemma_active_config):
        cfg = load_config(str(gemma_active_config))
        assert cfg["model"]["lora_r"] == 16

    def test_gemma4_training_output_dir_overridden(self, gemma_active_config):
        cfg = load_config(str(gemma_active_config))
        assert cfg["training"]["output_dir"] == "./models/kaya_gemma4_e4b"

    def test_qwen3_model_id_overridden(self, full_config):
        cfg = load_config(str(full_config))
        assert cfg["model"]["model_id"] == "unsloth/Qwen3-14B-bnb-4bit"

    def test_qwen3_lora_r_overridden(self, full_config):
        cfg = load_config(str(full_config))
        assert cfg["model"]["lora_r"] == 32

    def test_qwen3_training_output_dir_overridden(self, full_config):
        cfg = load_config(str(full_config))
        assert cfg["training"]["output_dir"] == "./models/kaya_qwen3_14b"


# ---------------------------------------------------------------------------
# TestBackwardCompatibility — legacy configs without profiles pass through
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Configs that predate the profile system are returned unchanged."""

    def test_model_id_preserved(self, legacy_config):
        cfg = load_config(str(legacy_config))
        assert cfg["model"]["model_id"] == "unsloth/Qwen3-14B-bnb-4bit"

    def test_lora_r_preserved(self, legacy_config):
        cfg = load_config(str(legacy_config))
        assert cfg["model"]["lora_r"] == 32

    def test_output_dir_preserved(self, legacy_config):
        cfg = load_config(str(legacy_config))
        assert cfg["training"]["output_dir"] == "./models/kaya_v2_synthetic"

    def test_no_model_profiles_key_injected(self, legacy_config):
        """model_profiles key should NOT appear when the base config lacks it."""
        cfg = load_config(str(legacy_config))
        # The returned dict should not have model_profiles injected
        assert cfg.get("model_profiles") is None

    def test_config_without_active_profile_key(self, tmp_path):
        """Config with model_profiles but no active_model_profile → raw config."""
        data = {
            "model": dict(_BASE_MODEL),
            "training": dict(_BASE_TRAINING),
            "model_profiles": {
                "qwen3-14b": dict(_QWEN3_PROFILE),
            },
        }
        path = _write_yaml(tmp_path / "no_active.yaml", data)
        cfg = load_config(str(path))
        # No profile was applied — base model_id should remain
        assert cfg["model"]["model_id"] == "unsloth/Qwen3-14B-bnb-4bit"
        assert cfg["training"]["output_dir"] == "./models/kaya_v2_synthetic"


# ---------------------------------------------------------------------------
# TestInvalidProfile — clear ValueError for unknown profiles
# ---------------------------------------------------------------------------

class TestInvalidProfile:
    """Requesting a non-existent profile raises ValueError, not KeyError."""

    def test_raises_value_error(self, full_config):
        with pytest.raises(ValueError):
            load_config(str(full_config), profile_override="nonexistent")

    def test_error_mentions_profile_name(self, full_config):
        with pytest.raises(ValueError, match="nonexistent"):
            load_config(str(full_config), profile_override="nonexistent")

    def test_error_lists_available_profiles(self, full_config):
        with pytest.raises(ValueError, match="qwen3-14b"):
            load_config(str(full_config), profile_override="nonexistent")

    def test_not_key_error(self, full_config):
        try:
            load_config(str(full_config), profile_override="nonexistent")
        except KeyError:
            pytest.fail("load_config raised KeyError instead of ValueError")
        except ValueError:
            pass  # expected


# ---------------------------------------------------------------------------
# TestProfileOverrideKwarg — explicit kwarg wins over file's active profile
# ---------------------------------------------------------------------------

class TestProfileOverrideKwarg:
    """profile_override kwarg takes precedence over active_model_profile in file."""

    def test_override_switches_from_qwen3_to_gemma4(self, full_config):
        """full_config has active_model_profile='qwen3-14b'; override should apply gemma4-e4b."""
        cfg = load_config(str(full_config), profile_override="gemma4-e4b")
        assert cfg["model"]["model_id"] == "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

    def test_override_switches_lora_r(self, full_config):
        cfg = load_config(str(full_config), profile_override="gemma4-e4b")
        assert cfg["model"]["lora_r"] == 16

    def test_override_switches_output_dir(self, full_config):
        cfg = load_config(str(full_config), profile_override="gemma4-e4b")
        assert cfg["training"]["output_dir"] == "./models/kaya_gemma4_e4b"

    def test_override_selects_qwen3_from_gemma_config(self, gemma_active_config):
        """gemma_active_config has active_model_profile='gemma4-e4b'; override → qwen3-14b."""
        cfg = load_config(str(gemma_active_config), profile_override="qwen3-14b")
        assert cfg["model"]["model_id"] == "unsloth/Qwen3-14B-bnb-4bit"
        assert cfg["model"]["lora_r"] == 32

    def test_override_none_uses_file_active_profile(self, full_config):
        """Passing profile_override=None should use the file's active_model_profile."""
        cfg = load_config(str(full_config), profile_override=None)
        assert cfg["training"]["output_dir"] == "./models/kaya_qwen3_14b"


# ---------------------------------------------------------------------------
# TestDeepMergePassThrough — unspecified keys fall through from base config
# ---------------------------------------------------------------------------

class TestDeepMergePassThrough:
    """Unspecified keys in the profile are preserved from the base config."""

    def test_lora_dropout_preserved_after_gemma4_merge(self, gemma_active_config):
        """gemma4-e4b profile does NOT declare lora_dropout → base value kept."""
        cfg = load_config(str(gemma_active_config))
        assert cfg["model"]["lora_dropout"] == 0.05

    def test_lora_alpha_preserved_after_gemma4_merge(self, gemma_active_config):
        cfg = load_config(str(gemma_active_config))
        assert cfg["model"]["lora_alpha"] == 32

    def test_max_steps_preserved_after_profile_merge(self, gemma_active_config):
        """Profile does NOT declare max_steps → base training value kept."""
        cfg = load_config(str(gemma_active_config))
        assert cfg["training"]["max_steps"] == 1500

    def test_learning_rate_preserved_after_profile_merge(self, gemma_active_config):
        cfg = load_config(str(gemma_active_config))
        assert cfg["training"]["learning_rate"] == 0.0001

    def test_profile_only_overrides_declared_keys(self, tmp_path):
        """A profile that only changes model_id should leave all other keys intact."""
        data = {
            "active_model_profile": "minimal",
            "model": {
                "model_id": "base-model",
                "lora_r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.1,
                "max_seq_length": 2048,
            },
            "training": {
                "output_dir": "./models/base",
                "max_steps": 500,
            },
            "model_profiles": {
                "minimal": {
                    "model": {"model_id": "override-model"},
                }
            },
        }
        path = _write_yaml(tmp_path / "minimal_profile.yaml", data)
        cfg = load_config(str(path))

        assert cfg["model"]["model_id"] == "override-model"
        assert cfg["model"]["lora_r"] == 32
        assert cfg["model"]["lora_alpha"] == 64
        assert cfg["model"]["lora_dropout"] == 0.1
        assert cfg["model"]["max_seq_length"] == 2048
        assert cfg["training"]["output_dir"] == "./models/base"
        assert cfg["training"]["max_steps"] == 500


# ---------------------------------------------------------------------------
# TestRealConfig — integration tests against the live config.yaml
# ---------------------------------------------------------------------------

class TestRealConfig:
    """Integration tests that load the actual project config.yaml."""

    def test_default_path_resolves_to_project_root(self):
        """load_config() with no arguments should resolve to the real config.yaml."""
        config = load_config()
        assert "model" in config
        assert "training" in config

    def test_active_model_profile_in_yaml(self, tmp_path):
        """active_model_profile set in YAML is used when no override provided."""
        cfg_text = """
model:
  model_id: "base_model"
  lora_r: 32
  lora_alpha: 32
training:
  output_dir: "./models/base"
  learning_rate: 0.0001
active_model_profile: profile-b
model_profiles:
  profile-a:
    model:
      model_id: "model_a"
      lora_r: 16
    training:
      output_dir: "./models/a"
  profile-b:
    model:
      model_id: "model_b"
    training:
      output_dir: "./models/b"
"""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(cfg_text, encoding="utf-8")
        config = load_config(str(cfg_file))
        assert config["model"]["model_id"] == "model_b"
        assert config["training"]["output_dir"] == "./models/b"

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
