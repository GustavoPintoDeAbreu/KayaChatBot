"""
Config loader with model profile support.

Loads a YAML configuration file and deep-merges the active model profile's
``model:`` and ``training:`` sections into the top-level equivalents.
If no active profile is set, the top-level values are returned unchanged
(backward compatible).

Usage::

    from src.config_loader import load_config

    config = load_config()                          # uses config.yaml in project root
    config = load_config("config.docker.yaml")      # explicit path
    config = load_config(profile_override="gemma4-e4b")  # override profile at runtime
"""

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* (override wins on conflict).

    Nested dicts are merged recursively; all other types are replaced.
    Neither argument is modified in-place.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# Public alias so callers that imported `deep_merge` continue to work.
deep_merge = _deep_merge


def load_config(path: Optional[str] = None, profile_override: Optional[str] = None) -> dict:
    """Load a YAML config and apply the active model profile.

    Args:
        path: Path to the config YAML file.  Defaults to ``config.yaml`` in the
              project root (two levels above this file).
        profile_override: Profile name to activate instead of the
                          ``active_model_profile`` key in the YAML file.
                          Pass an empty string or ``None`` to disable profile
                          merging entirely (use top-level values as-is).

    Returns:
        Resolved config dict.  The ``model:`` and ``training:`` sections
        reflect the active profile's values merged on top of the top-level
        defaults.
    """
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"

    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Determine which profile to activate.
    # profile_override="" means "no profile, use top-level values".
    if profile_override is not None:
        profile_name = profile_override if profile_override else None
    else:
        profile_name = config.get("active_model_profile") or None

    if not profile_name:
        return config  # backward compatible: no profile set

    profiles = config.get("model_profiles", {})
    if profile_name not in profiles:
        available = sorted(profiles.keys())
        raise ValueError(
            f"Model profile '{profile_name}' not found. "
            f"Available profiles: {available}"
        )

    profile = profiles[profile_name]

    # Deep-merge profile's model: and training: into the top-level sections.
    if "model" in profile:
        config["model"] = _deep_merge(config.get("model", {}), profile["model"])
    if "training" in profile:
        config["training"] = _deep_merge(config.get("training", {}), profile["training"])

    return config
