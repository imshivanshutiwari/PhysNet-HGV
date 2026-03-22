"""
YAML configuration loader with environment-variable overrides.

Reads a YAML file and flattens nested keys into dot-separated paths so that
any value can be overridden via an environment variable of the form
``PHYSNET__section__key=value``.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overrides* into *base* (in-place)."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_env_overrides(cfg: Dict[str, Any], prefix: str = "PHYSNET") -> Dict[str, Any]:
    """Override config values with ``PHYSNET__section__key`` env vars."""
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix + "__"):
            continue
        parts = env_key[len(prefix) + 2 :].lower().split("__")
        target = cfg
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        # Attempt type coercion
        final_key = parts[-1]
        try:
            target[final_key] = yaml.safe_load(env_val)
        except yaml.YAMLError:
            target[final_key] = env_val
    return cfg


def load_config(path: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load a YAML config file and apply optional overrides.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.
    overrides : dict or None
        Additional key-value pairs to merge on top of the loaded config.

    Returns
    -------
    dict
        The fully-resolved configuration dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh) or {}

    # Apply environment overrides first, then explicit overrides
    cfg = _apply_env_overrides(cfg)

    if overrides:
        cfg = _deep_update(cfg, overrides)

    return cfg
