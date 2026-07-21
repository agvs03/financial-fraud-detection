"""Typed configuration loader.

Reads ``config.yaml`` from the project root and exposes it as nested
dataclass-like objects via a thin wrapper so downstream modules can do
``cfg.model.ann.epochs`` instead of dictionary indexing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class _AttrDict(dict):
    """Dictionary whose keys are also accessible as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        if isinstance(value, dict):
            return _AttrDict(value)
        return value


def load_config(path: str | os.PathLike | None = None) -> _AttrDict:
    """Load the YAML config into an attribute-accessible dict.

    Parameters
    ----------
    path:
        Optional override. Falls back to ``config.yaml`` at the repo root,
        which can itself be overridden with the ``FRAUD_CONFIG`` env var.
    """
    resolved = Path(path or os.getenv("FRAUD_CONFIG", DEFAULT_CONFIG_PATH))
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _AttrDict(raw)


# Convenience singleton for scripts that just want the defaults.
CONFIG = load_config()
