"""Shared helpers: reproducibility, IO, and small plotting utilities."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy, and TensorFlow for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:  # pragma: no cover - TF optional at import time
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` (and parents) if missing and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict[str, Any], path: str | os.PathLike) -> None:
    """Write a dict to JSON, creating parent directories as needed."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)


def load_json(path: str | os.PathLike) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    """Make numpy scalars/arrays JSON-serializable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
