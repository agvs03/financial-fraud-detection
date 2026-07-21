"""Model loading + inference.

Loads the Keras ANN, the fitted RobustScaler, and the tuned threshold once at
process start (via a cached singleton) and turns raw feature dicts into fraud
probabilities. TensorFlow is imported lazily so that importing this module for
tests that stub the predictor doesn't force a TF import.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from .config import Settings, get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FraudPredictor:
    """Wraps the ANN + scaler + threshold behind a single ``score`` call."""

    def __init__(self, settings: Settings):
        import tensorflow as tf  # local import keeps TF out of light paths
        import json

        self.settings = settings
        self.feature_order = list(settings.feature_order)
        self.scale_cols = list(settings.robust_scale_cols)

        model_path = PROJECT_ROOT / settings.ann_model_path
        scaler_path = PROJECT_ROOT / settings.scaler_path
        threshold_path = PROJECT_ROOT / settings.threshold_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Train it first: "
                "`python -m src.train`."
            )

        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        with threshold_path.open("r", encoding="utf-8") as fh:
            self.threshold = float(json.load(fh)["threshold"])

        # Index of each scaled column within the ordered feature vector.
        self._scale_idx = [self.feature_order.index(c) for c in self.scale_cols]

    def _vectorize(self, features: dict) -> np.ndarray:
        """Order a feature dict into the model's expected vector layout."""
        try:
            return np.array(
                [float(features[name]) for name in self.feature_order],
                dtype=np.float32,
            )
        except KeyError as exc:
            raise ValueError(f"Missing feature: {exc.args[0]}") from exc

    def _apply_scaler(self, matrix: np.ndarray) -> np.ndarray:
        """Scale only the RobustScaler columns, matching training."""
        matrix = matrix.copy()
        sub = matrix[:, self._scale_idx]
        matrix[:, self._scale_idx] = self.scaler.transform(sub)
        return matrix

    def score_many(self, feature_dicts: list[dict]) -> list[float]:
        matrix = np.vstack([self._vectorize(f) for f in feature_dicts])
        matrix = self._apply_scaler(matrix)
        probs = self.model.predict(matrix, verbose=0).ravel()
        return [float(p) for p in probs]

    def score(self, features: dict) -> float:
        return self.score_many([features])[0]

    def decide(self, probability: float) -> bool:
        return probability >= self.threshold


@lru_cache
def get_predictor() -> FraudPredictor:
    """Cached predictor singleton for the API dependency system."""
    return FraudPredictor(get_settings())
