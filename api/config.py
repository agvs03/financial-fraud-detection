"""Runtime settings for the API, loaded from environment / .env.

Kept separate from the training-side ``src.config`` so the service can be
deployed without the training dependencies. Values are read once at import.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="FRAUD_", extra="ignore"
    )

    # PostgreSQL connection (SQLAlchemy URL). Example:
    #   postgresql+psycopg2://user:pass@localhost:5432/fraud
    database_url: str = (
        "postgresql+psycopg2://fraud:fraud@localhost:5432/fraud"
    )

    # Artifact locations (relative to repo root by default).
    ann_model_path: str = "models/ann_fraud.keras"
    scaler_path: str = "models/scaler.joblib"
    threshold_path: str = "models/threshold.json"

    # Columns the scaler was fit on, in order. Must match training.
    robust_scale_cols: tuple[str, ...] = ("Amount", "Time")

    # Full feature order expected by the model (Time, V1..V28, Amount).
    feature_order: tuple[str, ...] = (
        ("Time",)
        + tuple(f"V{i}" for i in range(1, 29))
        + ("Amount",)
    )

    # Name of the table holding transactions to score.
    transactions_table: str = "transactions"


@lru_cache
def get_settings() -> Settings:
    return Settings()
