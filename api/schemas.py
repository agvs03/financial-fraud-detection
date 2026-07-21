"""Pydantic request/response models for the scoring API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """Raw transaction features supplied inline (Time, V1..V28, Amount).

    Field names match the Kaggle dataset columns exactly so a row can be
    posted verbatim. ``V*`` fields are required; validation errors are raised
    by FastAPI if any are missing.
    """

    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float
    V7: float; V8: float; V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float; V17: float; V18: float
    V19: float; V20: float; V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float

    model_config = {"extra": "forbid"}


class ScoreResponse(BaseModel):
    transaction_id: int | None = Field(
        default=None,
        description="Present when scoring a row fetched from PostgreSQL.",
    )
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Model P(fraud)."
    )
    is_fraud: bool = Field(
        ..., description="fraud_probability >= tuned decision threshold."
    )
    threshold: float = Field(..., description="Decision threshold applied.")


class BatchScoreRequest(BaseModel):
    transaction_ids: list[int] = Field(..., min_length=1, max_length=1000)


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    missing_ids: list[int] = Field(
        default_factory=list,
        description="Requested ids not found in the transactions table.",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
