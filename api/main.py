"""FastAPI application exposing fraud-scoring endpoints.

Endpoints
---------
GET  /health                  liveness + model/db status
POST /score                   score an inline transaction payload
GET  /score/{transaction_id}  fetch a row from PostgreSQL and score it
POST /score/batch             fetch many rows by id and score them

Run locally:
    uvicorn api.main:app --reload
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, status

from . import database
from .config import get_settings
from .predict import FraudPredictor, get_predictor
from .schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ScoreResponse,
    TransactionFeatures,
)

app = FastAPI(
    title="Financial Fraud Detection API",
    version="1.0.0",
    description="Scores credit-card transactions with an ANN and returns "
    "calibrated fraud probabilities using a validation-tuned threshold.",
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Report whether the model and database are reachable."""
    db_ok = database.healthcheck()
    model_ok = True
    try:
        get_predictor()
    except Exception:  # noqa: BLE001 - report as not-loaded rather than 500
        model_ok = False
    overall = "ok" if (db_ok and model_ok) else "degraded"
    return HealthResponse(
        status=overall, model_loaded=model_ok, database_connected=db_ok
    )


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score_inline(
    payload: TransactionFeatures,
    predictor: FraudPredictor = Depends(get_predictor),
) -> ScoreResponse:
    """Score a transaction whose features are supplied in the request body."""
    prob = predictor.score(payload.model_dump())
    return ScoreResponse(
        transaction_id=None,
        fraud_probability=prob,
        is_fraud=predictor.decide(prob),
        threshold=predictor.threshold,
    )


@app.get(
    "/score/{transaction_id}",
    response_model=ScoreResponse,
    tags=["scoring"],
)
def score_by_id(
    transaction_id: int,
    predictor: FraudPredictor = Depends(get_predictor),
) -> ScoreResponse:
    """Fetch a transaction from PostgreSQL by id and score it."""
    row = database.fetch_transaction(transaction_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found.",
        )
    prob = predictor.score(row)
    return ScoreResponse(
        transaction_id=transaction_id,
        fraud_probability=prob,
        is_fraud=predictor.decide(prob),
        threshold=predictor.threshold,
    )


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["scoring"])
def score_batch(
    request: BatchScoreRequest,
    predictor: FraudPredictor = Depends(get_predictor),
) -> BatchScoreResponse:
    """Fetch many transactions by id and score them in one model call."""
    found = database.fetch_transactions(request.transaction_ids)
    missing = [tid for tid in request.transaction_ids if tid not in found]

    ids = list(found.keys())
    results: list[ScoreResponse] = []
    if ids:
        probs = predictor.score_many([found[i] for i in ids])
        for tid, prob in zip(ids, probs):
            results.append(
                ScoreResponse(
                    transaction_id=tid,
                    fraud_probability=prob,
                    is_fraud=predictor.decide(prob),
                    threshold=predictor.threshold,
                )
            )
    return BatchScoreResponse(results=results, missing_ids=missing)


@app.get("/", tags=["ops"])
def root() -> dict:
    settings = get_settings()
    return {
        "service": "financial-fraud-detection",
        "docs": "/docs",
        "transactions_table": settings.transactions_table,
    }
