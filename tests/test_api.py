"""API tests that stub out the model and database.

These verify request/response wiring and the decision logic without needing a
trained model or a live PostgreSQL instance. Run with ``pytest``.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from api import database, main
from api.predict import get_predictor


class _StubPredictor:
    """Deterministic stand-in: probability is Amount / 1000, capped at 1."""

    threshold = 0.5

    def score(self, features: dict) -> float:
        return min(float(features["Amount"]) / 1000.0, 1.0)

    def score_many(self, feature_dicts: list[dict]) -> list[float]:
        return [self.score(f) for f in feature_dicts]

    def decide(self, probability: float) -> bool:
        return probability >= self.threshold


def _sample_features(amount: float = 100.0) -> dict:
    feats = {"Time": 0.0, "Amount": amount}
    for i in range(1, 29):
        feats[f"V{i}"] = 0.0
    return feats


def _client() -> TestClient:
    main.app.dependency_overrides[get_predictor] = lambda: _StubPredictor()
    return TestClient(main.app)


def teardown_function() -> None:
    main.app.dependency_overrides.clear()


def test_score_inline_legit() -> None:
    client = _client()
    resp = client.post("/score", json=_sample_features(amount=100.0))
    assert resp.status_code == 200
    body = resp.json()
    assert body["fraud_probability"] == 0.1
    assert body["is_fraud"] is False
    assert body["threshold"] == 0.5


def test_score_inline_fraud() -> None:
    client = _client()
    resp = client.post("/score", json=_sample_features(amount=900.0))
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_fraud"] is True


def test_score_inline_rejects_extra_field() -> None:
    client = _client()
    payload = _sample_features()
    payload["unexpected"] = 1.0
    resp = client.post("/score", json=payload)
    assert resp.status_code == 422  # extra="forbid"


def test_score_by_id_found(monkeypatch) -> None:
    monkeypatch.setattr(
        database, "fetch_transaction", lambda tid: _sample_features(700.0)
    )
    client = _client()
    resp = client.get("/score/42")
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == 42
    assert body["is_fraud"] is True


def test_score_by_id_missing(monkeypatch) -> None:
    monkeypatch.setattr(database, "fetch_transaction", lambda tid: None)
    client = _client()
    resp = client.get("/score/999")
    assert resp.status_code == 404


def test_score_batch(monkeypatch) -> None:
    monkeypatch.setattr(
        database,
        "fetch_transactions",
        lambda ids: {1: _sample_features(100.0), 2: _sample_features(800.0)},
    )
    client = _client()
    resp = client.post("/score/batch", json={"transaction_ids": [1, 2, 3]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert body["missing_ids"] == [3]
