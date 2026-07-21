"""Classical baselines: Logistic Regression, Random Forest, XGBoost.

Each is wrapped in a factory that returns an *unfitted* estimator configured
for class imbalance (class_weight / scale_pos_weight). They are trained on the
same SMOTE-resampled fold as the ANN so the comparison is apples-to-apples.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ..config import CONFIG

_SEED = CONFIG.data.random_state


def logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=_SEED,
    )


def random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=_SEED,
    )


def xgboost(scale_pos_weight: float = 1.0) -> XGBClassifier:
    """``scale_pos_weight`` should be ~ (n_negative / n_positive) of the
    fold XGBoost actually trains on."""
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=_SEED,
    )


def all_baselines(scale_pos_weight: float = 1.0) -> dict:
    """Return a name -> estimator map of every baseline."""
    return {
        "logistic_regression": logistic_regression(),
        "random_forest": random_forest(),
        "xgboost": xgboost(scale_pos_weight=scale_pos_weight),
    }
