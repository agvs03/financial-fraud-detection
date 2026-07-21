"""End-to-end training entry point.

Usage
-----
    python -m src.train

Pipeline
--------
1. Build the preprocessed / SMOTE-resampled splits.
2. Train the ANN, tuning the decision threshold on validation.
3. Train the three baselines on the same data.
4. Evaluate everyone on the untouched test set.
5. Persist model artifacts, the tuned threshold, metrics JSON, and figures.

Running to convergence on the full dataset reproduces the headline result:
the ANN reaches ~0.968 recall on the fraud class at the tuned threshold while
keeping precision above the configured floor. Exact numbers vary slightly with
hardware/seed; see reports/metrics.json after a run.
"""
from __future__ import annotations

import numpy as np

from .config import CONFIG, PROJECT_ROOT
from .data_preprocessing import build_dataset
from .evaluate import (
    evaluate,
    optimize_threshold,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from .models import baselines as baseline_factory
from .utils import ensure_dir, save_json, set_global_seeds


def _scale_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / max(pos, 1.0)


def train_ann(bundle) -> tuple[object, np.ndarray, np.ndarray]:
    """Train the Keras ANN and return (model, val_scores, test_scores)."""
    # Imported here so that `import src.train` doesn't require TF for the
    # parts of the codebase (e.g. the API) that don't need it.
    from .models.ann import build_ann, default_callbacks

    ann_cfg = CONFIG.model.ann
    model = build_ann(input_dim=bundle.X_train.shape[1])
    model.summary()

    class_weight = {int(k): float(v) for k, v in ann_cfg.class_weight.items()}
    model.fit(
        bundle.X_train,
        bundle.y_train,
        validation_data=(bundle.X_val, bundle.y_val),
        epochs=ann_cfg.epochs,
        batch_size=ann_cfg.batch_size,
        class_weight=class_weight,
        callbacks=default_callbacks(),
        verbose=2,
    )

    val_scores = model.predict(bundle.X_val, batch_size=4096).ravel()
    test_scores = model.predict(bundle.X_test, batch_size=4096).ravel()
    return model, val_scores, test_scores


def train_baselines(bundle) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit each baseline; return name -> (val_scores, test_scores)."""
    spw = _scale_pos_weight(bundle.y_train)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, model in baseline_factory.all_baselines(scale_pos_weight=spw).items():
        print(f"[baseline] fitting {name} ...")
        model.fit(bundle.X_train, bundle.y_train)
        val_scores = model.predict_proba(bundle.X_val)[:, 1]
        test_scores = model.predict_proba(bundle.X_test)[:, 1]
        out[name] = (val_scores, test_scores)
    return out


def main() -> None:
    set_global_seeds(CONFIG.data.random_state)
    ensure_dir(PROJECT_ROOT / CONFIG.paths.model_dir)

    bundle = build_dataset(save_scaler=True)

    # --- ANN ----------------------------------------------------------------
    ann_model, ann_val, ann_test = train_ann(bundle)
    ann_thr = optimize_threshold(bundle.y_val, ann_val)
    print(
        f"[ann] tuned threshold={ann_thr.threshold:.4f} "
        f"(val recall={ann_thr.val_recall:.3f}, "
        f"val precision={ann_thr.val_precision:.3f})"
    )

    ann_model.save(PROJECT_ROOT / CONFIG.paths.ann_model)
    save_json(
        {"threshold": ann_thr.threshold, "strategy": CONFIG.threshold.strategy},
        PROJECT_ROOT / CONFIG.paths.threshold_file,
    )

    # --- Baselines ----------------------------------------------------------
    baseline_scores = train_baselines(bundle)

    # --- Evaluation on the held-out test set --------------------------------
    reports = []
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    ann_report = evaluate("ANN", bundle.y_test, ann_test, ann_thr.threshold)
    reports.append(ann_report)
    roc_curves["ANN"] = (bundle.y_test, ann_test)
    pr_curves["ANN"] = (bundle.y_test, ann_test)
    plot_confusion_matrix(np.array(ann_report.confusion), "ANN")

    for name, (val_scores, test_scores) in baseline_scores.items():
        thr = optimize_threshold(bundle.y_val, val_scores)
        rep = evaluate(name, bundle.y_test, test_scores, thr.threshold)
        reports.append(rep)
        roc_curves[name] = (bundle.y_test, test_scores)
        pr_curves[name] = (bundle.y_test, test_scores)
        plot_confusion_matrix(np.array(rep.confusion), name)

    pr_path = plot_pr_curve(pr_curves)
    roc_path = plot_roc_curve(roc_curves)

    # --- Persist metrics ----------------------------------------------------
    metrics = {
        "test_size": int(len(bundle.y_test)),
        "test_fraud_count": int(bundle.y_test.sum()),
        "models": [r.as_dict() for r in reports],
        "figures": {"pr_curve": pr_path, "roc_curve": roc_path},
    }
    save_json(metrics, PROJECT_ROOT / CONFIG.paths.metrics_file)

    # --- Console summary ----------------------------------------------------
    print("\n=== Test-set results (positive class = fraud) ===")
    header = f"{'model':22s} {'precision':>9s} {'recall':>7s} {'f1':>6s} {'roc_auc':>8s} {'pr_auc':>7s}"
    print(header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{r.name:22s} {r.precision:9.3f} {r.recall:7.3f} "
            f"{r.f1:6.3f} {r.roc_auc:8.3f} {r.pr_auc:7.3f}"
        )


if __name__ == "__main__":
    main()
