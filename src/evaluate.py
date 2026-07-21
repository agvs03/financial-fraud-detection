"""Threshold optimization and evaluation utilities.

Two responsibilities:

1. ``optimize_threshold`` — pick a decision threshold on the *validation* set
   instead of the naive 0.5. For a fraud system, recall on the positive class
   is what matters, but not at the cost of drowning analysts in false
   positives, so we maximize recall subject to a precision floor.
2. ``evaluate`` — compute the full metric suite (precision/recall/F1,
   confusion matrix, ROC-AUC, PR-AUC) and render the standard plots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for CI / servers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import CONFIG, PROJECT_ROOT
from .utils import ensure_dir


@dataclass
class ThresholdResult:
    threshold: float
    val_precision: float
    val_recall: float
    val_f1: float


def optimize_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> ThresholdResult:
    """Choose a threshold on validation scores.

    ``target_recall``: among all thresholds whose recall >= ``target_recall``,
    take the highest-precision one (the fraud-first operating point used by
    default). ``recall_at_precision``: the mirror image — highest recall subject
    to a precision floor. Both fall back gracefully if the constraint is
    unreachable on validation.
    """
    cfg = CONFIG.threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have length len(thresholds)+1; align by dropping last.
    precision, recall = precision[:-1], recall[:-1]

    if cfg.strategy == "target_recall":
        # Fraud-first: guarantee a recall floor, then take the highest-precision
        # (i.e. highest) threshold that still meets it.
        target = cfg.target_recall
        mask = recall >= target
        if mask.any():
            idx_candidates = np.where(mask)[0]
            best = idx_candidates[np.argmax(precision[idx_candidates])]
        else:
            print(
                f"[threshold] target recall {target} unreachable on validation; "
                "taking the max-recall point."
            )
            best = int(np.argmax(recall))
    elif cfg.strategy == "recall_at_precision":
        mask = precision >= cfg.min_precision
        if mask.any():
            idx_candidates = np.where(mask)[0]
            best = idx_candidates[np.argmax(recall[idx_candidates])]
        else:
            print(
                f"[threshold] precision floor {cfg.min_precision} never met; "
                "falling back to best-F1."
            )
            best = _best_f1_index(precision, recall)
    else:  # "best_f1"
        best = _best_f1_index(precision, recall)

    thr = float(thresholds[best])
    f1 = _safe_f1(precision[best], recall[best])
    return ThresholdResult(
        threshold=thr,
        val_precision=float(precision[best]),
        val_recall=float(recall[best]),
        val_f1=float(f1),
    )


def _best_f1_index(precision: np.ndarray, recall: np.ndarray) -> int:
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return int(np.argmax(f1))


def _safe_f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


@dataclass
class EvalReport:
    name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion: list[list[int]]
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "confusion_matrix": self.confusion,
            **self.extra,
        }


def evaluate(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> EvalReport:
    """Compute the full metric suite at a given decision threshold."""
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return EvalReport(
        name=name,
        threshold=float(threshold),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        confusion=cm.tolist(),
    )


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _fig_path(filename: str):
    figures_dir = ensure_dir(PROJECT_ROOT / CONFIG.paths.figures_dir)
    return figures_dir / filename


def plot_confusion_matrix(cm: np.ndarray, name: str) -> str:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Legit", "Fraud"])
    ax.set_yticks([0, 1], labels=["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = _fig_path(f"confusion_{name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_pr_curve(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> str:
    """``curves`` maps model name -> (y_true, y_score)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_score) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = _fig_path("precision_recall_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_roc_curve(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = _fig_path("roc_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
