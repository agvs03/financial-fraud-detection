"""Dataset loading, splitting, scaling, and SMOTE oversampling.

The Kaggle "Credit Card Fraud Detection" dataset (ULB) has 284,807 rows with
492 frauds (~0.172%). Features ``V1``..``V28`` are already PCA components;
only ``Time`` and ``Amount`` are on their original scale, so those two get a
RobustScaler. SMOTE is fit **only** on the training fold to prevent leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from .config import CONFIG, PROJECT_ROOT
from .utils import ensure_dir


@dataclass
class DataBundle:
    """Container for all splits produced by the pipeline."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: RobustScaler


def load_raw(path: str | None = None) -> pd.DataFrame:
    """Load the raw CSV. Path is resolved relative to the project root."""
    cfg = CONFIG.data
    raw_path = PROJECT_ROOT / (path or cfg.raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. "
            "Download it from Kaggle (mlg-ulb/creditcardfraud) and place "
            "creditcard.csv under data/. See README for details."
        )
    df = pd.read_csv(raw_path)
    expected = 284_807
    if len(df) < 200_000:  # sanity check, not exact (dataset is fixed size)
        print(f"[warn] Expected ~{expected} rows, got {len(df)}.")
    return df


def _scale(
    train: pd.DataFrame, *frames: pd.DataFrame, cols: list[str]
) -> tuple[RobustScaler, list[pd.DataFrame]]:
    """Fit a RobustScaler on the training frame and transform every frame."""
    scaler = RobustScaler()
    train = train.copy()
    train[cols] = scaler.fit_transform(train[cols])
    out = [train]
    for frame in frames:
        frame = frame.copy()
        frame[cols] = scaler.transform(frame[cols])
        out.append(frame)
    return scaler, out


def build_dataset(save_scaler: bool = True) -> DataBundle:
    """Full preprocessing pipeline returning train/val/test splits.

    Steps
    -----
    1. Stratified train/test split, then a stratified val split off of train.
    2. RobustScaler on ``Amount``/``Time`` fit on train only.
    3. SMOTE on the (scaled) training fold only.
    """
    data_cfg = CONFIG.data
    prep_cfg = CONFIG.preprocessing

    df = load_raw()
    target = data_cfg.target_col
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    feature_names = list(X.columns)

    # 1. Stratified splits -----------------------------------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=data_cfg.test_size,
        stratify=y,
        random_state=data_cfg.random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=data_cfg.val_size,
        stratify=y_trainval,
        random_state=data_cfg.random_state,
    )

    # 2. Scaling (fit on train only) ------------------------------------------
    scale_cols = prep_cfg.robust_scale_cols
    scaler, (X_train, X_val, X_test) = _scale(
        X_train, X_val, X_test, cols=scale_cols
    )

    # 3. SMOTE on the training fold only --------------------------------------
    X_train_arr = X_train.to_numpy(dtype=np.float32)
    y_train_arr = y_train.to_numpy(dtype=np.int64)
    if prep_cfg.smote.enabled:
        smote = SMOTE(
            sampling_strategy=prep_cfg.smote.sampling_strategy,
            k_neighbors=prep_cfg.smote.k_neighbors,
            random_state=data_cfg.random_state,
        )
        X_train_arr, y_train_arr = smote.fit_resample(X_train_arr, y_train_arr)
        n_fraud = int(y_train_arr.sum())
        print(
            f"[smote] resampled train -> {len(y_train_arr):,} rows "
            f"({n_fraud:,} fraud / {len(y_train_arr) - n_fraud:,} legit)"
        )

    if save_scaler:
        ensure_dir(PROJECT_ROOT / CONFIG.paths.model_dir)
        joblib.dump(scaler, PROJECT_ROOT / CONFIG.paths.scaler)

    return DataBundle(
        X_train=X_train_arr,
        y_train=y_train_arr,
        X_val=X_val.to_numpy(dtype=np.float32),
        y_val=y_val.to_numpy(dtype=np.int64),
        X_test=X_test.to_numpy(dtype=np.float32),
        y_test=y_test.to_numpy(dtype=np.int64),
        feature_names=feature_names,
        scaler=scaler,
    )


if __name__ == "__main__":
    bundle = build_dataset()
    print("train:", bundle.X_train.shape, "val:", bundle.X_val.shape,
          "test:", bundle.X_test.shape)
