"""Generate a small synthetic creditcard.csv for smoke-testing the pipeline.

This is NOT a substitute for the real Kaggle dataset — it only mimics the
schema and imbalance so `python -m src.train` and the notebook run end-to-end
without a 150 MB download. Reported metrics on this data are meaningless.

Usage
-----
    python scripts/make_sample_data.py --rows 20000 --out data/creditcard.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--fraud-rate", type=float, default=0.0172)
    parser.add_argument("--out", default="data/creditcard.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.rows
    n_fraud = max(1, int(n * args.fraud_rate))
    y = np.zeros(n, dtype=int)
    fraud_idx = rng.choice(n, size=n_fraud, replace=False)
    y[fraud_idx] = 1

    # 28 PCA-like features; shift a few of them for the fraud class so there is
    # learnable signal (mimicking V14/V17/etc. in the real data).
    X = rng.normal(0, 1, size=(n, 28))
    signal_cols = [3, 9, 11, 13, 16]
    for c in signal_cols:
        X[fraud_idx, c] += rng.normal(-3.0, 1.0, size=n_fraud)

    time = np.sort(rng.uniform(0, 172_800, size=n))  # 48h in seconds
    amount = np.abs(rng.gamma(2.0, 40.0, size=n))
    amount[fraud_idx] *= rng.uniform(1.5, 4.0, size=n_fraud)

    cols = {"Time": time}
    for i in range(28):
        cols[f"V{i + 1}"] = X[:, i]
    cols["Amount"] = amount
    cols["Class"] = y
    df = pd.DataFrame(cols)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows ({n_fraud} fraud) -> {out_path}")


if __name__ == "__main__":
    main()
