"""Score real transactions with the trained model — no API or database needed.

This is the quickest way to see prediction output. It pulls a few real rows
out of the dataset (some fraud, some legit), runs them through the saved ANN +
scaler + tuned threshold, and prints the fraud probability and decision next to
the true label so you can see the model working.

Usage
-----
    # score a handful of sampled rows from the training CSV
    python scripts/predict_example.py

    # score every row of your own CSV (same columns, Class optional)
    python scripts/predict_example.py --csv data/new_transactions.csv --limit 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from api.predict import FraudPredictor
from api.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="data/creditcard.csv",
        help="CSV of transactions to score (must have Time, V1..V28, Amount).",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="How many rows to score."
    )
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    has_label = "Class" in df.columns

    # If the file has labels, sample a mix of fraud + legit so the demo is
    # interesting; otherwise just take the first N rows.
    if has_label:
        frauds = df[df.Class == 1].head(max(1, args.limit // 2))
        legit = df[df.Class == 0].head(args.limit - len(frauds))
        sample = pd.concat([frauds, legit]).sample(frac=1, random_state=0)
        labels = sample["Class"].tolist()
        sample = sample.drop(columns=["Class"])
    else:
        sample = df.head(args.limit)
        labels = [None] * len(sample)

    predictor = FraudPredictor(get_settings())
    feature_dicts = sample.to_dict(orient="records")
    probs = predictor.score_many(feature_dicts)

    print(f"\nDecision threshold: {predictor.threshold:.4f}\n")
    header = f"{'#':>3}  {'P(fraud)':>9}  {'predicted':>9}  {'actual':>7}  {'Amount':>10}"
    print(header)
    print("-" * len(header))
    for i, (prob, row, actual) in enumerate(zip(probs, feature_dicts, labels)):
        pred = "FRAUD" if predictor.decide(prob) else "legit"
        actual_str = "-" if actual is None else ("FRAUD" if actual == 1 else "legit")
        print(
            f"{i:>3}  {prob:>9.4f}  {pred:>9}  {actual_str:>7}  {row['Amount']:>10.2f}"
        )


if __name__ == "__main__":
    main()
