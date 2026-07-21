"""Load the Kaggle creditcard.csv into the PostgreSQL ``transactions`` table.

Usage
-----
    python -m db.load_csv --csv data/creditcard.csv

Reads the connection URL from the same env var the API uses (FRAUD_DATABASE_URL).
The CSV's ``Class`` column, if present, is stored in ``label``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from api.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/creditcard.csv")
    parser.add_argument("--chunksize", type=int, default=10_000)
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    settings = get_settings()
    engine = create_engine(settings.database_url, future=True)

    df = pd.read_csv(csv_path)
    if "Class" in df.columns:
        df = df.rename(columns={"Class": "label"})

    print(f"Loading {len(df):,} rows into 'transactions' ...")
    df.to_sql(
        settings.transactions_table,
        engine,
        if_exists="append",
        index=False,
        chunksize=args.chunksize,
        method="multi",
    )
    print("Done.")


if __name__ == "__main__":
    main()
