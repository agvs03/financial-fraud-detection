"""PostgreSQL access layer using SQLAlchemy Core.

The scoring endpoint pulls transaction feature rows from a ``transactions``
table by id. The table is expected to have one column per model feature
(``Time``, ``V1``..``V28``, ``Amount``) plus a primary-key ``id``.

A single Engine (with a connection pool) is created lazily and reused.
"""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import Row

from .config import get_settings


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,   # recycle dead connections transparently
        pool_size=5,
        max_overflow=10,
        future=True,
    )


def fetch_transaction(transaction_id: int) -> dict | None:
    """Return a single transaction's features as a dict, or None if absent."""
    settings = get_settings()
    cols = ", ".join(f'"{c}"' for c in settings.feature_order)
    query = text(
        f'SELECT {cols} FROM {settings.transactions_table} '
        f'WHERE id = :tid'
    )
    with get_engine().connect() as conn:
        row: Row | None = conn.execute(query, {"tid": transaction_id}).first()
    if row is None:
        return None
    return dict(row._mapping)


def fetch_transactions(transaction_ids: list[int]) -> dict[int, dict]:
    """Batch fetch. Returns {id: feature_dict} for the ids that exist."""
    if not transaction_ids:
        return {}
    settings = get_settings()
    cols = ", ".join(f'"{c}"' for c in settings.feature_order)
    query = text(
        f'SELECT id, {cols} FROM {settings.transactions_table} '
        f'WHERE id = ANY(:ids)'
    )
    with get_engine().connect() as conn:
        rows = conn.execute(query, {"ids": transaction_ids}).all()
    out: dict[int, dict] = {}
    for row in rows:
        mapping = dict(row._mapping)
        out[int(mapping.pop("id"))] = mapping
    return out


def healthcheck() -> bool:
    """Return True if the database answers a trivial query."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - surfaced as an unhealthy status
        return False
