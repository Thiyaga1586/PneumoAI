from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from pneumoai.common.settings import settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set")

    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        future=True,
    )


def init_db() -> None:
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    request_id TEXT PRIMARY KEY,
                    model_version TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    probability DOUBLE PRECISION NOT NULL,
                    threshold DOUBLE PRECISION NOT NULL,
                    latency_ms DOUBLE PRECISION NOT NULL,
                    true_label TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_predictions_model_created
                ON predictions (model_version, created_at DESC)
                """
            )
        )


def get_connection():
    """
    Compatibility helper for older imports.
    Prefer get_engine() for new code.
    """
    return get_engine().connect()