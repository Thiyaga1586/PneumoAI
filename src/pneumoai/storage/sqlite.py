from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

from pneumoai.common.settings import settings


def _effective_database_url() -> str:
    """
    Prefer DATABASE_URL when present.
    Fall back to the local runtime SQLite file for CI/local/dev.
    """
    return settings.database_url or f"sqlite:///{settings.sqlite_path}"


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    url = make_url(database_url)
    database = url.database

    if not database or database == ":memory:":
        return

    db_path = Path(database)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    database_url = _effective_database_url()

    if database_url.startswith("sqlite"):
        _ensure_sqlite_parent_dir(database_url)
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            future=True,
        )

    return create_engine(
        database_url,
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
                    probability REAL NOT NULL,
                    threshold REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    true_label TEXT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
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