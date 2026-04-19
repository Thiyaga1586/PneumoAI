from __future__ import annotations

import sqlite3
from pathlib import Path

from pneumoai.common.settings import settings


_DB_INITIALIZED = False


def _db_path() -> Path:
    path = Path(settings.sqlite_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def init_db(force: bool = False) -> None:
    global _DB_INITIALIZED

    if _DB_INITIALIZED and not force:
        return

    db_path = _db_path()
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            request_id TEXT PRIMARY KEY,
            model_version TEXT NOT NULL,
            prediction TEXT NOT NULL,
            probability REAL NOT NULL,
            threshold REAL NOT NULL,
            latency_ms REAL NOT NULL,
            true_label TEXT,
            created_at TEXT NOT NULL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            request_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            submitted_at TEXT NOT NULL,
            image_uri TEXT NOT NULL,
            true_label TEXT
        )
        """)

        conn.commit()
        _DB_INITIALIZED = True
    finally:
        conn.close()


def get_connection() -> sqlite3.Connection:
    init_db()
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn