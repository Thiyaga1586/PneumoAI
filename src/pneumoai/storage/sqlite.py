import sqlite3
from pathlib import Path

from pneumoai.common.settings import settings


def get_connection():
    db_path = Path(settings.sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            request_id TEXT PRIMARY KEY,
            model_version TEXT,
            prediction TEXT,
            probability REAL,
            threshold REAL,
            latency_ms REAL,
            true_label TEXT,
            created_at TEXT
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
    conn.close()