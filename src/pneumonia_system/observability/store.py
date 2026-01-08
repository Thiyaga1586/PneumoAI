import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "runtime" / "requests.db"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH))

def init_db(db_path: str = "requests.db"):
    global _DB_PATH
    _DB_PATH = str(db_path)

    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT,
            model_version TEXT,
            latency_ms REAL,
            label TEXT,
            probability REAL,
            hist_json TEXT,
            error TEXT,
            true_label TEXT
        )
        """
    )

    conn.commit()
    conn.close()

def log_request(
    ts_utc: str,
    model_version: str,
    latency_ms: float,
    label: str,
    probability: float,
    hist: Optional[List[float]],
    error: Optional[str],
    true_label: Optional[str] = None,
) -> None:
    hist_json = json.dumps(hist) if hist is not None else None

    with _connect() as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO requests (ts_utc, model_version, latency_ms, label, probability, hist_json, error, true_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts_utc, model_version, float(latency_ms), label, float(probability), hist_json, error, true_label),
        )
        con.commit()


def recent_requests(n: int = 10) -> List[Tuple]:
    with _connect() as con:
        cur = con.cursor()
        cur.execute(
            """
            SELECT ts_utc, model_version, latency_ms, label, probability, hist_json, error, true_label
            FROM requests
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(n),),
        )
        return cur.fetchall()


def latest_rows(limit: int = 200, model_version: Optional[str] = None) -> List[Dict[str, Any]]:
    with _connect() as con:
        cur = con.cursor()

        if model_version:
            cur.execute(
                """
                SELECT id, ts_utc, model_version, latency_ms, label, probability, error, true_label
                FROM requests
                WHERE model_version = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (model_version, int(limit)),
            )
        else:
            cur.execute(
                """
                SELECT id, ts_utc, model_version, latency_ms, label, probability, error, true_label
                FROM requests
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            )

        rows = []
        for r in cur.fetchall():
            prob = float(r[5])
            if r[6] is not None:
                band = "error"
            elif prob >= 0.8:
                band = "high_positive"
            elif prob >= 0.5:
                band = "borderline_positive"
            elif prob >= 0.2:
                band = "borderline_negative"
            else:
                band = "high_negative"

            rows.append(
                {
                    "id": int(r[0]),
                    "ts_utc": r[1],
                    "model_version": r[2],
                    "latency_ms": float(r[3]),
                    "pred_label": r[4],
                    "probability": prob,
                    "error": r[6],
                    "true_label": r[7],
                    "band": band,
                }
            )
        return rows

