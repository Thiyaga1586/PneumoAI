from typing import Optional

from pneumoai.serving.dispatcher.task_store import get_task
from pneumoai.storage.sqlite import get_connection, init_db


def set_result(request_id: str, payload: dict) -> None:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO predictions
            (request_id, model_version, prediction, probability, threshold, latency_ms, true_label, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request_id,
            payload.get("model_version"),
            payload.get("prediction"),
            payload.get("probability"),
            payload.get("threshold"),
            payload.get("latency_ms"),
            payload.get("true_label"),
            payload.get("created_at"),
        ))
        conn.commit()
    finally:
        conn.close()


def get_result(request_id: str) -> Optional[dict]:
    init_db()
    task = get_task(request_id)

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT request_id, model_version, prediction, probability, threshold, latency_ms, true_label, created_at
            FROM predictions
            WHERE request_id = ?
        """, (request_id,))
        row = cursor.fetchone()
    finally:
        conn.close()

    if row is not None:
        return {
            "request_id": row["request_id"],
            "status": "completed",
            "model_version": row["model_version"],
            "prediction": row["prediction"],
            "probability": row["probability"],
            "threshold": row["threshold"],
            "latency_ms": row["latency_ms"],
            "true_label": row["true_label"],
            "created_at": row["created_at"],
        }

    return task


def clear_results() -> None:
    init_db(force=True)
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM tasks")
        conn.commit()
    finally:
        conn.close()