from typing import Optional

from pneumoai.serving.dispatcher.task_store import get_task
from pneumoai.storage.sqlite import get_connection


def set_result(request_id: str, payload: dict) -> None:
    conn = get_connection()
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
    conn.close()


def get_result(request_id: str) -> Optional[dict]:
    task = get_task(request_id)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT request_id, model_version, prediction, probability, threshold, latency_ms, true_label, created_at
        FROM predictions
        WHERE request_id = ?
    """, (request_id,))
    row = cursor.fetchone()
    conn.close()

    if row is not None:
        return {
            "request_id": row[0],
            "status": "completed",
            "model_version": row[1],
            "prediction": row[2],
            "probability": row[3],
            "threshold": row[4],
            "latency_ms": row[5],
            "true_label": row[6],
            "created_at": row[7],
        }

    return task


def clear_results() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()