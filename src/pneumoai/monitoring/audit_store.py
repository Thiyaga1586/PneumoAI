from __future__ import annotations

from datetime import UTC, datetime

from pneumoai.storage.sqlite import get_connection, init_db


def log_prediction(
    request_id: str,
    model_version: str,
    prediction: str,
    probability: float,
    threshold: float,
    latency_ms: float,
    true_label: str | None,
) -> None:
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO predictions
            (request_id, model_version, prediction, probability, threshold, latency_ms, true_label, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                model_version,
                prediction,
                probability,
                threshold,
                latency_ms,
                true_label,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()