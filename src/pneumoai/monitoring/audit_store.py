from datetime import datetime, UTC
from pneumoai.storage.sqlite import get_connection


def log_prediction(
    request_id: str,
    model_version: str,
    prediction: str,
    probability: float,
    threshold: float,
    latency_ms: float,
    true_label: str | None,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO predictions
        (request_id, model_version, prediction, probability, threshold, latency_ms, true_label, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        request_id,
        model_version,
        prediction,
        probability,
        threshold,
        latency_ms,
        true_label,
        datetime.now(UTC).isoformat(),
    ))
    conn.commit()
    conn.close()