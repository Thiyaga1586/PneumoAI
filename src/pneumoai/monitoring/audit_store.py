from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text

from pneumoai.storage.sqlite import get_engine, init_db


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

    engine = get_engine()
    created_at = datetime.now(UTC)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO predictions
                (
                    request_id,
                    model_version,
                    prediction,
                    probability,
                    threshold,
                    latency_ms,
                    true_label,
                    created_at
                )
                VALUES
                (
                    :request_id,
                    :model_version,
                    :prediction,
                    :probability,
                    :threshold,
                    :latency_ms,
                    :true_label,
                    :created_at
                )
                ON CONFLICT (request_id)
                DO UPDATE SET
                    model_version = EXCLUDED.model_version,
                    prediction = EXCLUDED.prediction,
                    probability = EXCLUDED.probability,
                    threshold = EXCLUDED.threshold,
                    latency_ms = EXCLUDED.latency_ms,
                    true_label = EXCLUDED.true_label,
                    created_at = EXCLUDED.created_at
                """
            ),
            {
                "request_id": request_id,
                "model_version": model_version,
                "prediction": prediction,
                "probability": float(probability),
                "threshold": float(threshold),
                "latency_ms": float(latency_ms),
                "true_label": true_label,
                "created_at": created_at,
            },
        )