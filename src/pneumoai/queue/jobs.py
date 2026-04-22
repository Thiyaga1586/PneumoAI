from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Optional

from pneumoai.common.settings import settings
from pneumoai.queue.redis_client import get_redis_client, job_key


def enqueue_prediction_job(
    request_id: str,
    image_uri: str,
    true_label: Optional[str] = None,
) -> dict:
    client = get_redis_client()
    submitted_at = datetime.now(UTC).isoformat()

    payload = {
        "request_id": request_id,
        "status": "queued",
        "image_uri": image_uri,
        "true_label": true_label,
        "submitted_at": submitted_at,
    }

    client.hset(
        job_key(request_id),
        mapping={k: "" if v is None else str(v) for k, v in payload.items()},
    )
    client.rpush(settings.redis_queue_key, json.dumps(payload))

    return {
        "request_id": request_id,
        "status": "queued",
    }


def get_job_status(request_id: str) -> Optional[dict]:
    client = get_redis_client()
    data = client.hgetall(job_key(request_id))
    if not data:
        return None

    raw = dict(data)
    status = raw["status"]

    result: dict = {
        "request_id": raw["request_id"],
        "status": status,
    }

    if status == "queued":
        result["created_at"] = raw.get("submitted_at")

    elif status == "processing":
        result["created_at"] = raw.get("started_at")

    elif status == "completed":
        result.update(
            {
                "model_version": raw.get("model_version"),
                "prediction": raw.get("prediction"),
                "probability": float(raw["probability"]) if raw.get("probability") else None,
                "threshold": float(raw["threshold"]) if raw.get("threshold") else None,
                "latency_ms": float(raw["latency_ms"]) if raw.get("latency_ms") else None,
                "backend": raw.get("backend"),
                "true_label": raw.get("true_label") or None,
                "created_at": raw.get("completed_at"),
            }
        )

    elif status == "failed":
        result["error"] = raw.get("error")
        result["created_at"] = raw.get("completed_at")

    return result


def update_job_status(request_id: str, **fields) -> None:
    client = get_redis_client()
    serialized = {k: "" if v is None else str(v) for k, v in fields.items()}
    client.hset(job_key(request_id), mapping=serialized)