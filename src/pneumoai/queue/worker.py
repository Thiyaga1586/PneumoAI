from __future__ import annotations

import json
import logging
import socket
import time
from datetime import UTC, datetime

from redis.exceptions import RedisError

from pneumoai.common.settings import settings
from pneumoai.monitoring.audit_store import log_prediction
from pneumoai.monitoring.metrics import ASYNC_COMPLETIONS_TOTAL
from pneumoai.queue.jobs import update_job_status
from pneumoai.queue.redis_client import get_redis_client
from pneumoai.serving.dispatcher.inference_service import run_inference

from pneumoai.monitoring.metrics import (
    ASYNC_COMPLETIONS_TOTAL,
    PREDICTION_ERRORS_TOTAL,
    PREDICTION_LATENCY_MS,
)

logger = logging.getLogger(__name__)


def process_one_job():
    client = get_redis_client()
    item = client.lpop(settings.redis_queue_key)

    if item is None:
        return None

    payload = json.loads(item)
    request_id = payload["request_id"]
    image_uri = payload["image_uri"]
    true_label = payload.get("true_label") or None

    update_job_status(
        request_id,
        status="processing",
        started_at=datetime.now(UTC).isoformat(),
    )

    try:
        inference = run_inference(image_uri=image_uri)
        latency_ms = float(inference["latency_ms"])
        PREDICTION_LATENCY_MS.observe(latency_ms)
        completed_at = datetime.now(UTC).isoformat()

        update_job_status(
            request_id,
            status="completed",
            completed_at=completed_at,
            model_version=inference["model_version"],
            prediction=inference["prediction"],
            probability=inference["probability"],
            threshold=inference["threshold"],
            latency_ms=latency_ms,
            backend=inference["backend"],
            true_label=true_label,
        )

        log_prediction(
            request_id=request_id,
            model_version=inference["model_version"],
            prediction=inference["prediction"],
            probability=float(inference["probability"]),
            threshold=float(inference["threshold"]),
            latency_ms=latency_ms,
            true_label=true_label,
        )

        ASYNC_COMPLETIONS_TOTAL.inc()
        logger.info("async_job_completed", extra={"request_id": request_id})

        return {
            "request_id": request_id,
            "status": "completed",
            "model_version": inference["model_version"],
            "prediction": inference["prediction"],
            "probability": float(inference["probability"]),
            "threshold": float(inference["threshold"]),
            "latency_ms": latency_ms,
            "backend": inference["backend"],
            "true_label": true_label,
        }

    except Exception as exc:
        PREDICTION_ERRORS_TOTAL.inc()
        completed_at = datetime.now(UTC).isoformat()

        update_job_status(
            request_id,
            status="failed",
            completed_at=completed_at,
            error=str(exc),
        )
        logger.exception("async_job_failed", extra={"request_id": request_id})

        return {
            "request_id": request_id,
            "status": "failed",
            "error": str(exc),
            "true_label": true_label,
        }


def run_worker_loop() -> None:
    logger.info("redis_worker_started")

    while True:
        try:
            processed = process_one_job()
            if processed is None:
                time.sleep(settings.worker_poll_interval_seconds)

        except (RedisError, socket.gaierror, OSError) as exc:
            logger.warning("redis_unavailable_retrying", extra={"error": str(exc)})
            time.sleep(max(settings.worker_poll_interval_seconds, 2.0))

        except Exception:
            logger.exception("worker_loop_unhandled_error")
            time.sleep(max(settings.worker_poll_interval_seconds, 2.0))