from datetime import UTC, datetime
from typing import Optional

from pneumoai.serving.dispatcher.inference_service import run_inference
from pneumoai.serving.dispatcher.status_store import set_result
from pneumoai.serving.dispatcher.task_store import mark_task_completed, pop_next_queued_task


def process_next_task() -> Optional[dict]:
    task = pop_next_queued_task()
    if task is None:
        return None

    request_id = task["request_id"]
    image_uri = task["image_uri"]
    true_label = task.get("true_label")

    inference = run_inference(image_uri=image_uri)

    result = {
        "request_id": request_id,
        "model_version": inference["model_version"],
        "prediction": inference["prediction"],
        "probability": inference["probability"],
        "threshold": inference["threshold"],
        "latency_ms": inference["latency_ms"],
        "true_label": true_label,
        "created_at": datetime.now(UTC).isoformat(),
    }

    set_result(request_id, result)
    mark_task_completed(request_id)

    return {
        "request_id": request_id,
        "status": "completed",
        **result,
    }