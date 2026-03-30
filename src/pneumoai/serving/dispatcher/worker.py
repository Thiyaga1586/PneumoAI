from datetime import UTC, datetime
import time
from typing import Optional

from pneumoai.models.loader import load_model_bundle
from pneumoai.serving.dispatcher.status_store import set_result
from pneumoai.serving.dispatcher.task_store import mark_task_completed, pop_next_queued_task
from pneumoai.preprocessing.image import read_image_bytes


def process_next_task() -> Optional[dict]:
    task = pop_next_queued_task()
    if task is None:
        return None

    request_id = task["request_id"]
    image_uri = task["image_uri"]
    true_label = task.get("true_label")

    with open(image_uri, "rb") as f:
        raw = f.read()

    image_array = read_image_bytes(raw)

    start = time.perf_counter()
    model, version, threshold, metadata = load_model_bundle()

    probability = float(model.predict_proba(image_array))
    prediction = "PNEUMONIA" if probability >= threshold else "NORMAL"
    latency_ms = (time.perf_counter() - start) * 1000.0

    result = {
        "request_id": request_id,
        "model_version": version,
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
        "latency_ms": latency_ms,
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