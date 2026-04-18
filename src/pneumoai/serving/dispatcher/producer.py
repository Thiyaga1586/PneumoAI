from datetime import UTC, datetime
from typing import Optional

from pneumoai.contracts.events import PredictionTask
from pneumoai.serving.dispatcher.task_store import add_task


class LocalPredictionProducer:
    def publish(
        self,
        request_id: str,
        image_uri: str,
        true_label: Optional[str] = None,
        requested_model_version: Optional[str] = None,
    ) -> PredictionTask:
        task = PredictionTask(
            request_id=request_id,
            created_at=datetime.now(UTC),
            image_uri=image_uri,
            true_label=true_label,
            requested_model_version=requested_model_version,
        )

        task_payload = {
            "request_id": request_id,
            "status": "queued",
            "submitted_at": task.created_at.isoformat(),
            "image_uri": image_uri,
            "true_label": true_label,
        }

        add_task(request_id, task_payload)

        return task