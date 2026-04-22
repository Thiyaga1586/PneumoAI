from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from pneumoai.common.ids import generate_request_id
from pneumoai.common.settings import settings
from pneumoai.monitoring.metrics import ASYNC_REQUESTS_TOTAL
from pneumoai.preprocessing.validation import validate_upload
from pneumoai.queue.jobs import enqueue_prediction_job, get_job_status

router = APIRouter()


def _runtime_upload_dir() -> Path:
    path = Path(settings.runtime_dir) / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


@router.post("/predict")
async def create_async_prediction(
    file: UploadFile = File(...),
    true_label: Optional[str] = Form(default=None),
):
    await validate_upload(file)

    request_id = generate_request_id()
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    target_path = _runtime_upload_dir() / f"{request_id}{suffix}"

    raw = await file.read()
    with open(target_path, "wb") as f:
        f.write(raw)

    ASYNC_REQUESTS_TOTAL.inc()

    return enqueue_prediction_job(
        request_id=request_id,
        image_uri=str(target_path),
        true_label=true_label,
    )


@router.get("/predict/{request_id}")
def get_async_prediction_status(request_id: str):
    job = get_job_status(request_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Request not found")
    return job