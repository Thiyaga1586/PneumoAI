from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from pneumoai.common.ids import generate_request_id
from pneumoai.common.settings import settings
from pneumoai.contracts.api import PredictResponse
from pneumoai.monitoring.audit_store import log_prediction
from pneumoai.monitoring.metrics import (
    PREDICTION_ERRORS_TOTAL,
    PREDICTION_LATENCY_MS,
    PREDICTION_REQUESTS_TOTAL,
)
from pneumoai.preprocessing.validation import validate_upload
from pneumoai.serving.dispatcher.inference_service import run_inference

router = APIRouter()
logger = logging.getLogger(__name__)


def _runtime_upload_dir() -> Path:
    path = Path(settings.runtime_dir) / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


@router.post("/predict-sync", response_model=PredictResponse)
async def predict_sync(
    file: UploadFile = File(...),
    true_label: Optional[str] = Form(default=None),
):
    PREDICTION_REQUESTS_TOTAL.inc()

    await validate_upload(file)
    raw = await file.read()

    request_id = generate_request_id()
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    target_path = _runtime_upload_dir() / f"{request_id}{suffix}"

    with open(target_path, "wb") as f:
        f.write(raw)

    try:
        inference = run_inference(image_uri=str(target_path))
        latency_ms = float(inference["latency_ms"])
        PREDICTION_LATENCY_MS.observe(latency_ms)

        log_prediction(
            request_id=request_id,
            model_version=inference["model_version"],
            prediction=inference["prediction"],
            probability=float(inference["probability"]),
            threshold=float(inference["threshold"]),
            latency_ms=latency_ms,
            true_label=true_label,
        )

        logger.info(
            "prediction_completed",
            extra={"request_id": request_id},
        )

        return PredictResponse(
            request_id=request_id,
            status="completed",
            model_version=inference["model_version"],
            prediction=inference["prediction"],
            probability=float(inference["probability"]),
            threshold=float(inference["threshold"]),
            latency_ms=latency_ms,
            true_label=true_label,
        )

    except FileNotFoundError as exc:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    except ValueError as exc:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        PREDICTION_ERRORS_TOTAL.inc()
        logger.exception(
            "prediction_failed",
            extra={"request_id": request_id},
        )
        raise HTTPException(status_code=500, detail="Prediction failed") from exc