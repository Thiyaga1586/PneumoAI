import time
from typing import Optional

import torch
from fastapi import APIRouter, UploadFile, File, Form

from pneumoai.common.ids import generate_request_id
from pneumoai.contracts.api import PredictResponse
from pneumoai.preprocessing.validation import validate_upload
from pneumoai.preprocessing.image import read_image_bytes
from pneumoai.models.loader import load_model_bundle
from pneumoai.monitoring.audit_store import log_prediction

router = APIRouter()


@router.post("/predict-sync", response_model=PredictResponse)
async def predict_sync(
    file: UploadFile = File(...),
    true_label: Optional[str] = Form(default=None),
):
    await validate_upload(file)
    raw = await file.read()
    image_array = read_image_bytes(raw)

    request_id = generate_request_id()

    start = time.perf_counter()
    model, version, threshold, metadata = load_model_bundle()

    with torch.no_grad():
        tensor = torch.tensor(image_array, dtype=torch.float32)
        logits = model(tensor)
        probability = float(torch.sigmoid(logits).squeeze().item())

    prediction = "PNEUMONIA" if probability >= threshold else "NORMAL"
    latency_ms = (time.perf_counter() - start) * 1000.0

    log_prediction(
        request_id=request_id,
        model_version=version,
        prediction=prediction,
        probability=probability,
        threshold=threshold,
        latency_ms=latency_ms,
        true_label=true_label,
    )

    return PredictResponse(
        request_id=request_id,
        status="completed",
        model_version=version,
        prediction=prediction,
        probability=probability,
        threshold=threshold,
        latency_ms=latency_ms,
        true_label=true_label,
    )