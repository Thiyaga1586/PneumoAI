from fastapi import APIRouter, Query

from pneumoai.common.settings import settings
from pneumoai.models.loader import resolve_device, validate_model_artifacts
from pneumoai.monitoring.drift import detect_prediction_drift
from pneumoai.storage.sqlite import init_db

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    init_db()
    validate_model_artifacts(settings.default_model_version)
    return {
        "status": "ready",
        "model_version": settings.default_model_version,
        "device": resolve_device(),
        "backend": settings.inference_backend,
    }


@router.get("/drift")
def drift(
    version: str = Query(default=settings.default_model_version),
    limit: int = Query(default=500, ge=20, le=5000),
    threshold: float = Query(default=0.08, gt=0.0, lt=1.0),
):
    return detect_prediction_drift(
        version=version,
        limit=limit,
        threshold=threshold,
    )