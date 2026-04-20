from fastapi import APIRouter, HTTPException, Query, Response

from pneumoai.common.settings import settings
from pneumoai.models.loader import resolve_device, validate_model_artifacts
from pneumoai.monitoring.drift import detect_prediction_drift
from pneumoai.monitoring.metrics import DRIFT_CHECKS_TOTAL, DRIFT_SCORE, render_metrics
from pneumoai.storage.sqlite import init_db

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    try:
        init_db()
        validate_model_artifacts(settings.default_model_version)
        return {
            "status": "ready",
            "model_version": settings.default_model_version,
            "device": resolve_device(),
            "backend": settings.inference_backend,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/drift")
def drift(
    version: str = Query(default=settings.default_model_version),
    limit: int = Query(default=500, ge=20, le=5000),
    threshold: float = Query(default=0.08, gt=0.0, lt=1.0),
):
    try:
        result = detect_prediction_drift(
            version=version,
            limit=limit,
            threshold=threshold,
        )
        DRIFT_CHECKS_TOTAL.inc()
        if result.get("js_divergence") is not None:
            DRIFT_SCORE.observe(result["js_divergence"])
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Drift detection failed") from exc


@router.get("/metrics")
def metrics():
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)