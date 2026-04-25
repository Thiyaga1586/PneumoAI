from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Query

from pneumoai.common.settings import settings
from pneumoai.mlops.promotion_service import promote_with_gate
from pneumoai.models.registry import (
    get_registry,
    promote_version,
    rollback_version,
)
from pneumoai.monitoring.metrics import ADMIN_ACTIONS_TOTAL
from pneumoai.serving.dispatcher.inference_service import clear_model_cache

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)

def _try_tag_mlflow_promotion(
    *,
    run_id: str | None,
    version: str,
    notes: str | None,
    mode: str,
) -> None:
    if not run_id:
        return

    try:
        import mlflow
        from pneumoai.mlops.mlflow_registry import configure_mlflow

        configure_mlflow()
        mlflow.set_tag("promotion.version", version)
        mlflow.set_tag("promotion.notes", notes or "")
        mlflow.set_tag("promotion.source", "admin_api")
        mlflow.set_tag("promotion.mode", mode)
    except Exception:
        logger.exception(
            "mlflow_tag_update_failed",
            extra={"request_id": f"promote-{version}"},
        )


def _require_admin(x_api_key: str | None) -> None:
    if x_api_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/registry")
def admin_registry(x_api_key: str | None = Header(default=None)):
    _require_admin(x_api_key)
    ADMIN_ACTIONS_TOTAL.labels(action="registry").inc()
    return get_registry()


@router.get("/mlflow")
def admin_mlflow_info(x_api_key: str | None = Header(default=None)):
    _require_admin(x_api_key)
    return {
        "tracking_uri": settings.mlflow_tracking_uri,
        "experiment_name": settings.mlflow_experiment_name,
        "runtime_mlflow_optional": True,
    }


@router.post("/promote/{version}")
def admin_promote(
    version: str,
    run_id: str | None = Query(default=None),
    notes: str | None = Query(default=None),
    candidate_metrics_path: str | None = Query(default=None),
    champion_metrics_path: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None),
):
    _require_admin(x_api_key)

    gated_mode = candidate_metrics_path is not None or champion_metrics_path is not None
    if gated_mode and not (candidate_metrics_path and champion_metrics_path):
        raise HTTPException(
            status_code=400,
            detail="Both candidate_metrics_path and champion_metrics_path are required for gated promotion",
        )

    try:
        if gated_mode:
            result = promote_with_gate(
                version=version,
                candidate_metrics_path=candidate_metrics_path,
                champion_metrics_path=champion_metrics_path,
                run_id=run_id,
                notes=notes,
                promoted_by="admin_api",
            )
            clear_model_cache()
            ADMIN_ACTIONS_TOTAL.labels(action="promote_gated").inc()
        else:
            result = promote_version(
                version,
                run_id=run_id,
                notes=notes,
                promoted_by="admin_api",
            )
            clear_model_cache()
            ADMIN_ACTIONS_TOTAL.labels(action="promote").inc()

        _try_tag_mlflow_promotion(
            run_id=run_id,
            version=version,
            notes=notes,
            mode="gated" if gated_mode else "manual",
        )

        logger.info(
            "model_promoted",
            extra={
                "request_id": f"promote-{version}",
                "promotion_mode": "gated" if gated_mode else "manual",
            },
        )
        return result

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/rollback")
def admin_rollback(
    notes: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None),
):
    _require_admin(x_api_key)
    try:
        result = rollback_version(
            notes=notes,
            rolled_back_by="admin_api",
        )
        clear_model_cache()
        ADMIN_ACTIONS_TOTAL.labels(action="rollback").inc()
        logger.info("model_rolled_back", extra={"request_id": "rollback"})
        return result

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from Exception