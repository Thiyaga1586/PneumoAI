from __future__ import annotations

import logging

import mlflow
from fastapi import APIRouter, Header, HTTPException, Query

from pneumoai.common.settings import settings
from pneumoai.mlops.mlflow_registry import configure_mlflow
from pneumoai.models.registry import (
    get_registry,
    promote_version,
    rollback_version,
)
from pneumoai.monitoring.metrics import ADMIN_ACTIONS_TOTAL

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)


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
    configure_mlflow()
    return {
        "tracking_uri": settings.mlflow_tracking_uri,
        "experiment_name": settings.mlflow_experiment_name,
    }


@router.post("/promote/{version}")
def admin_promote(
    version: str,
    run_id: str | None = Query(default=None),
    notes: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None),
):
    _require_admin(x_api_key)
    try:
        result = promote_version(
            version,
            run_id=run_id,
            notes=notes,
            promoted_by="admin_api",
        )

        if run_id:
            configure_mlflow()
            try:
                mlflow.set_tag("promotion.version", version)
                mlflow.set_tag("promotion.notes", notes or "")
                mlflow.set_tag("promotion.source", "admin_api")
            except Exception:
                logger.exception("mlflow_tag_update_failed", extra={"request_id": f"promote-{version}"})

        ADMIN_ACTIONS_TOTAL.labels(action="promote").inc()
        logger.info("model_promoted", extra={"request_id": f"promote-{version}"})
        return result

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
        ADMIN_ACTIONS_TOTAL.labels(action="rollback").inc()
        logger.info("model_rolled_back", extra={"request_id": "rollback"})
        return result

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc