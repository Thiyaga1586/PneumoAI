import logging

from fastapi import APIRouter, Header, HTTPException

from pneumoai.common.settings import settings
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


@router.post("/promote/{version}")
def admin_promote(version: str, x_api_key: str | None = Header(default=None)):
    _require_admin(x_api_key)
    try:
        result = promote_version(version)
        ADMIN_ACTIONS_TOTAL.labels(action="promote").inc()
        logger.info("model_promoted", extra={"request_id": f"promote-{version}"})
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/rollback")
def admin_rollback(x_api_key: str | None = Header(default=None)):
    _require_admin(x_api_key)
    try:
        result = rollback_version()
        ADMIN_ACTIONS_TOTAL.labels(action="rollback").inc()
        logger.info("model_rolled_back", extra={"request_id": "rollback"})
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from Exception