from fastapi import APIRouter, Header, HTTPException

from pneumoai.common.settings import settings
from pneumoai.models.registry import (
    get_registry,
    promote_version,
    rollback_version,
)

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(x_api_key: str | None) -> None:
    if x_api_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/registry")
def admin_registry(x_api_key: str | None = Header(default=None)):
    _require_admin(x_api_key)
    return get_registry()


@router.post("/promote/{version}")
def admin_promote(version: str, x_api_key: str | None = Header(default=None)):
    _require_admin(x_api_key)
    try:
        return promote_version(version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/rollback")
def admin_rollback(x_api_key: str | None = Header(default=None)):
    _require_admin(x_api_key)
    try:
        return rollback_version()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc