from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from pneumoai.common.logging import configure_logging
from pneumoai.common.settings import settings
from pneumoai.models.loader import validate_model_artifacts
from pneumoai.models.registry import get_current_version
from pneumoai.serving.dispatcher.inference_service import _get_model_bundle
from pneumoai.storage.sqlite import init_db
from pneumoai.serving.api.routes_admin import router as admin_router
from pneumoai.serving.api.routes_async import router as async_router
from pneumoai.serving.api.routes_health import router as health_router
from pneumoai.serving.api.routes_predict import router as predict_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    init_db()

    current_version = get_current_version()
    validate_model_artifacts(current_version)

    bundle = _get_model_bundle(current_version)
    logger.info(
        "model_warmed_on_startup",
        extra={
            "request_id": "startup",
            "model_version": bundle["version"],
            "backend": settings.inference_backend,
        },
    )

    yield


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(async_router)
app.include_router(admin_router)