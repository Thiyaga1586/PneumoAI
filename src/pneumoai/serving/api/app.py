from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from pneumoai.monitoring.metrics import render_metrics

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

    logger.info(
        "application_startup",
        extra={
            "request_id": "startup",
            "environment": settings.env,
            "backend": settings.inference_backend,
        },
    )

    init_db()

    current_version = get_current_version()

    logger.info(
        "validating_model_artifacts",
        extra={
            "request_id": "startup",
            "model_version": current_version,
        },
    )

    validate_model_artifacts(current_version)

    bundle = _get_model_bundle(current_version)

    logger.info(
        "model_warmed_on_startup",
        extra={
            "request_id": "startup",
            "model_version": bundle["version"],
            "backend": settings.inference_backend,
            "device": bundle["device"],
        },
    )

    yield

    logger.info(
        "application_shutdown",
        extra={
            "request_id": "shutdown",
        },
    )


app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


allowed_origins = [
    origin.strip()
    for origin in settings.cors_origins.split(",")
    if origin.strip()
]

if not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

logger.info(
    "metrics_endpoint_enabled",
    extra={
        "request_id": "startup",
        "path": "/metrics",
    },
)


app.include_router(health_router)
app.include_router(predict_router)
app.include_router(async_router)
app.include_router(admin_router)

@app.get("/metrics")
def metrics():
    payload, content_type = render_metrics()

    return Response(
        content=payload,
        media_type=content_type,
    )

@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "status": "running",
        "environment": settings.env,
        "model_version": get_current_version(),
        "backend": settings.inference_backend,
    }