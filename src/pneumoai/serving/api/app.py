from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from pneumoai.common.logging import configure_logging
from pneumoai.common.settings import settings
from pneumoai.models.loader import validate_model_artifacts
from pneumoai.models.registry import get_current_version
from pneumoai.monitoring.metrics import render_metrics
from pneumoai.serving.api.routes_admin import router as admin_router
from pneumoai.serving.api.routes_async import router as async_router
from pneumoai.serving.api.routes_health import router as health_router
from pneumoai.serving.api.routes_predict import router as predict_router
from pneumoai.serving.dispatcher.inference_service import _get_model_bundle
from pneumoai.storage.sqlite import init_db

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


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    start = time.perf_counter()

    logger.info(
        "request_started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"

    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(elapsed_ms, 2),
        },
    )

    return response


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


app.include_router(health_router)
app.include_router(predict_router)
app.include_router(async_router)
app.include_router(admin_router)


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "status": "running",
        "environment": settings.env,
        "model_version": get_current_version(),
        "backend": settings.inference_backend,
    }