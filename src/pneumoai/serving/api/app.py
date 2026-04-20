from contextlib import asynccontextmanager

from fastapi import FastAPI

from pneumoai.common.logging import configure_logging
from pneumoai.common.settings import settings
from pneumoai.storage.sqlite import init_db
from pneumoai.serving.api.routes_async import router as async_router
from pneumoai.serving.api.routes_health import router as health_router
from pneumoai.serving.api.routes_predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(async_router)