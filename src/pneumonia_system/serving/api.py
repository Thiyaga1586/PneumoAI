import time
from datetime import datetime, timezone
from typing import Dict, Any

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image

from .admin import create_admin_router
from ..observability.metrics import record_latency, record_error, p95, latencies
from ..model.loader import load_model
from ..model.preprocess import preprocess_image, extract_intensity_histogram
from ..observability.store import log_request, init_db

app = FastAPI(title="Pneumonia ML Inference API", version="0.3")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
MODEL_VERSION = "unknown"


def _reload_model():
    global MODEL, MODEL_VERSION
    MODEL, MODEL_VERSION = load_model(device=device)


def _served_version() -> str:
    return MODEL_VERSION


app.include_router(
    create_admin_router(reload_model=_reload_model, get_served_version=_served_version),
    prefix="/admin",
)

@app.get("/")
def root():
    return {
        "message": "Pneumonia Inference API is running",
        "docs": "/docs",
        "health": "/health",
        "admin": "/admin/status"
    }



@app.on_event("startup")
def startup_event():
    init_db()
    _reload_model()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": str(device),
        "model_version": MODEL_VERSION,
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    ls = list(latencies)
    return {
        "request_count_in_memory": len(ls),
        "p95_latency_ms": round(p95(ls), 3),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    true_label: str = Form(None) 
):
    ts = datetime.now(timezone.utc).isoformat()

    try:
        if MODEL is None:
            _reload_model()

        img = Image.open(file.file)
        x, gray_np = preprocess_image(img, device=device)

        start = time.perf_counter()
        with torch.no_grad():
            logits = MODEL(x)
            prob = torch.sigmoid(logits).item()
        latency_ms = (time.perf_counter() - start) * 1000.0

        record_latency(latency_ms)

        label = "Pneumonia" if prob >= 0.5 else "Normal"
        hist = extract_intensity_histogram(gray_np, bins=32)

        log_request(
            ts_utc=ts,
            model_version=MODEL_VERSION,
            latency_ms=latency_ms,
            label=label,
            probability=prob,
            hist=hist,
            error=None,
            true_label=true_label,
        )


        return JSONResponse(
            {
                "label": label,
                "probability": round(prob, 6),
                "latency_ms": round(latency_ms, 3),
                "model_version": MODEL_VERSION,
            }
        )

    except Exception as e:
        record_error()

        log_request(
            ts_utc=ts,
            model_version=MODEL_VERSION or "unknown",
            latency_ms=0.0,
            label="error",
            probability=0.0,
            hist=None,
            error=str(e),
        )
        return JSONResponse({"error": str(e)}, status_code=500)
