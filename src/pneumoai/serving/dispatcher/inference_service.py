from __future__ import annotations

import threading
import time
from typing import Optional

import torch

from pneumoai.common.settings import settings
from pneumoai.models.loader import load_model_bundle, resolve_device
from pneumoai.models.registry import get_current_version
from pneumoai.preprocessing.image import read_image_bytes


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, object] = {}


def clear_model_cache() -> None:
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()


def _get_model_bundle(requested_version: Optional[str] = None):
    version = requested_version or get_current_version()

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(version)
        if cached is not None:
            return cached

        model, resolved_version, threshold, metadata = load_model_bundle(version)
        bundle = {
            "model": model,
            "version": resolved_version,
            "threshold": threshold,
            "metadata": metadata,
            "device": resolve_device(),
        }
        _MODEL_CACHE[resolved_version] = bundle
        return bundle


def run_local_inference(image_uri: str, requested_version: Optional[str] = None) -> dict:
    with open(image_uri, "rb") as f:
        raw = f.read()

    image_array = read_image_bytes(raw)
    bundle = _get_model_bundle(requested_version)

    model = bundle["model"]
    version = bundle["version"]
    threshold = float(bundle["threshold"])
    device = bundle["device"]

    start = time.perf_counter()

    with torch.inference_mode():
        tensor = torch.tensor(
            image_array,
            dtype=torch.float32,
            device=device,
        )
        logits = model(tensor)
        probability = float(torch.sigmoid(logits).squeeze().item())

    latency_ms = (time.perf_counter() - start) * 1000.0
    prediction = "PNEUMONIA" if probability >= threshold else "NORMAL"

    return {
        "model_version": version,
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
        "latency_ms": latency_ms,
        "backend": "local",
    }


def run_triton_inference(image_uri: str, requested_version: Optional[str] = None) -> dict:
    from pneumoai.serving.triton.client import TritonInferenceClient

    client = TritonInferenceClient(settings.triton_url)

    start = time.perf_counter()
    result = client.predict(image_uri=image_uri, requested_version=requested_version)
    latency_ms = (time.perf_counter() - start) * 1000.0

    threshold = 0.5
    probability = float(result["probability"])
    prediction = "PNEUMONIA" if probability >= threshold else "NORMAL"

    return {
        "model_version": requested_version or "1",
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
        "latency_ms": latency_ms,
        "backend": "triton",
    }


def run_inference(image_uri: str, requested_version: Optional[str] = None) -> dict:
    if settings.inference_backend == "triton":
        try:
            return run_triton_inference(
                image_uri=image_uri,
                requested_version=requested_version,
            )
        except Exception:
            if not getattr(settings, "triton_fallback_to_local", False):
                raise
            return run_local_inference(
                image_uri=image_uri,
                requested_version=requested_version,
            )

    return run_local_inference(
        image_uri=image_uri,
        requested_version=requested_version,
    )