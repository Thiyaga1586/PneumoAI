import time
from typing import Optional

import torch

from pneumoai.common.settings import settings
from pneumoai.models.loader import load_model_bundle, resolve_device
from pneumoai.preprocessing.image import read_image_bytes
from pneumoai.serving.triton.client import TritonInferenceClient


def run_local_inference(image_uri: str, requested_version: Optional[str] = None) -> dict:
    with open(image_uri, "rb") as f:
        raw = f.read()

    image_array = read_image_bytes(raw)
    device = resolve_device()

    start = time.perf_counter()
    model, version, threshold, metadata = load_model_bundle(requested_version)

    with torch.no_grad():
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
    client = TritonInferenceClient(settings.triton_url)

    start = time.perf_counter()
    result = client.predict(image_uri=image_uri, requested_version=requested_version)
    latency_ms = (time.perf_counter() - start) * 1000.0

    threshold = 0.5
    probability = result["probability"]
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