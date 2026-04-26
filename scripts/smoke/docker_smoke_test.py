from __future__ import annotations

import io
import os
import time

import requests
from PIL import Image

BASE = os.getenv("PNEUMOAI_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
EXPECTED_MODEL_VERSION = os.getenv("PNEUMOAI_EXPECTED_MODEL_VERSION", "v2")
MAX_SYNC_LATENCY_MS = float(os.getenv("PNEUMOAI_MAX_SYNC_LATENCY_MS", "1500"))
MAX_ASYNC_LATENCY_MS = float(os.getenv("PNEUMOAI_MAX_ASYNC_LATENCY_MS", "1500"))


def make_test_image() -> tuple[str, bytes, str]:
    img = Image.new("L", (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "test.png", buf.getvalue(), "image/png"


def wait_ready(timeout_s: int = 60) -> dict:
    deadline = time.time() + timeout_s
    last_observation = "no request attempted"

    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE}/ready", timeout=3)
            last_observation = f"status={r.status_code}, body={r.text[:500]}"

            if r.status_code == 200:
                data = r.json()
                if (
                    data.get("status") == "ready"
                    and data.get("model_version") == EXPECTED_MODEL_VERSION
                ):
                    return data

        except Exception as exc:
            last_observation = repr(exc)

        time.sleep(1)

    raise RuntimeError(
        f"Service did not become ready at {BASE}/ready. "
        f"Expected model_version={EXPECTED_MODEL_VERSION}. "
        f"Last observation: {last_observation}"
    )


def post_prediction(path: str, files: dict) -> dict:
    r = requests.post(f"{BASE}{path}", files=files, timeout=60)
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"{path} returned non-JSON: status={r.status_code}, body={r.text[:500]}")

    if r.status_code >= 400:
        raise RuntimeError(f"{path} failed: status={r.status_code}, body={data}")

    return data


def main() -> None:
    print(f"Using BASE={BASE}")
    print("Checking readiness...")
    ready = wait_ready()
    print(f"Ready: {ready}")

    filename, image_bytes, content_type = make_test_image()
    files = {"file": (filename, image_bytes, content_type)}

    print("Checking sync prediction...")
    sync = post_prediction("/predict-sync", files)
    assert sync["status"] == "completed", sync
    assert sync["model_version"] == EXPECTED_MODEL_VERSION, sync
    assert sync["latency_ms"] < MAX_SYNC_LATENCY_MS, sync

    filename, image_bytes, content_type = make_test_image()
    files = {"file": (filename, image_bytes, content_type)}

    print("Checking async prediction...")
    queued = post_prediction("/predict", files)
    assert queued["status"] == "queued", queued

    result = None
    for _ in range(30):
        r = requests.get(f"{BASE}/predict/{queued['request_id']}", timeout=5)
        result = r.json()
        if result.get("status") == "completed":
            break
        time.sleep(1)

    assert result is not None
    assert result["status"] == "completed", result
    assert result["model_version"] == EXPECTED_MODEL_VERSION, result
    assert result["latency_ms"] < MAX_ASYNC_LATENCY_MS, result

    print("Docker smoke test passed.")


if __name__ == "__main__":
    main()