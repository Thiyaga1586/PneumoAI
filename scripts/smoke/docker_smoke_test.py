from __future__ import annotations

import io
import time
from pathlib import Path

import requests
from PIL import Image

BASE = "http://127.0.0.1:8080"


def make_test_image() -> tuple[str, bytes, str]:
    img = Image.new("L", (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "test.png", buf.getvalue(), "image/png"


def wait_ready(timeout_s: int = 60) -> dict:
    deadline = time.time() + timeout_s
    last_error = None

    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE}/ready", timeout=3)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "ready" and data.get("model_version") == "v2":
                    return data
        except Exception as exc:
            last_error = exc

        time.sleep(1)

    raise RuntimeError(f"Service did not become ready. Last error: {last_error}")


def main() -> None:
    print("Checking readiness...")
    ready = wait_ready()
    print(f"Ready: {ready}")

    filename, image_bytes, content_type = make_test_image()
    files = {"file": (filename, image_bytes, content_type)}

    print("Checking sync prediction...")
    sync = requests.post(f"{BASE}/predict-sync", files=files, timeout=30).json()
    assert sync["status"] == "completed", sync
    assert sync["model_version"] == "v2", sync
    assert sync["latency_ms"] < 500, sync

    filename, image_bytes, content_type = make_test_image()
    files = {"file": (filename, image_bytes, content_type)}

    print("Checking async prediction...")
    queued = requests.post(f"{BASE}/predict", files=files, timeout=30).json()
    assert queued["status"] == "queued", queued

    result = None
    for _ in range(30):
        result = requests.get(f"{BASE}/predict/{queued['request_id']}", timeout=5).json()
        if result["status"] == "completed":
            break
        time.sleep(1)

    assert result is not None
    assert result["status"] == "completed", result
    assert result["model_version"] == "v2", result
    assert result["latency_ms"] < 500, result

    print("Docker smoke test passed.")


if __name__ == "__main__":
    main()