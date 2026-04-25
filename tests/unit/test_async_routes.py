from unittest.mock import patch

from fastapi.testclient import TestClient

from pneumoai.serving.api.app import app
from tests.helpers import make_test_image_bytes

client = TestClient(app)


def test_predict_async_returns_queued():
    image_bytes = make_test_image_bytes()

    with patch(
        "pneumoai.serving.api.routes_async.enqueue_prediction_job",
        return_value={"request_id": "req-123", "status": "queued"},
    ):
        response = client.post(
            "/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "request_id": "req-123",
        "status": "queued",
    }


def test_predict_status_returns_saved_result():
    expected = {
        "request_id": "req-123",
        "status": "completed",
        "model_version": "v2",
        "prediction": "PNEUMONIA",
        "probability": 0.98,
        "threshold": 0.45,
        "latency_ms": 51.2,
        "backend": "local",
        "true_label": None,
        "created_at": "2026-04-24T00:00:00Z",
    }

    with patch(
        "pneumoai.serving.api.routes_async.get_job_status",
        return_value=expected,
    ):
        response = client.get("/predict/req-123")

    assert response.status_code == 200
    assert response.json() == expected