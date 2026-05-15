from fastapi.testclient import TestClient

from pneumoai.serving.api.app import app
from pneumoai.serving.api import routes_predict
from tests.helpers import make_test_image_bytes

client = TestClient(app)


def test_predict_sync(monkeypatch):
    audit_calls = []

    def fake_log_prediction(
        request_id: str,
        model_version: str,
        prediction: str,
        probability: float,
        threshold: float,
        latency_ms: float,
        true_label: str | None,
    ) -> None:
        audit_calls.append(
            {
                "request_id": request_id,
                "model_version": model_version,
                "prediction": prediction,
                "probability": probability,
                "threshold": threshold,
                "latency_ms": latency_ms,
                "true_label": true_label,
            }
        )

    monkeypatch.setattr(routes_predict, "log_prediction", fake_log_prediction)

    image_bytes = make_test_image_bytes()
    response = client.post(
        "/predict-sync",
        files={"file": ("test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "completed"
    assert data["model_version"] == "v2"
    assert data["prediction"] in {"NORMAL", "PNEUMONIA"}
    assert 0.0 <= data["probability"] <= 1.0
    assert data["threshold"] == 0.45
    assert data["latency_ms"] >= 0.0
    assert data["request_id"]

    assert len(audit_calls) == 1
    assert audit_calls[0]["request_id"] == data["request_id"]
    assert audit_calls[0]["model_version"] == data["model_version"]