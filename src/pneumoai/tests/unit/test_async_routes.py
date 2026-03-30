from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from pneumoai.serving.api.app import app

client = TestClient(app)


def make_test_image_bytes() -> bytes:
    image = Image.new("L", (224, 224), color=128)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_predict_async_returns_queued():
    image_bytes = make_test_image_bytes()

    response = client.post(
        "/predict",
        files={"file": ("test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert "request_id" in body


def test_predict_status_returns_saved_result():
    image_bytes = make_test_image_bytes()

    create_response = client.post(
        "/predict",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    request_id = create_response.json()["request_id"]

    status_response = client.get(f"/predict/{request_id}")
    assert status_response.status_code == 200

    body = status_response.json()
    assert body["request_id"] == request_id
    assert body["status"] == "queued"