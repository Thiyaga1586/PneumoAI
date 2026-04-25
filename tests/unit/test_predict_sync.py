from io import BytesIO
from fastapi.testclient import TestClient
from PIL import Image

from pneumoai.serving.api.app import app
from pneumoai.models.registry import get_current_version

client = TestClient(app)


def make_test_image_bytes() -> bytes:
    image = Image.new("L", (224, 224), color=128)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_predict_sync():
    image_bytes = make_test_image_bytes()
    response = client.post(
        "/predict-sync",
        files={"file": ("test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["model_version"] == get_current_version()
    assert body["prediction"] in {"PNEUMONIA", "NORMAL"}