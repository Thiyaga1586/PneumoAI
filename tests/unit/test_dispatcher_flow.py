from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from pneumoai.serving.api.app import app
from pneumoai.serving.dispatcher.consumer import LocalPredictionConsumer
from pneumoai.serving.dispatcher.status_store import clear_results
from pneumoai.serving.dispatcher.task_store import clear_tasks

client = TestClient(app)


def make_test_image_bytes() -> bytes:
    image = Image.new("L", (224, 224), color=128)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def setup_function():
    clear_results()
    clear_tasks()


def test_dispatcher_producer_consumer_flow():
    image_bytes = make_test_image_bytes()

    create_response = client.post(
        "/predict",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert create_response.status_code == 200

    request_id = create_response.json()["request_id"]

    queued_response = client.get(f"/predict/{request_id}")
    assert queued_response.status_code == 200
    assert queued_response.json()["status"] == "queued"

    consumer = LocalPredictionConsumer()
    processed = consumer.consume_once()

    assert processed is not None
    assert processed["request_id"] == request_id
    assert processed["status"] == "completed"

    final_response = client.get(f"/predict/{request_id}")
    assert final_response.status_code == 200
    assert final_response.json()["status"] == "completed"