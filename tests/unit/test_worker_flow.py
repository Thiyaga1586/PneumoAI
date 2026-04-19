from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from pneumoai.serving.api.app import app
from pneumoai.serving.dispatcher.status_store import get_result, clear_results
from pneumoai.serving.dispatcher.task_store import clear_tasks
from pneumoai.serving.dispatcher.worker import process_next_task

client = TestClient(app)


def make_test_image_bytes() -> bytes:
    image = Image.new("L", (224, 224), color=128)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def setup_function():
    clear_results()
    clear_tasks()


def test_worker_processes_queued_task():
    image_bytes = make_test_image_bytes()

    response = client.post(
        "/predict",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200

    request_id = response.json()["request_id"]

    queued = get_result(request_id)
    assert queued is not None
    assert queued["status"] == "queued"

    processed = process_next_task()
    assert processed is not None
    assert processed["request_id"] == request_id
    assert processed["status"] == "completed"

    final_result = get_result(request_id)
    assert final_result is not None
    assert final_result["status"] == "completed"
    assert final_result["prediction"] in {"PNEUMONIA", "NORMAL"}