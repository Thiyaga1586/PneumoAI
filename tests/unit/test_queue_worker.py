import json

from pneumoai.queue.jobs import get_job_status, job_key, update_job_status


class FakeRedis:
    def __init__(self):
        self.store = {}

    def hset(self, key, mapping):
        self.store.setdefault(key, {})
        self.store[key].update(mapping)

    def hgetall(self, key):
        return self.store.get(key, {})


def test_update_job_status_serializes_and_get_job_status_completed(monkeypatch):
    fake = FakeRedis()

    monkeypatch.setattr("pneumoai.queue.jobs.get_redis_client", lambda: fake)

    request_id = "req-123"

    update_job_status(
        request_id,
        status="completed",
        model_version="v2",
        prediction="PNEUMONIA",
        probability=0.98,
        threshold=0.45,
        latency_ms=51.2,
        backend="local",
        true_label=None,
        completed_at="2026-04-24T00:00:00Z",
    )

    result = get_job_status(request_id)

    assert result == {
        "request_id": request_id,
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


def test_get_job_status_returns_none_for_missing_job(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr("pneumoai.queue.jobs.get_redis_client", lambda: fake)

    assert get_job_status("missing") is None