from io import BytesIO

from PIL import Image

from pneumoai.serving.dispatcher.inference_service import (
    run_inference,
    run_local_inference,
    run_triton_inference,
)


def make_test_image_file(tmp_path):
    image = Image.new("L", (224, 224), color=128)
    file_path = tmp_path / "test.png"
    image.save(file_path, format="PNG")
    return str(file_path)


def test_run_local_inference_returns_expected_shape(tmp_path):
    image_path = make_test_image_file(tmp_path)

    result = run_local_inference(image_path)

    assert result is not None
    assert result["backend"] == "local"
    assert "model_version" in result
    assert "prediction" in result
    assert result["prediction"] in {"PNEUMONIA", "NORMAL"}
    assert isinstance(result["probability"], float)
    assert isinstance(result["threshold"], float)
    assert isinstance(result["latency_ms"], float)


def test_run_inference_uses_local_backend_when_config_is_local(tmp_path, monkeypatch):
    image_path = make_test_image_file(tmp_path)

    from pneumoai.serving.dispatcher import inference_service

    monkeypatch.setattr(inference_service.settings, "inference_backend", "local")

    result = run_inference(image_path)

    assert result["backend"] == "local"
    assert result["prediction"] in {"PNEUMONIA", "NORMAL"}


def test_run_inference_falls_back_to_local_when_triton_fails(tmp_path, monkeypatch):
    image_path = make_test_image_file(tmp_path)

    from pneumoai.serving.dispatcher import inference_service

    monkeypatch.setattr(inference_service.settings, "inference_backend", "triton")
    monkeypatch.setattr(inference_service.settings, "triton_fallback_to_local", True)

    def fake_triton_inference(image_uri, requested_version=None):
        raise RuntimeError("Triton unavailable")

    monkeypatch.setattr(
        inference_service,
        "run_triton_inference",
        fake_triton_inference,
    )

    result = run_inference(image_path)

    assert result["backend"] == "local"
    assert result["prediction"] in {"PNEUMONIA", "NORMAL"}


def test_run_inference_raises_when_triton_fails_and_fallback_disabled(tmp_path, monkeypatch):
    image_path = make_test_image_file(tmp_path)

    from pneumoai.serving.dispatcher import inference_service

    monkeypatch.setattr(inference_service.settings, "inference_backend", "triton")
    monkeypatch.setattr(inference_service.settings, "triton_fallback_to_local", False)

    def fake_triton_inference(image_uri, requested_version=None):
        raise RuntimeError("Triton unavailable")

    monkeypatch.setattr(
        inference_service,
        "run_triton_inference",
        fake_triton_inference,
    )

    try:
        run_inference(image_path)
        assert False, "Expected RuntimeError when Triton fails and fallback is disabled"
    except RuntimeError as exc:
        assert "Triton unavailable" in str(exc)