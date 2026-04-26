import numpy as np
from PIL import Image

from pneumoai.serving.triton.client import TritonInferenceClient


def make_test_image_file(tmp_path):
    image = Image.new("L", (224, 224), color=128)
    file_path = tmp_path / "test.png"
    image.save(file_path, format="PNG")
    return str(file_path)


class FakeResponse:
    def as_numpy(self, name):
        assert name == "logits"
        return np.array([[0.75]], dtype=np.float32)


class FakeInferInput:
    def __init__(self, name, shape, datatype):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.data = None

    def set_data_from_numpy(self, array):
        self.data = array


class FakeRequestedOutput:
    def __init__(self, name):
        self.name = name


class FakeGrpcClient:
    def __init__(self, url):
        self.url = url

    def infer(self, model_name, model_version, inputs, outputs):
        assert model_name == "pneumonia_classifier"
        assert isinstance(inputs, list)
        assert isinstance(outputs, list)
        return FakeResponse()


def test_triton_client_predict_returns_probability(tmp_path, monkeypatch):
    image_path = make_test_image_file(tmp_path)

    from pneumoai.serving.triton import client as triton_client_module

    class FakeModule:
        InferenceServerClient = FakeGrpcClient

    monkeypatch.setattr(
        triton_client_module,
        "grpcclient",
        FakeModule,
    )
    monkeypatch.setattr(
        triton_client_module.grpcclient,
        "InferInput",
        FakeInferInput,
    )
    monkeypatch.setattr(
        triton_client_module.grpcclient,
        "InferRequestedOutput",
        FakeRequestedOutput,
    )

    client = TritonInferenceClient("localhost:8001")
    result = client.predict(image_path)

    assert result is not None
    assert "probability" in result
    assert isinstance(result["probability"], float)
    assert "raw_logits" in result