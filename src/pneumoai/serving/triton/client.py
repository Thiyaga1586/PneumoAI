from typing import Optional

import numpy as np
try:
    import tritonclient.grpc as grpcclient
    TRITON_AVAILABLE = True
except ImportError:
    grpcclient = None
    TRITON_AVAILABLE = False

from pneumoai.preprocessing.image import read_image_bytes


class TritonInferenceClient:
    def __init__(self, url: str):
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton client is not installed. Install tritonclient if using Triton backend."
            )
        self.client = grpcclient.InferenceServerClient(url=url)

    def predict(self, image_uri: str, requested_version: Optional[str] = None) -> dict:
        with open(image_uri, "rb") as f:
            raw = f.read()

        tensor = read_image_bytes(raw).astype(np.float32)

        infer_input = grpcclient.InferInput("input_image", tensor.shape, "FP32")
        infer_input.set_data_from_numpy(tensor)

        output = grpcclient.InferRequestedOutput("logits")

        response = self.client.infer(
            model_name="pneumonia_classifier",
            model_version=requested_version or "",
            inputs=[infer_input],
            outputs=[output],
        )

        logits = response.as_numpy("logits")
        probability = 1.0 / (1.0 + np.exp(-logits))
        prob_value = float(probability.reshape(-1)[0])

        return {
            "probability": prob_value,
            "raw_logits": logits.tolist(),
        }