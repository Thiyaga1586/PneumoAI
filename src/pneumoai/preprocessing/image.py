from io import BytesIO
from PIL import Image
import numpy as np


def read_image_bytes(raw_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(raw_bytes)).convert("L")
    image = image.resize((224, 224))
    arr = np.array(image, dtype="float32") / 255.0
    arr = arr.reshape(1, 1, 224, 224)
    return arr