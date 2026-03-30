import unittest
import numpy as np
from PIL import Image
import torch

from src.pneumonia_system.model.preprocess import (
    resize_with_padding,
    preprocess_image,
    extract_intensity_histogram,
)

class TestPreprocess(unittest.TestCase):
    def test_resize_with_padding_output_shape(self):
        img = Image.fromarray(np.random.randint(0, 255, (300, 500, 3), dtype=np.uint8), mode="RGB")
        out = resize_with_padding(img, target_size=224)
        self.assertEqual(out.size, (224, 224))
        self.assertEqual(out.mode, "L")  # grayscale

    def test_preprocess_image_tensor_shape_and_hist(self):
        device = torch.device("cpu")
        img = Image.fromarray(np.random.randint(0, 255, (512, 256, 3), dtype=np.uint8), mode="RGB")

        x, gray_np = preprocess_image(img, device=device)

        self.assertEqual(tuple(x.shape), (1, 1, 224, 224))
        self.assertTrue(x.dtype.is_floating_point)

        self.assertEqual(gray_np.shape, (224, 224))
        self.assertEqual(gray_np.dtype, np.uint8)

        hist = extract_intensity_histogram(gray_np, bins=32)
        self.assertEqual(len(hist), 32)
        self.assertTrue(all(h >= 0 for h in hist))

        s = float(np.sum(hist))
        # because density=True, integral is ~1.0 (sum isn't exactly 1.0 but should be close)
        self.assertTrue(0.8 <= s <= 1.2, f"hist sum out of expected range: {s}")

if __name__ == "__main__":
    unittest.main()
