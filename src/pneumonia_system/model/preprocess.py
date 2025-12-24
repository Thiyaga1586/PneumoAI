import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def resize_with_padding(img: Image.Image, target_size: int = 224) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size

    scale = target_size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)

    img = img.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    canvas.paste(img, ((target_size - nw) // 2, (target_size - nh) // 2))

    return canvas.convert("L")  # grayscale (matches cleaning)

_tensor_norm = transforms.Compose([
    transforms.ToTensor(),               # [0,1]
    transforms.Normalize([0.5], [0.5])   # -> [-1,1] scale
])

# IMPORTANT: must match training distribution (padded grayscale)
class PadTo224Transform:
    def __call__(self, img: Image.Image):
        img = resize_with_padding(img, 224)
        return _tensor_norm(img)

val_transform = transforms.Compose([
    PadTo224Transform(),
])

# 4) Main preprocessing entrypoint (API uses this)
def preprocess_image(pil_img: Image.Image, device: torch.device):
    """
    Full preprocessing pipeline for inference.
    Raw image -> padded grayscale 224x224 -> tensor normalized
    Returns:
        tensor: (1,1,224,224)
        gray_np: uint8 (224,224) for drift histogram
    """
    resized = resize_with_padding(pil_img, 224)
    gray_np = np.array(resized, dtype=np.uint8)

    tensor = _tensor_norm(resized).unsqueeze(0).to(device)
    return tensor, gray_np

# 5) Drift feature extraction
def extract_intensity_histogram(gray_np: np.ndarray, bins: int = 32):
    hist, _ = np.histogram(
        gray_np.flatten(),
        bins=bins,
        range=(0, 255),
        density=True
    )
    return hist.tolist()
