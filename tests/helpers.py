from io import BytesIO

from PIL import Image


def make_test_image_bytes() -> bytes:
    image = Image.new("L", (224, 224), color=128)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()