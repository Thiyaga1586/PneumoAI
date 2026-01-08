import io
from fastapi import HTTPException
from PIL import Image, ImageFile, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False


def read_and_validate_image(upload_file) -> Image.Image:
    ct = (upload_file.content_type or "").lower()
    if ct not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail={"error": "unsupported_media_type", "allowed": sorted(ALLOWED_CONTENT_TYPES), "got": ct},
        )

    data = upload_file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) == 0:
        raise HTTPException(status_code=400, detail={"error": "empty_file"})
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail={"error": "file_too_large", "max_bytes": MAX_UPLOAD_BYTES})

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail={"error": "image_too_large_pixels", "max_pixels": MAX_IMAGE_PIXELS})
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail={"error": "invalid_image"})
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "corrupted_image"})

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    return img
