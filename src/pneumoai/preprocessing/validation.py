from pathlib import Path

from fastapi import HTTPException, UploadFile

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
}

ALLOWED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}


async def validate_upload(file: UploadFile) -> None:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPEG image files are allowed.",
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid image content type.",
        )

    if not filename.strip():
        raise HTTPException(
            status_code=400,
            detail="Filename is missing.",
        )