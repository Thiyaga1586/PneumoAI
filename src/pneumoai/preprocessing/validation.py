from fastapi import UploadFile, HTTPException


ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg"}


async def validate_upload(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPEG images are allowed",
        )