from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from pneumoai.common.ids import generate_request_id
from pneumoai.preprocessing.validation import validate_upload
from pneumoai.serving.dispatcher.producer import LocalPredictionProducer
from pneumoai.serving.dispatcher.status_store import get_result
from pneumoai.storage.request_store import save_request_image

router = APIRouter()
producer = LocalPredictionProducer()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    true_label: Optional[str] = Form(default=None),
):
    await validate_upload(file)

    request_id = generate_request_id()
    raw = await file.read()
    image_uri = save_request_image(
        request_id=request_id,
        raw_bytes=raw,
        filename=file.filename or "upload.png",
    )

    producer.publish(
        request_id=request_id,
        image_uri=image_uri,
        true_label=true_label,
    )

    return {
        "request_id": request_id,
        "status": "queued",
    }


@router.get("/predict/{request_id}")
def get_prediction(request_id: str):
    result = get_result(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result