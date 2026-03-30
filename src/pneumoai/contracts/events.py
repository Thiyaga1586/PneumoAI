from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class PredictionTask(BaseModel):
    request_id: str
    created_at: datetime
    image_uri: str
    true_label: Optional[str] = None
    requested_model_version: Optional[str] = None


class PredictionStatus(BaseModel):
    request_id: str
    status: str
    image_uri: Optional[str] = None
    submitted_at: Optional[str] = None
    result: Optional[dict] = None