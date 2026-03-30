from pydantic import BaseModel
from typing import Optional


class PredictResponse(BaseModel):
    request_id: str
    status: str
    model_version: str
    prediction: str
    probability: float
    threshold: float
    latency_ms: float
    true_label: Optional[str] = None