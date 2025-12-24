from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    device: str
    model_version: str


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted class label")
    probability: float = Field(..., ge=0.0, le=1.0, description="Sigmoid probability for Pneumonia")
    latency_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds")
    model_version: str
    request_id: Optional[str] = Field(None, description="Unique request id for tracing")


class DriftResponse(BaseModel):
    before_version: str
    after_version: str
    rolled_back: bool
    window: int
    threshold: float
    psi: float
    served_in_memory: str


class MetricsResponse(BaseModel):
    request_count_in_memory: int
    p95_latency_ms: float
