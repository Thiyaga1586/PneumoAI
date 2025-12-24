import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def new_request_id() -> str:
    return uuid.uuid4().hex


def get_logger(name: str = "pneumonia_system") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    request_id: Optional[str] = None,
    **fields: Any,
) -> None:
    payload: Dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "request_id": request_id,
        **fields,
    }
    logger.info(json.dumps(payload, default=str))
