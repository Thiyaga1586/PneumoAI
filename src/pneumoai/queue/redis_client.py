from __future__ import annotations

import redis

from pneumoai.common.settings import settings


def get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def job_key(request_id: str) -> str:
    return f"pneumoai:job:{request_id}"