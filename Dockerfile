# syntax=docker/dockerfile:1.7

FROM python:3.11.14-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV MODEL_DEVICE=cpu
ENV PIP_DEFAULT_TIMEOUT=120

COPY requirements-serving.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements-serving.txt && \
    python -m pip uninstall -y setuptools wheel || true

COPY src src
COPY scripts scripts
COPY models models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "pneumoai.serving.api.app:app", "--host", "0.0.0.0", "--port", "8000"]