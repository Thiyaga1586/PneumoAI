FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV MODEL_DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-serving.txt

COPY src src
COPY scripts scripts
COPY models models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "pneumoai.serving.api.app:app", "--host", "0.0.0.0", "--port", "8000"]