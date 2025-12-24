FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "src.pneumonia_system.serving.api:app", "--host", "0.0.0.0", "--port", "7860"]
