install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest -q tests/unit

run:
	python -m uvicorn pneumoai.serving.api.app:app --host 127.0.0.1 --port 8001 --reload --reload-dir src

docker-build:
	docker build -t pneumoai:latest .

docker-run:
	docker compose up --build

metrics:
	curl http://127.0.0.1:8001/metrics

health:
	curl http://127.0.0.1:8001/ready