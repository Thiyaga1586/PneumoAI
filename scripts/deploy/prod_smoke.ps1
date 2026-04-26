$ErrorActionPreference = "Stop"

Copy-Item .env.production.example .env.production -Force

docker compose -f docker-compose.prod.yml down --remove-orphans
docker compose -f docker-compose.prod.yml build --no-cache
docker compose -f docker-compose.prod.yml up -d

python .\scripts\smoke\docker_smoke_test.py

docker compose -f docker-compose.prod.yml ps