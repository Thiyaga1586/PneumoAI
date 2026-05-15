# PneumoAI Ubuntu VM Deployment

## VM target

Recommended minimum:

- Ubuntu 22.04/24.04
- 2 vCPU
- 4-8 GB RAM
- 30+ GB disk
- Ports open: 22, 80

## Install Docker

```bash
sudo apt update
sudo apt install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
docker --version
docker compose version