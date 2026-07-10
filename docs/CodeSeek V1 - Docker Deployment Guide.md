# CodeSeek V1 – Docker Deployment Guide

This guide explains how to build, run, manage, and troubleshoot the complete **CodeSeek V1** application using Docker Compose. It covers everything from the initial setup to production deployment and routine maintenance.

## Prerequisites

Before getting started, ensure the following software is installed on your system:

* Docker Engine 24+
* Docker Compose v2
* Git

Verify the installation:

```bash
docker --version
docker compose version
git --version
```

## Clone the Repository

Clone the project and navigate into the repository:

```bash
git clone https://github.com/<your-username>/CodeSeek.git
cd CodeSeek
```

## Configure Environment Variables

Create your environment file:

```bash
cp deploy/.env.example .env
```

Edit `.env` and provide all required values.

Minimum required configuration:

```env
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
GITHUB_REDIRECT_URI=

CODESEEK_API_KEY=
CODESEEK_APP_ENCRYPTION_KEY=

POSTGRES_PASSWORD=

CODESEEK_EMBEDDING_PROVIDER=
CODESEEK_EMBEDDING_MODEL=
CODESEEK_EMBEDDING_API_KEY=
```

## Build the Application

Build all Docker images:

```bash
docker compose build
```

To rebuild everything from scratch:

```bash
docker compose build --no-cache
```

## Start the Application

Launch the entire stack in detached mode:

```bash
docker compose up -d
```

This starts:

* Frontend (React + Nginx)
* Backend (FastAPI)
* PostgreSQL
* Qdrant
* Caddy Reverse Proxy

## Verify Services

Check the status of every container:

```bash
docker compose ps
```

Expected output:

```
backend     healthy
frontend    healthy
postgres    healthy
qdrant      healthy
caddy       running
```

## Viewing Logs

View logs from every service:

```bash
docker compose logs -f
```

Specific services:

Backend

```bash
docker compose logs -f backend
```

Frontend

```bash
docker compose logs -f frontend
```

Caddy

```bash
docker compose logs -f caddy
```

PostgreSQL

```bash
docker compose logs -f postgres
```

Qdrant

```bash
docker compose logs -f qdrant
```

## Restart Services

Restart the complete application:

```bash
docker compose restart
```

Restart only the backend:

```bash
docker compose restart backend
```

Restart only the frontend:

```bash
docker compose restart frontend
```

Restart only Caddy:

```bash
docker compose restart caddy
```

## Stop the Application

Stop containers without removing them:

```bash
docker compose stop
```

Start previously stopped containers:

```bash
docker compose start
```

## Shut Down the Stack

Remove containers while keeping persistent data:

```bash
docker compose down
```

## Remove Everything (Including Data)

To completely reset the application:

```bash
docker compose down -v
```

This removes:

* PostgreSQL database
* Qdrant vector storage
* Repository workspace
* Caddy certificates and configuration

Use this only if you want a completely fresh installation.

## Rebuild After Code Changes

Rebuild images and restart services:

```bash
docker compose up --build -d
```

Force recreation of every container:

```bash
docker compose down
docker compose up --build --force-recreate -d
```

## Execute Commands Inside Containers

Backend shell:

```bash
docker compose exec backend bash
```

Frontend shell:

```bash
docker compose exec frontend sh
```

PostgreSQL shell:

```bash
docker compose exec postgres psql -U codeseek
```

Qdrant shell:

```bash
docker compose exec qdrant sh
```

## Container Information

Show running containers:

```bash
docker compose ps
```

Inspect a specific container:

```bash
docker inspect <container-name>
```

## Health Checks

Frontend:

```bash
curl https://codeseek.atharvapagar.xyz/healthz
```

Backend:

```bash
curl https://codeseek.atharvapagar.xyz/api/health
```

Expected response:

```json
{"status":"ok"}
```

## Docker Images

List all images:

```bash
docker images
```

View CodeSeek image sizes:

```bash
docker images | grep codeseek
```

## Docker Volumes

List all Docker volumes:

```bash
docker volume ls
```

Inspect a volume:

```bash
docker volume inspect codeseek_v2_postgres_data
```

Delete a specific volume:

```bash
docker volume rm codeseek_v2_postgres_data
```

## Docker Networks

List networks:

```bash
docker network ls
```

Inspect the application network:

```bash
docker network inspect codeseek_v2_codeseek_net
```

## Updating the Application

Pull the latest changes:

```bash
git pull
```

Rebuild and restart:

```bash
docker compose up --build -d
```

## Updating Dependencies

If Dockerfiles or dependency files have changed:

```bash
docker compose build --no-cache
docker compose up -d
```

## Troubleshooting

### Check service status

```bash
docker compose ps
```

### View backend logs

```bash
docker compose logs -f backend
```

### Restart backend

```bash
docker compose restart backend
```

### Perform a complete clean rebuild

```bash
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

## Daily Development Workflow

Start the application:

```bash
docker compose up -d
```

Monitor backend logs:

```bash
docker compose logs -f backend
```

After pulling new code:

```bash
git pull
docker compose up --build -d
```

Shutdown:

```bash
docker compose down
```

## First-Time Setup

```bash
git clone <repository>

cd CodeSeek

cp deploy/.env.example .env

# Configure .env

docker compose build

docker compose up -d

docker compose ps
```

## Production Deployment

```bash
git pull

docker compose build

docker compose up -d

docker compose ps

curl https://codeseek.atharvapagar.xyz/healthz

curl https://codeseek.atharvapagar.xyz/api/health
```

## Docker Architecture

```
                    Internet
                        │
                        ▼
                 Caddy Reverse Proxy
                  (HTTPS / Routing)
                  Ports: 80 / 443
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
 Frontend (React/Vite)          Backend (FastAPI)
      Nginx                           Uvicorn
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
                  PostgreSQL                  Qdrant Vector DB
```

This guide serves as the complete operational reference for **CodeSeek V1**, covering installation, container lifecycle management, updates, troubleshooting, and production deployment using Docker Compose.
