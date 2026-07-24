# Deployment

## Compose Stack

The root `docker-compose.yml` runs PostgreSQL, Qdrant, the backend, the built frontend, and Caddy on one Docker network.

```bash
cp deploy/.env.example .env
docker compose up -d --build
docker compose ps
```

Persistent named volumes hold PostgreSQL data, Qdrant data, repository workspaces, and Caddy state.

## Routing

`deploy/Caddyfile` currently serves `https://localhost:${CADDY_PORT}` with an internal certificate. `/api/*` and `/auth/*` are proxied to the `backend` service; other requests go to the frontend.

The Caddy configuration also adds content-type, frame, referrer, and XSS response headers.

## Required Configuration

Set strong values for:

- `POSTGRES_PASSWORD`
- `CODESEEK_API_KEY`
- `CODESEEK_APP_ENCRYPTION_KEY`
- GitHub OAuth credentials
- LLM and embedding provider credentials
- Frontend URL and CORS origins

Use PostgreSQL, secure cookies, HTTPS enforcement, and explicit encryption-key validation for deployment.

## Deployment Compose Variant

`docker-compose.deploy.yml` reads `deploy/.env` and exposes ports 80 and 443. In the current files its API service is named `codeseek-api`, while `deploy/Caddyfile` proxies to `backend`; those service names must be aligned before using that variant.

## Health Checks

Compose waits for PostgreSQL and Qdrant before starting the backend, then waits for backend and frontend health before starting Caddy.

Run `./scripts/smoke_test_deployment.sh` after startup.
