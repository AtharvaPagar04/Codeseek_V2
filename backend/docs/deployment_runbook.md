# Codeseek Deployment Runbook

This runbook covers the minimum production shape for a personal or small-group Codeseek deployment.

Deployment assets added in-repo:

- root compose stack: [docker-compose.deploy.yml](/home/arch/DEV/CodeSeek/docker-compose.deploy.yml)
- deploy env template: [deploy/.env.example](/home/arch/DEV/CodeSeek/deploy/.env.example)
- TLS reverse proxy config: [deploy/Caddyfile](/home/arch/DEV/CodeSeek/deploy/Caddyfile)
- frontend production image: [frontend/Dockerfile](/home/arch/DEV/CodeSeek/frontend/Dockerfile)
- frontend static server config: [frontend/nginx.conf](/home/arch/DEV/CodeSeek/frontend/nginx.conf)
- deployment smoke test: [scripts/smoke_test_deployment.sh](/home/arch/DEV/CodeSeek/scripts/smoke_test_deployment.sh)

## 1. Environment

Copy [.env.example](/home/arch/DEV/CodeSeek/backend/.env.example) to `.env` and set:

| Variable | Description |
|---|---|
| `CODESEEK_API_KEY` | Random bearer token required by all `/api/v1/*` calls |
| `CODESEEK_APP_ENCRYPTION_KEY` | 32+ char key used to encrypt stored secrets at rest |
| `CODESEEK_REQUIRE_EXPLICIT_APP_ENCRYPTION_KEY` | Set `1` in production; prevents fallback to API key |
| `CODESEEK_DB_BACKEND` | Set `postgres` in production |
| `CODESEEK_DATABASE_URL` | Postgres DSN, e.g. `postgresql://user:pass@host:5432/codeseek` |
| `CODESEEK_AUTH_SESSION_SECURE_COOKIE` | Set `1` in production (HTTPS only) |
| `CODESEEK_ENFORCE_HTTPS` | Set `1` in production; rejects plain-HTTP requests |
| `CODESEEK_ALLOW_PLAINTEXT_SECRET_SUBMISSION` | Set `0` in production; forces RSA-encrypted API key submission |
| `CODESEEK_CORS_ORIGINS` | Comma-separated list of allowed frontend origins |
| `CODESEEK_FRONTEND_URL` | Frontend origin used by the OAuth popup redirect |
| `GITHUB_CLIENT_ID` | GitHub OAuth App client ID (if OAuth login is enabled) |
| `GITHUB_CLIENT_SECRET` | GitHub OAuth App client secret |
| `GITHUB_REDIRECT_URI` | Backend callback URL: `https://api.your-domain.com/auth/github/callback` |
| `CODESEEK_REPO_WORKSPACE` | Writable directory for cloned repository checkouts |
| `CODESEEK_TENANT_ID` | Logical tenant label used to namespace sessions (default: `local`) |

> **Important**: `GITHUB_REDIRECT_URI` must point to the **backend** callback URL, not the frontend.
> Update the "Authorization callback URL" in your GitHub OAuth App settings to match exactly.

If TLS is terminated by a reverse proxy, forward `X-Forwarded-Proto: https` to the backend.

For the full deploy stack, copy [deploy/.env.example](/home/arch/DEV/CodeSeek/deploy/.env.example) to `deploy/.env` and fill in:

- frontend domain
- backend API domain
- ACME email
- backend secrets
- Postgres password
- frontend build-time `VITE_API_BASE_URL`
- frontend build-time `VITE_API_KEY`

## 2. GitHub Auth Mode

Codeseek supports two GitHub authentication modes.  Choose one (or enable both):

### GitHub OAuth (recommended for multi-user deployments)

Users log in via GitHub OAuth. A popup window opens the GitHub authorization page, and the backend
handles the callback, issues a session cookie, and stores an encrypted GitHub access token.

**Required env vars**: `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, `GITHUB_REDIRECT_URI`,
`CODESEEK_FRONTEND_URL`.

**GitHub OAuth App settings**:
- Authorization callback URL: `https://api.your-domain.com/auth/github/callback`
- This URL must match `GITHUB_REDIRECT_URI` exactly.

### Personal Access Token (PAT) connect

Users paste a GitHub PAT (`ghp_...`) in the frontend settings panel. The token is submitted
encrypted via RSA-OAEP and stored server-side. No OAuth App is required.

**Required env vars**: none beyond the base set.

## 3. Start Order

1. Start Postgres and wait for health.
2. Start Qdrant and wait for health.
3. Start `codeseek-api`.
4. Start the frontend behind HTTPS.

Using Docker Compose:

```bash
docker compose -f docker-compose.deploy.yml --env-file deploy/.env up -d --build
```

Verify:

```bash
docker compose -f docker-compose.deploy.yml --env-file deploy/.env ps
curl https://api.your-domain.com/api/v1/health -H "Authorization: Bearer <CODESEEK_API_KEY>"
```

Expected result: backend status is `ok` or `degraded`, never connection-refused.

## 4. HTTPS / Reverse Proxy

Codeseek should not be exposed directly on plain HTTP in deployment.

- Terminate TLS at Nginx, Caddy, Traefik, or a cloud load balancer.
- Proxy `/api/`, `/auth/`, `/health`, and `/metrics` to `codeseek-api:8000`.
- Keep the backend on a private network.
- Set secure cookies and keep `CODESEEK_ENFORCE_HTTPS=1`.

The included [deploy/Caddyfile](/home/arch/DEV/CodeSeek/deploy/Caddyfile) assumes:

- frontend on `https://$CODESEEK_FRONTEND_DOMAIN`
- backend on `https://$CODESEEK_API_DOMAIN`

and forwards `X-Forwarded-Proto: https` automatically.

## 5. Restart Flow

Safe restart order:

1. Restart frontend or reverse proxy.
2. Restart `codeseek-api`.
3. Restart Postgres or Qdrant only when required.

Commands:

```bash
docker compose -f docker-compose.deploy.yml --env-file deploy/.env restart codeseek-api
docker compose -f docker-compose.deploy.yml --env-file deploy/.env logs --tail=200 codeseek-api
```

After restart, verify:

- `GET /api/v1/health` responds
- existing GitHub auth session still works
- expired GitHub sessions show a reconnect notice in the header
- provider credentials still list correctly
- existing repo sessions and chat threads still load

## 6. Rollback Flow

If a new backend image introduces a regression:

1. **Identify** the last known-good image tag or git commit.
2. **Stop** the current container:
   ```bash
   docker compose -f docker-compose.deploy.yml --env-file deploy/.env stop codeseek-api
   ```
3. **Re-tag or rebuild** the previous image:
   ```bash
   git checkout <previous-commit>
   docker compose -f docker-compose.deploy.yml --env-file deploy/.env build codeseek-api frontend
   ```
4. **Restart**:
   ```bash
   docker compose -f docker-compose.deploy.yml --env-file deploy/.env up -d codeseek-api frontend caddy
   ```
5. **Verify** health and smoke-check the UI.

> Database schema changes: Codeseek uses `CREATE TABLE IF NOT EXISTS` and `ALTER TABLE ... ADD COLUMN`
> guards in `init_db()`. There is no downward migration support. If a rollback requires removing
> a column, do so manually via `psql` before restarting the old image.

## 7. Backup Flow

### Postgres

Create a logical backup:

```bash
docker compose exec postgres pg_dump -U codeseek -d codeseek > codeseek-postgres.sql
```

Restore:

```bash
cat codeseek-postgres.sql | docker compose exec -T postgres psql -U codeseek -d codeseek
```

**Schedule**: Run `pg_dump` daily via cron or your cloud provider's managed backup.  
**Retention**: Keep at least 7 daily backups and 4 weekly backups.

### Qdrant

Use the bundled scripts:

```bash
# Backup a specific collection
./.venv/bin/python scripts/qdrant_snapshot_backup.py \
  --collection repository_chunks \
  --out-dir /backups/qdrant

# Restore from a snapshot file
./.venv/bin/python scripts/qdrant_snapshot_restore.py \
  --collection repository_chunks \
  --snapshot-file /backups/qdrant/repository_chunks__<snapshot-name>
```

Schedule recurring Qdrant backups with retention:

```bash
# Keep last 14 snapshots, delete snapshots older than 30 days
./.venv/bin/python scripts/qdrant_snapshot_schedule.py \
  --collections repository_chunks \
  --out-dir /backups/qdrant \
  --keep-last 14 \
  --max-age-days 30
```

> Add this command to a daily cron job on the host running Qdrant.

## 8. Cleanup Jobs

Two cleanup scripts prevent unbounded disk and database growth.  Run them periodically (e.g., weekly via cron).

### Stale repo workspaces

Removes workspace directories that no longer correspond to any active session:

```bash
PYTHONPATH=/path/to/backend \
CODESEEK_DB_BACKEND=postgres \
CODESEEK_DATABASE_URL=postgresql://codeseek:codeseek@localhost:5432/codeseek \
CODESEEK_APP_ENCRYPTION_KEY=<key> \
./.venv/bin/python scripts/cleanup_stale_workspaces.py \
  --max-age-days 30 \
  [--dry-run]
```

### Expired auth sessions

Removes expired `auth_sessions` rows from the database:

```bash
PYTHONPATH=/path/to/backend \
CODESEEK_DB_BACKEND=postgres \
CODESEEK_DATABASE_URL=postgresql://codeseek:codeseek@localhost:5432/codeseek \
CODESEEK_APP_ENCRYPTION_KEY=<key> \
./.venv/bin/python scripts/cleanup_expired_auth_sessions.py \
  [--dry-run]
```

Always do a `--dry-run` first to confirm what will be affected.

## 9. Monitoring

Minimum checks:

- backend: `GET /api/v1/health`
- backend metrics: `GET /api/v1/metrics`
- Postgres: `pg_isready`
- Qdrant: `GET /healthz`

Alert on:

- repeated `401` or auth-expired complaints
- repeated `429` provider failures (check `codeseek_retrieval_errors_total{error_type="http_exception"}`)
- sessions stuck in `indexing` for > 30 minutes
- sessions moving to `failed`
- Qdrant or Postgres healthcheck failures
- backend startup failure (container exits at launch)

## 10. Known Failure Modes and Operator Responses

| Symptom | Likely Cause | Response |
|---|---|---|
| Backend returns `500` on startup / health `degraded` | Qdrant not reachable | Check Qdrant container logs; verify `QDRANT_HOST` and `QDRANT_PORT` |
| Session stuck in `indexing` | Git clone timeout or network issue | Check backend logs; use the frontend "Retry indexing" action or `POST /api/v1/sessions/{id}/retry` |
| Session moves to `failed` | Qdrant full, clone auth error, ingestion OOM | Read `session.error` field; fix cause; retry indexing from the UI or API |
| GitHub OAuth popup does nothing / closes immediately | `GITHUB_REDIRECT_URI` mismatch | Verify callback URL in GitHub OAuth App settings matches env var exactly |
| Provider queries fail with 400 "No active provider credential" | User never added an API key, or credential was deleted | Prompt user to add a provider credential in settings |
| Provider queries fail with 400 "Provider API key rejected or lacks permission" | Wrong provider key, revoked model access, or mismatched provider/model pair | Re-save the provider credential and confirm the selected provider/model is valid for that key |
| Provider queries fail with 429 | LLM provider rate limit exceeded | Retry after backoff, or switch to a different saved provider credential |
| Auth cookie not set after OAuth | `CODESEEK_AUTH_SESSION_SECURE_COOKIE=1` but serving over HTTP | Ensure TLS is correctly terminated; check proxy headers |
| Encrypted secret submission fails | Frontend key_id stale after backend restart (ephemeral RSA key) | Refresh the page to fetch a new public key |
| Postgres connection refused | Postgres not started or wrong DSN | Check `CODESEEK_DATABASE_URL`; verify Postgres container health |
| `CODESEEK_APP_ENCRYPTION_KEY must be set` on startup | Missing env var | Set the encryption key in `.env`; never use the default in production |

## 11. Smoke Checklist

Run this after each deployment:

1. Sign in with GitHub.
2. Open the repo picker and confirm repositories load.
2. Add a provider credential.
3. Refresh once and confirm the provider credential still appears in API Config.
4. Create a repo session.
5. Wait for indexing to reach `ready`.
6. Ask a query and confirm sources render.
7. Verify source copy includes file, symbol, and line details.
8. Refresh the browser and confirm chat/session persistence.

For a quick non-UI smoke test, run:

```bash
./scripts/smoke_test_deployment.sh \
  https://your-domain.com \
  https://api.your-domain.com \
  <CODESEEK_API_KEY>
```

## 12. Postgres Readiness Validation

For a focused persistence validation outside the full UI flow, run:

```bash
PYTHONPATH=/home/arch/DEV/CodeSeek/backend \
CODESEEK_DB_BACKEND=postgres \
CODESEEK_DATABASE_URL=postgresql://codeseek:codeseek@localhost:5432/codeseek \
CODESEEK_APP_ENCRYPTION_KEY=replace-with-real-key \
./.venv/bin/python scripts/validate_postgres_readiness.py
```

This verifies:

- schema creation in Postgres
- `users`, provider credential, repo session, thread, message, and memory row creation
- persistence across backend re-init
- no accidental SQLite file usage in Postgres mode
