# CodeSeek Free-Tier Deployment Guide

## 1. Target Stack
- Frontend: Vercel
- Backend: Render Web Service
- Metadata DB: Neon Postgres
- Vector DB: Qdrant Cloud
- Embeddings: AICredits / OpenAI-compatible
- LLM: existing configured provider

## 2. Architecture
The CodeSeek web application is deployed via serverless/PaaS platforms to maintain a generous free tier. The React frontend is statically compiled and served via Vercel. It communicates directly with the Python/FastAPI backend hosted on Render, which orchestrates database operations, embeddings, and LLM completions.

## 3. Branch Strategy
- Deploy from `monorepo-restructure` first.
- Merge `monorepo-restructure` into `main` only after live smoke tests pass.

## 4. Provision Neon Postgres
- Create a Neon project.
- Copy the Postgres connection string.
- Set the `CODESEEK_DATABASE_URL` backend environment variable in Render.
- Production starts clean; do not migrate local SQLite data.

## 5. Provision Qdrant Cloud
- Create a Qdrant Cloud cluster.
- Copy the cluster URL and API key.
- Set `QDRANT_URL` and `QDRANT_API_KEY` in Render.
- `QDRANT_URL` takes precedence over `QDRANT_HOST`/`QDRANT_PORT`.

## 6. Configure Render Backend
- Root directory: `backend`
- Environment: Docker
- Health check path: `/api/v1/health`
- Instance type: Free initially
- Docker CMD uses `${PORT:-8000}` automatically provided by Render

## 7. Configure Vercel Frontend
- Root directory: `frontend`
- Framework: Vite
- Build command: `npm run build`
- Output directory: `dist`
- Environment variable: `VITE_API_BASE_URL=https://<your-render-service>.onrender.com`

## 8. Environment Variables
See [deployment_env_checklist.md](deployment_env_checklist.md) for the exhaustive list of required keys.

## 9. First Deploy Order
1. Deploy Neon and Qdrant.
2. Deploy Render Web Service (ensure all env variables are set).
3. Wait for the Render Web Service to go live (check `/api/v1/health`).
4. Set `VITE_API_BASE_URL` in Vercel.
5. Deploy Vercel Frontend.
6. Verify CORS policies apply by hitting Render APIs via Vercel frontend.

## 10. Live Smoke Test Checklist
- Can the frontend load properly?
- Can a GitHub repo be successfully ingested?
- Are backend logs clear of connection errors to Neon and Qdrant?
- Can chat queries successfully hit the embedding provider, perform retrieval against Qdrant, and respond via the LLM?

## 11. Troubleshooting
- **CORS Errors**: Check if `CORS_ALLOWED_ORIGINS` in Render accurately matches your Vercel URL.
- **Port Binding**: Ensure Render injects `$PORT` and your Docker CMD consumes it.
- **Qdrant Timeout**: Verify the Qdrant Cloud instance is awake.

## 12. Free-Tier Caveats
- Render free tier spins down after inactivity. Initial requests may take 30+ seconds.
- Neon databases sleep after inactivity.
- Vercel functions have strict execution time limits, though CodeSeek's static deployment bypasses most.

## 13. Upgrade Path
For production payloads or faster startup times, consider upgrading the Render Web Service to a paid tier. Moving Qdrant and Postgres closer to the Render region lowers latency significantly.

## 14. Merge-to-main Gate
Do not merge deployment branches to `main` until the complete smoke test checklist is successfully validated on live infrastructure.
