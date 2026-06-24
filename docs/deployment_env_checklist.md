# Deployment Environment Checklist

This checklist contains all necessary environment variables needed for CodeSeek deployment.

## 1. Render Backend Required Env
| Variable | Description | Example |
|---|---|---|
| `CODESEEK_API_KEY` | Secret key for authenticated API routes | `your-secure-random-key` |
| `CODESEEK_APP_ENCRYPTION_KEY` | AES encryption key for secret storage | `your-32-byte-base64-encoded-key` |
| `CODESEEK_DATABASE_URL` | Neon Postgres Connection String | `postgresql://codeseek:pass@host/db` |
| `QDRANT_URL` | Qdrant Cloud URL | `https://your-cluster.qdrant.tech:6333` |
| `QDRANT_API_KEY` | Qdrant Cloud API Key | `your-qdrant-api-key` |
| `CORS_ALLOWED_ORIGINS` | Permitted frontend origins | `https://your-vercel-app.vercel.app` |
| `CODESEEK_EMBEDDING_PROVIDER` | Must be `openai_compatible` for AICredits | `openai_compatible` |
| `CODESEEK_EMBEDDING_BASE_URL` | Cloud provider embedding endpoint | `https://api.aicredits.in/v1` |
| `CODESEEK_EMBEDDING_API_KEY` | Provider API Key | `your-provider-key` |
| `CODESEEK_EMBEDDING_MODEL` | Explicit model identifier | `text-embedding-3-small` |

## 2. Render Backend Optional Env
| Variable | Description | Default |
|---|---|---|
| `CODESEEK_EMBEDDING_DIMENSIONS` | Explicit vector size; leave blank for Auto | (blank) |
| `CODESEEK_EMBEDDING_TIMEOUT_SECONDS` | Timeout for cloud embedding requests | 60 |
| `CODESEEK_EMBEDDING_BATCH_SIZE` | Chunk ingestion grouping count | 100 |

## 3. Vercel Frontend Env
| Variable | Description | Example |
|---|---|---|
| `VITE_API_BASE_URL` | Points to the live Render backend URL | `https://your-service.onrender.com` |

## 4. Neon Env
- Supply the primary connection string to `CODESEEK_DATABASE_URL` in Render.

## 5. Qdrant Cloud Env
- Provide the cluster URL to `QDRANT_URL`.
- Provide the API Key to `QDRANT_API_KEY`.

## 6. Embedding Provider Env
Recommended (AICredits):
- `CODESEEK_EMBEDDING_PROVIDER=openai_compatible`
- `CODESEEK_EMBEDDING_BASE_URL=https://api.aicredits.in/v1`
- `CODESEEK_EMBEDDING_MODEL=text-embedding-3-small` (Use plain ID without prefix if required)
- Dimension scaling: Auto (leave `CODESEEK_EMBEDDING_DIMENSIONS` blank). CodeSeek will adopt provider-returned dimensions when safe.

## 7. LLM Provider Env
LLM credentials and model configurations are securely configured on a per-user basis directly through the CodeSeek UI. Ensure the deployment allows plaintext or encrypted submission logic based on your `CODESEEK_ALLOW_PLAINTEXT_SECRET_SUBMISSION` choice.

## 8. Local-only Env
Do not set these in production:
- `QDRANT_HOST` / `QDRANT_PORT` (Overrides cloud if set improperly)
- `INGESTION_EMBEDDING_MODEL` / `INGESTION_EMBEDDING_DIM` (For local/offline transformer usage)

## 9. Variables that must never be set in frontend
> **WARNING**: The following must never be injected into Vercel or any client-side build step.
- `CODESEEK_API_KEY`
- `QDRANT_API_KEY`
- `CODESEEK_EMBEDDING_API_KEY`
- Database URLs (like `CODESEEK_DATABASE_URL`)

## 10. Secret safety checklist
- Are Vercel variables strictly limited to `VITE_API_BASE_URL`?
- Are backend secrets configured through Render's secret manager?
- Did you commit any actual secrets to the repository? (Run `git diff` before pushing)
