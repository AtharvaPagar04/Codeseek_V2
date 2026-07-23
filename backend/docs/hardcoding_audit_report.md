# CodeSeek V2 - Hardcoding & Configuration Audit Report

## 1. Executive Summary

This document provides a comprehensive technical audit of all hardcoded parameters, magic numbers, default values, fallback strings, static routing rules, prompt templates, and security constants across the entire **CodeSeek V2** repository (`backend`, `frontend`, `scripts`, `deploy`, `docker`).

---

## 2. Inventory of Hardcoded Items by Category

---

### Category A: Database, Storage & File System Paths

| File Location | Line Numbers | Variable / Parameter Name | Hardcoded Value | Context & Purpose | Recommended Action / Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `backend/retrieval/db.py` | Line 24 | `_SQLITE_PATH_DEFAULT` | `Path("data") / "codeseek.db"` | Fallback SQLite database path when environment variables are unset. | Good fallback; ensure parent `data/` directory permissions in containers. |
| `backend/rag_ingestion/config.py` | Line 45 | `INGESTION_STATE_FILENAME` | `".rag_ingestion_state.json"` | Metadata file stored in repo roots for incremental skip tracking. | Standard hidden file name. Consider configurable option for read-only mounts. |
| `backend/rag_ingestion/config.py` | Line 56 | `TEMP_CLONE_DIR` | `"/tmp/rag_ingestion"` | Directory for cloning Git repositories during ingestion. | Move to dynamic system temp dir or env var to prevent `/tmp` permission conflicts. |
| `backend/retrieval/support/qdrant_config.py` | Line 21-22 | `host`, `port` | `"localhost"`, `6333` | Default Qdrant vector database connection settings. | Standard fallback; overridden cleanly via `QDRANT_HOST` / `QDRANT_PORT`. |
| `backend/rag_ingestion/config.py` | Line 38 | `COLLECTION_NAME` | `"repository_chunks"` | Default Qdrant collection name fallback. | Fine as global default; dynamic per-repo collections override this at runtime. |
| `backend/retrieval/config.py` | Line 251-253 | `ANSWER_TRACE_OUTPUT_PATH` | `"evals/reports/answer_traces.jsonl"` | File path for logging evaluation answer traces. | Enforce directory existence check before writing during automated eval runs. |

---

### Category B: LLM Models, Provider Endpoints & Embedding Specifications

| File Location | Line Numbers | Variable / Parameter Name | Hardcoded Value | Context & Purpose | Recommended Action / Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `backend/rag_ingestion/config.py` | Lines 47–48 | `EMBEDDING_MODEL`, `EMBEDDING_DIM` | `"BAAI/bge-small-en-v1.5"`, `384` | Default text embedding model name and vector dimension size for ingestion. | Changing model without updating dimension causes Qdrant vector mismatch. |
| `backend/retrieval/config.py` | Lines 58–59 | `EMBEDDING_MODEL`, `EMBEDDING_DIM` | `"BAAI/bge-small-en-v1.5"`, `384` | Default retrieval embedding model name and vector dimension size. | Must remain synchronized with `rag_ingestion/config.py`. |
| `backend/rag_ingestion/config.py` | Lines 105–106 | `CODESEEK_DESCRIPTION_MODEL`, `CODESEEK_LABEL_MODEL` | `"qwen2.5-coder:3b"` | Default Ollama models for chunk description and labeling enrichment. | Recommend configuring smaller default for lower VRAM environments. |
| `backend/retrieval/config.py` | Lines 228–229 | `LOCAL_LLM_PRIMARY_MODEL`, `LOCAL_LLM_COMPLEX_MODEL` | `"qwen2.5-coder:3b-8k"`, `"qwen-coder-7b-8192"` | Default primary and complex Ollama model tags for retrieval query routing. | Ensure instructions explain Ollama model pull commands (`ollama pull ...`). |
| `backend/retrieval/config.py` | Line 216 | `GROQ_MODEL` | `"llama-3.3-70b-versatile"` | Default LLM model for Groq provider execution. | Keep updated as Groq deprecates older model aliases. |
| `backend/retrieval/generation/llm.py` | Lines 71–75 | `OPENAI_MODEL`, `OPENROUTER_MODEL`, `GEMINI_MODEL`, `AICREDITS_MODEL` | `"gpt-4o-mini"`, `"openai/gpt-4o-mini"`, `"gemini-1.5-flash"`, `"gpt-5.4-mini"` | Default cloud provider LLM model IDs. | Configurable via env variables (`RETRIEVAL_OPENAI_MODEL`, etc.). |
| `backend/retrieval/generation/llm.py` | Line 75 | `AICREDITS_BASE_URL` | `"https://api.aicredits.in/v1"` | Default API base URL for AICredits provider wrapper. | Abstract into environment variable override `AICREDITS_BASE_URL`. |
| `backend/retrieval/config.py` | Line 226 | `LOCAL_LLM_BASE_URL` | `"http://localhost:11434/v1"` | Default base URL for local OpenAI-compatible Ollama service. | Standard default for local deployment. |

---

### Category C: Search, Retrieval & Ingestion Hyperparameters

| File Location | Line Numbers | Variable / Parameter Name | Hardcoded Value | Context & Purpose | Recommended Action / Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `backend/retrieval/config.py` | Lines 62–66 | `TOP_K_DENSE`, `TOP_K_LEXICAL`, `TOP_K_AFTER_MERGE` | `15`, `15`, `10` | Number of dense, BM25, and merged candidates retrieved per query. | Tuned for precision/recall tradeoff. Exposed via env variables. |
| `backend/retrieval/config.py` | Line 65 | `MAX_CONTEXT_TOKENS` | `7000` | Global maximum token ceiling for assembled context sent to LLM. | Prevents LLM context overflow window errors. |
| `backend/retrieval/config.py` | Lines 104–105 | `DISPLAY_SOURCES_CAP`, `REASONING_SOURCES_CAP` | `6`, `12` | Caps for UI cited display cards vs backend reasoning sources. | Enforces strict distinction between UI card sources and reasoning context. |
| `backend/retrieval/config.py` | Lines 112–126 | `INTENT_CONTEXT_BUDGETS` | `OVERVIEW: 5200`, `SYMBOL: 2800`, `TRACE: 6500`, `LOW_CONTEXT: 1800`, etc. | Per-intent context token allocation budgets. | Highly optimized for specific query intent types to balance cost/latency. |
| `backend/retrieval/config.py` | Lines 132–151 | `HISTORY_INJECT_THRESHOLD`, `PREVIOUS_CANDIDATE_*` | `0.65` (inject threshold), `0.55` (min score), `0.20` (max ratio), `0.85` (penalty) | Follow-up conversation history candidate injection controls and penalties. | Prevents conversation history from over-contaminating current turn retrieval. |
| `backend/rag_ingestion/config.py` | Lines 50–54 | `MAX_CHUNK_TOKENS`, `SLIDING_WINDOW_SIZE`, `SLIDING_OVERLAP` | `2048` tokens, `100` lines, `20` lines overlap | Code chunking window limits and overlap sizes. | Critical for AST and naive chunking balance. |
| `backend/rag_ingestion/config.py` | Lines 125–126 | `CODESEEK_DESCRIPTION_COOLDOWN_*` | `200` chunks, `60` seconds | GPU/Ollama cooldown timer to prevent thermal throttling or memory exhaustion. | Excellent safety guard for low-resource environments. |

---

### Category D: Security, Auth & Crypto Defaults

| File Location | Line Numbers | Variable / Parameter Name | Hardcoded Value | Context & Purpose | Recommended Action / Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `backend/retrieval/stores/crypto_store.py` | Lines 11–13 | `VERSION`, `NONCE_SIZE`, `TAG_SIZE` | `b"v1"`, `16` bytes, `32` bytes | Authenticated secret encryption payload specification (version flag, nonce size, HMAC tag size). | Strict security standard; ensures backwards compatibility. |
| `backend/retrieval/stores/provider_store.py` | Line 11 | `SUPPORTED_PROVIDER_TYPES` | `{"groq", "openai", "openrouter", "gemini", "aicredits", "local"}` | Set of allowed LLM provider types for user credentials. | When adding a new LLM provider, this set must be updated. |
| `backend/retrieval/support/submission_crypto.py` | Line 66 | Private Key Fallback | `rsa.generate_private_key(public_exponent=65537, key_size=2048)` | In-memory ephemeral RSA key pair generation when disk keys are missing. | Ephemeral keys restart on server reboot. For multi-replica cluster, mandate persistent RSA key env vars. |
| `backend/retrieval/api_service.py` | Lines 139–140 | `GITHUB_OAUTH_TOKEN_URL`, `GITHUB_API_USER_URL` | `"https://github.com/login/oauth/access_token"`, `"https://api.github.com/user"` | GitHub OAuth token exchange and user API endpoints. | Standard GitHub API URLs; could support Enterprise GitHub via env vars. |
| `backend/retrieval/api_service.py` | Line 102 | `AUTH_SESSION_COOKIE` | `"codeseek_session"` | HTTP-only cookie name for authentication session tokens. | Configurable via `CODESEEK_AUTH_SESSION_COOKIE`. |

---

### Category E: Networking, Ports & CORS Configurations

| File Location | Line Numbers | Variable / Parameter Name | Hardcoded Value | Context & Purpose | Recommended Action / Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `backend/retrieval/api_service.py` | Line 92 | `RATE_LIMIT_PER_MINUTE` | `60` | Default rate limit ceiling per user/IP per minute. | Configurable via `CODESEEK_RATE_LIMIT_PER_MINUTE`. |
| `backend/retrieval/api_service.py` | Lines 101, 225–235 | `DEFAULT_CORS_ORIGINS` | `"http://localhost:5173,http://127.0.0.1:5173"` | Permitted CORS origins for local web client interaction. | In local tenant mode, automatically permits local Vite ports (5173, 5174). |
| `frontend/src/utils/api.js` | Line 3 | `API_BASE` | `'http://127.0.0.1:8000'` | Default backend API URL for frontend HTTP requests. | Overridden by `VITE_API_BASE_URL` in environment or Vite build. |
| `frontend/vite.config.js` | Line 7 | `port` | `5173` | Local Vite development server port. | Standard Vite port. |
| `docker-compose.yml` | Lines 54, 63, 100 | Port Mappings | `5432` (postgres), `6333` (qdrant), `${BACKEND_PORT}` (backend), `${CADDY_PORT}` (caddy) | Container service internal and exposed ports. | Parameterized via environment variables where appropriate. |
| `deploy/Caddyfile` | Line 22 | Reverse Proxy Target | `frontend:80` | Internal Caddy reverse proxy mapping for single-page application static hosting. | Matches internal Docker nginx port. |

---

### Category F: Deterministic Logic, Heuristic Routes & Template Prompts

| File Location | Line Numbers | Variable / Parameter Name | Hardcoded Value | Context & Purpose | Recommended Action / Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `backend/retrieval/generation/llm.py` | Lines 42–69 | `SYSTEM_PROMPT` | Senior software engineer persona prompt & grounding rules. | Primary system prompt dictating voice, facts grounding, no fake files, no internal leaks. | Centralized and well-structured prompt definition. |
| `backend/retrieval/generation/code_answers.py` | Lines 121–362 | `FLOW_EVIDENCE_MODEL` | Multi-role flow descriptions for `orchestration`, `retrieval_pipeline`, `auth_session`, `indexing_session`, etc. | Structural models for generating deterministic system flow explanations. | Excellent deterministic fallback for architecture questions. |
| `backend/retrieval/search/searcher.py` | Lines 67–276 | `CODE_REQUEST_TOPIC_ROUTES` | Exact symbol and file path target lists for `auth`, `safe_eval_runner`, `qdrant_upsert`, `evaluation_report_api`, `retrieval_internals`. | Heuristic topic routing to ensure 100% precision on common code requests. | Keep maintained as file structures or symbol names evolve. |
| `backend/retrieval/generation/code_answers.py` | Lines 27–29 | `MAX_FULL_SNIPPET_LINES`, `HEAD_SNIPPET_LINES`, `TAIL_SNIPPET_LINES` | `120`, `80`, `30` | Line truncation thresholds for long code snippets in deterministic answers. | Preserves snippet readability and prevents context blowing. |

---

## 3. High-Risk Findings & Architectural Recommendations

### 1. In-Memory RSA Submission Key Fallback
* **Risk:** In `backend/retrieval/support/submission_crypto.py` (line 66), missing `CODESEEK_SUBMISSION_PRIVATE_KEY_PEM` triggers an in-memory 2048-bit RSA key pair generation.
* **Impact:** In multi-worker backend deployments (e.g. Uvicorn with multiple processes or load-balanced Docker containers), different processes will hold different RSA key pairs. A client requesting a key from Process A will fail decryption on Process B.
* **Recommendation:** Enforce private key persistence to a shared volume or mandate generating a static RSA key file during deployment.

### 2. Temp Directory Permissions for Ingestion
* **Risk:** In `backend/rag_ingestion/config.py` (line 56), `TEMP_CLONE_DIR` defaults to `/tmp/rag_ingestion`.
* **Impact:** Multiple users running under different Unix UIDs on a shared host will encounter permission errors if `/tmp/rag_ingestion` is owned by one UID.
* **Recommendation:** Use `tempfile.gettempdir()` appended with the current UID or process session ID, e.g., `/tmp/rag_ingestion_<uid>`.

### 3. Hardcoded Qdrant Model Dimension Coupling
* **Risk:** `EMBEDDING_DIM = 384` is hardcoded across both ingestion (`rag_ingestion/config.py:48`) and retrieval (`retrieval/config.py:59`).
* **Impact:** Changing the embedding model (e.g., to OpenAI `text-embedding-3-small` with 1536 dims or BGE-large with 1024 dims) requires updating dimensions in multiple files or via environment variables.
* **Recommendation:** Ensure all model/dimension changes are centrally validated in `EmbeddingProvider` metadata.

---

## 4. Conclusion & Maintenance Checklist

All hardcoded values within CodeSeek V2 are categorized cleanly into:
1. **Configurable Runtime Defaults** (overridable via `.env`)
2. **Deterministic Domain Rules** (topic routes, flow templates, system prompts)
3. **Internal Pipeline Constants** (chunk window sizes, line truncation caps, crypto specs)

The repository demonstrates strong discipline by providing environment variable overrides for almost all configuration items. Following the recommendations above will further solidify multi-worker clustering and deployment isolation.
