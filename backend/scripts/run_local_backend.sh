#!/usr/bin/env bash
# run_local_backend.sh — launch CodeSeek backend for local development.
#
# By default this script PRESERVES the local SQLite database to keep
# GitHub authentication sessions and repository session states.
#
# To perform a clean start (wipe DB, session data, ingestion caches):
#   ./scripts/run_local_backend.sh --clean
#   OR
#   CODESEEK_CLEAN_START=1 ./scripts/run_local_backend.sh
set -euo pipefail

# Parse command line options
cli_clean=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      cli_clean=1
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --clean    Force a clean start, wiping all local state (DB, session data, ingestion caches)."
      echo "  --help     Show this help message."
      echo ""
      echo "By default, this script preserves the local database to keep authentication sessions and repository session state."
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help to see available options."
      exit 1
      ;;
  esac
done


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${BACKEND_ROOT}/.." && pwd)"
DEFAULT_SQLITE_PATH="${ROOT_DIR}/data/codeseek.db"

cd "$BACKEND_ROOT"

echo "[local-backend] preparing local backend"

# Save existing environment variables to prevent .env from overriding explicit overrides
SAVED_DB_BACKEND="${CODESEEK_DB_BACKEND:-}"
SAVED_SQLITE_PATH="${CODESEEK_SQLITE_PATH:-}"
SAVED_DB_PATH="${CODESEEK_DB_PATH:-}"
SAVED_DATABASE_URL="${CODESEEK_DATABASE_URL:-}"
SAVED_DATABASE_URL_LEGACY="${DATABASE_URL:-}"

# Load .env
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Restore overrides
if [[ -n "$SAVED_DB_BACKEND" ]]; then export CODESEEK_DB_BACKEND="$SAVED_DB_BACKEND"; fi
if [[ -n "$SAVED_SQLITE_PATH" ]]; then export CODESEEK_SQLITE_PATH="$SAVED_SQLITE_PATH"; fi
if [[ -n "$SAVED_DB_PATH" ]]; then export CODESEEK_DB_PATH="$SAVED_DB_PATH"; fi
if [[ -n "$SAVED_DATABASE_URL" ]]; then export CODESEEK_DATABASE_URL="$SAVED_DATABASE_URL"; fi
if [[ -n "$SAVED_DATABASE_URL_LEGACY" ]]; then export DATABASE_URL="$SAVED_DATABASE_URL_LEGACY"; fi

# Determine and normalize backend
DB_BACKEND="${CODESEEK_DB_BACKEND:-sqlite}"
DB_BACKEND="$(echo "$DB_BACKEND" | tr '[:upper:]' '[:lower:]')"

if [[ "$DB_BACKEND" == "sqlite" ]]; then
  export CODESEEK_DB_BACKEND="sqlite"
  export CODESEEK_SQLITE_PATH="${CODESEEK_SQLITE_PATH:-${CODESEEK_DB_PATH:-$DEFAULT_SQLITE_PATH}}"
  # If it is relative (e.g. "../data/codeseek.db"), resolve it relative to BACKEND_ROOT
  if [[ "$CODESEEK_SQLITE_PATH" != /* ]]; then
    CODESEEK_SQLITE_PATH="$(cd "$BACKEND_ROOT" && mkdir -p "$(dirname "$CODESEEK_SQLITE_PATH")" && cd "$(dirname "$CODESEEK_SQLITE_PATH")" && pwd)/$(basename "$CODESEEK_SQLITE_PATH")"
  fi
  export CODESEEK_DB_PATH="$CODESEEK_SQLITE_PATH"
  mkdir -p "$(dirname "$CODESEEK_SQLITE_PATH")"
  unset DATABASE_URL
  unset CODESEEK_DATABASE_URL
elif [[ "$DB_BACKEND" == "postgres" ]]; then
  export CODESEEK_DB_BACKEND="postgres"
  POSTGRES_URL="${CODESEEK_DATABASE_URL:-${DATABASE_URL:-}}"
  if [[ -z "$POSTGRES_URL" ]]; then
    echo "[ERROR] CODESEEK_DB_BACKEND=postgres requires CODESEEK_DATABASE_URL or DATABASE_URL to be set." >&2
    exit 1
  fi
  export CODESEEK_DATABASE_URL="$POSTGRES_URL"
  export DATABASE_URL="$POSTGRES_URL"
  unset CODESEEK_SQLITE_PATH
  unset CODESEEK_DB_PATH
else
  echo "[ERROR] Unsupported CODESEEK_DB_BACKEND='$DB_BACKEND'. Use sqlite or postgres." >&2
  exit 1
fi

export RETRIEVAL_REPO_ROOT="$BACKEND_ROOT"
export CODESEEK_TENANT_ID="${CODESEEK_TENANT_ID:-local}"
export CODESEEK_API_KEY="${CODESEEK_API_KEY:-local-dev-key}"

CODESEEK_REPO_WORKSPACE="${CODESEEK_REPO_WORKSPACE:-/tmp/codeseek_repo_workspace}"
export CODESEEK_REPO_WORKSPACE

INGESTION_TEMP_CLONE_DIR="${INGESTION_TEMP_CLONE_DIR:-/tmp/rag_ingestion}"
export INGESTION_TEMP_CLONE_DIR

# Ingestion / retrieval tuning for local dev.
export QDRANT_RECREATE_COLLECTION="${QDRANT_RECREATE_COLLECTION:-0}"
export INGESTION_ENABLE_INCREMENTAL_FILE_SKIP="${INGESTION_ENABLE_INCREMENTAL_FILE_SKIP:-0}"
export RETRIEVAL_ENABLE_DENSE="${RETRIEVAL_ENABLE_DENSE:-0}"
export RETRIEVAL_ENABLE_LEXICAL="${RETRIEVAL_ENABLE_LEXICAL:-1}"

# LLM chunk description settings.
export CHUNK_DESCRIPTION_MAX_CHUNKS="${CHUNK_DESCRIPTION_MAX_CHUNKS:--1}"
export CHUNK_DESCRIPTION_SLEEP_SECONDS="${CHUNK_DESCRIPTION_SLEEP_SECONDS:-0}"
export CHUNK_DESCRIPTION_RETRY_ON_RATE_LIMIT="${CHUNK_DESCRIPTION_RETRY_ON_RATE_LIMIT:-0}"
export CHUNK_DESCRIPTION_MAX_INPUT_CHARS="${CHUNK_DESCRIPTION_MAX_INPUT_CHARS:-1800}"
export CODESEEK_DESCRIPTION_MAX_TOKENS="${CODESEEK_DESCRIPTION_MAX_TOKENS:-160}"
export RETRIEVAL_LOCAL_LLM_TIMEOUT_SECONDS="${RETRIEVAL_LOCAL_LLM_TIMEOUT_SECONDS:-90}"

# ---------------------------------------------------------------------------
# Clean-start execution:
# Preserves local database by default. If --clean is passed or
# CODESEEK_CLEAN_START=1 is specified in the environment, wips the state.
# ---------------------------------------------------------------------------
CLEAN_START="${CODESEEK_CLEAN_START:-0}"
if [[ "$cli_clean" == "1" ]]; then
  CLEAN_START=1
fi
export CODESEEK_CLEAN_START="$CLEAN_START"

if [[ "$CLEAN_START" == "1" ]]; then
  echo "[local-backend] clean start requested"
  echo "[local-backend] ⚠  If the browser shows 401 after restart, log in again — auth state was reset."

  # --- SQLite database (all persistent session/auth/provider state) ---
  if [[ -f "$CODESEEK_DB_PATH" ]]; then
    echo "[local-backend] removing local db: $CODESEEK_DB_PATH"
    rm -f "$CODESEEK_DB_PATH"
  fi
  # SQLite WAL / shared-memory sidecars.
  rm -f "${CODESEEK_DB_PATH}-wal" "${CODESEEK_DB_PATH}-shm" 2>/dev/null || true
  echo "[local-backend] database will be recreated on startup"

  # --- Cloned repo workspace (sessions checked out by the indexer) ---
  if [[ -d "$CODESEEK_REPO_WORKSPACE" ]]; then
    echo "[local-backend] removing repo workspace: $CODESEEK_REPO_WORKSPACE"
    rm -rf "$CODESEEK_REPO_WORKSPACE"
  fi

  # --- Ingestion temp clones ---
  if [[ -d "$INGESTION_TEMP_CLONE_DIR" ]]; then
    echo "[local-backend] removing ingestion temp dir: $INGESTION_TEMP_CLONE_DIR"
    rm -rf "$INGESTION_TEMP_CLONE_DIR"
  fi

  # --- .rag_ingestion_state.json files generated by CodeSeek inside backend dir only ---
  # (does NOT touch user source repos outside BACKEND_ROOT)
  find "$BACKEND_ROOT" -maxdepth 6 -name ".rag_ingestion_state.json" -print -delete 2>/dev/null || true

  # --- Python caches (optional, speeds up a cold import check) ---
  find "$BACKEND_ROOT" -type d \( -name "__pycache__" -o -name ".pytest_cache" \) \
    -prune -exec rm -rf {} + 2>/dev/null || true

else
  if [[ "$DB_BACKEND" == "sqlite" ]]; then
    echo "[local-backend] using SQLite db: $CODESEEK_SQLITE_PATH"
  else
    # Mask username:password in CODESEEK_DATABASE_URL for logs
    masked_db_url=$(echo "${CODESEEK_DATABASE_URL:-}" | sed -E 's/([^:]+:\/\/)[^@]+@/\1***:***@/')
    echo "[local-backend] preserving Postgres db: $masked_db_url"
  fi
fi

echo "[local-backend] starting backend (port 8000)"

exec ./.venv/bin/uvicorn retrieval.api_service:app --host 0.0.0.0 --port 8000
