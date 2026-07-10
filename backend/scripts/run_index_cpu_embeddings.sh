#!/usr/bin/env bash
# run_index_cpu_embeddings.sh — run the ingestion pipeline with CPU-only embeddings.
#
# Usage:
#   ./scripts/run_index_cpu_embeddings.sh /path/to/repo collection_name
#
# Example:
#   ./scripts/run_index_cpu_embeddings.sh /home/arch/DEV/Portfolio my_portfolio_collection
#
# Optional env:
#   ENABLE_LLM_CHUNK_DESCRIPTIONS=1   — enable Ollama descriptions
#   LOCAL_LLM_UNLOAD_MODEL=<model>    — Ollama model to evict after indexing
#   QDRANT_RECREATE_COLLECTION=0      — keep existing collection (incremental)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$BACKEND_ROOT"

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
REPO_PATH="${1:-}"
COLLECTION="${2:-}"

if [[ -z "$REPO_PATH" ]]; then
  echo "ERROR: repository path is required." >&2
  echo "" >&2
  echo "Usage: $0 /path/to/repo collection_name" >&2
  exit 1
fi

if [[ -z "$COLLECTION" ]]; then
  echo "ERROR: collection name is required." >&2
  echo "" >&2
  echo "Usage: $0 /path/to/repo collection_name" >&2
  exit 1
fi

echo "[cpu-index] Repository : $REPO_PATH"
echo "[cpu-index] Collection : $COLLECTION"

# ---------------------------------------------------------------------------
# Load .env if present.
# ---------------------------------------------------------------------------
if [[ -f .env ]]; then
  echo "[cpu-index] Loading .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# ---------------------------------------------------------------------------
# GPU isolation — hide GPU from PyTorch / SentenceTransformer.
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=""
echo "[cpu-index] CUDA_VISIBLE_DEVICES=\"\" — PyTorch cannot see GPU"

# ---------------------------------------------------------------------------
# Embedding settings.
# ---------------------------------------------------------------------------
export EMBEDDING_DEVICE=cpu
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-4}"

# ---------------------------------------------------------------------------
# GPU cleanup flags.
# ---------------------------------------------------------------------------
export ENABLE_GPU_CLEANUP_AFTER_STAGES="${ENABLE_GPU_CLEANUP_AFTER_STAGES:-1}"
export UNLOAD_EMBEDDING_MODEL_AFTER_INDEXING="${UNLOAD_EMBEDDING_MODEL_AFTER_INDEXING:-1}"
export UNLOAD_LOCAL_LLM_AFTER_INDEXING="${UNLOAD_LOCAL_LLM_AFTER_INDEXING:-1}"

# ---------------------------------------------------------------------------
# Indexing behavior defaults.
# ---------------------------------------------------------------------------
export CODESEEK_STRICT_ISOLATION="${CODESEEK_STRICT_ISOLATION:-0}"
export INGESTION_ENABLE_INCREMENTAL_FILE_SKIP="${INGESTION_ENABLE_INCREMENTAL_FILE_SKIP:-0}"
export QDRANT_RECREATE_COLLECTION="${QDRANT_RECREATE_COLLECTION:-1}"

# ---------------------------------------------------------------------------
# Ollama model to evict after indexing.
# ---------------------------------------------------------------------------
export LOCAL_LLM_UNLOAD_MODEL="${LOCAL_LLM_UNLOAD_MODEL:-qwen2.5-coder:3b-5k}"

echo "[cpu-index] EMBEDDING_DEVICE=cpu  EMBEDDING_BATCH_SIZE=$EMBEDDING_BATCH_SIZE"
echo "[cpu-index] QDRANT_RECREATE_COLLECTION=$QDRANT_RECREATE_COLLECTION"
echo "[cpu-index] UNLOAD_LOCAL_LLM_AFTER_INDEXING=$UNLOAD_LOCAL_LLM_AFTER_INDEXING"
echo "[cpu-index] LOCAL_LLM_UNLOAD_MODEL=$LOCAL_LLM_UNLOAD_MODEL"
echo "[cpu-index] Starting ingestion pipeline..."
echo ""

PYTHONPATH=. .venv/bin/python rag_ingestion/main.py \
  "$REPO_PATH" \
  --collection "$COLLECTION"
