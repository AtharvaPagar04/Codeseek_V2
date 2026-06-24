#!/usr/bin/env bash
# run_backend_cpu_embeddings.sh — start CodeSeek backend with CPU-only embeddings.
#
# On RTX 3050 / 4 GB VRAM systems, running SentenceTransformer on CUDA
# competes with Ollama and triggers CUDA OOM.  This script hides the GPU
# from PyTorch entirely (CUDA_VISIBLE_DEVICES="") so the embedding model
# loads on CPU.  Ollama can still use the GPU for chat / description LLM
# calls because it runs in a separate process.
#
# Usage:
#   ./scripts/run_backend_cpu_embeddings.sh
#
# Options (env overrides):
#   CODESEEK_CLEAN_START=0   — preserve existing session/DB state
#   LOCAL_LLM_UNLOAD_MODEL   — Ollama model to evict after indexing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$BACKEND_ROOT"

echo "[cpu-backend] CodeSeek backend — CPU embedding mode"
echo "[cpu-backend] Backend root: $BACKEND_ROOT"

# ---------------------------------------------------------------------------
# Load .env if present (allows project-specific overrides).
# ---------------------------------------------------------------------------
if [[ -f .env ]]; then
  echo "[cpu-backend] Loading .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# ---------------------------------------------------------------------------
# GPU isolation — hide GPU from PyTorch / SentenceTransformer.
# Ollama is a separate binary and ignores this variable.
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=""
echo "[cpu-backend] CUDA_VISIBLE_DEVICES=\"\" — PyTorch cannot see GPU"

# ---------------------------------------------------------------------------
# Embedding defaults — CPU-safe values.
# ---------------------------------------------------------------------------
export EMBEDDING_DEVICE=cpu
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-4}"

# ---------------------------------------------------------------------------
# GPU cleanup flags (best-effort; never crash indexing).
# ---------------------------------------------------------------------------
export ENABLE_GPU_CLEANUP_AFTER_STAGES="${ENABLE_GPU_CLEANUP_AFTER_STAGES:-1}"
export UNLOAD_EMBEDDING_MODEL_AFTER_INDEXING="${UNLOAD_EMBEDDING_MODEL_AFTER_INDEXING:-1}"
export UNLOAD_LOCAL_LLM_AFTER_INDEXING="${UNLOAD_LOCAL_LLM_AFTER_INDEXING:-1}"

# ---------------------------------------------------------------------------
# Ollama model to evict after indexing.
# ---------------------------------------------------------------------------
export LOCAL_LLM_UNLOAD_MODEL="${LOCAL_LLM_UNLOAD_MODEL:-qwen2.5-coder:3b-5k}"

echo "[cpu-backend] EMBEDDING_DEVICE=cpu  EMBEDDING_BATCH_SIZE=$EMBEDDING_BATCH_SIZE"
echo "[cpu-backend] UNLOAD_LOCAL_LLM_AFTER_INDEXING=$UNLOAD_LOCAL_LLM_AFTER_INDEXING"
echo "[cpu-backend] LOCAL_LLM_UNLOAD_MODEL=$LOCAL_LLM_UNLOAD_MODEL"
echo "[cpu-backend] Starting uvicorn on port 8000..."

exec PYTHONPATH=. .venv/bin/uvicorn retrieval.api_service:app \
  --host 0.0.0.0 \
  --port 8000
