#!/usr/bin/env bash
# check_backend_gpu_usage.sh — report which processes are using GPU VRAM,
# with focused warnings for backend Python / uvicorn processes.
#
# Usage:
#   ./scripts/check_backend_gpu_usage.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================================"
echo "  CodeSeek GPU Usage Check"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Full nvidia-smi summary
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found — no NVIDIA GPU detected or drivers not installed."
  echo ""
  echo "If you are on a system without a discrete GPU this is expected."
  exit 0
fi

echo "--- nvidia-smi (GPU totals) ---"
nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits \
  | awk -F',' '{printf "  GPU: %-30s  used=%s MiB  free=%s MiB  total=%s MiB\n", $1, $2, $3, $4}'
echo ""

# ---------------------------------------------------------------------------
# Compute process list
# ---------------------------------------------------------------------------
echo "--- Compute processes ---"
PROC_RAW="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits 2>/dev/null || true)"

if [[ -z "$PROC_RAW" ]]; then
  echo "  (no compute processes using GPU)"
  echo ""
else
  echo "$PROC_RAW" | while IFS=',' read -r pid pname vmem; do
    pid="${pid// /}"
    pname="${pname// /}"
    vmem="${vmem// /}"
    printf "  PID %-7s  VRAM %-6s MiB  %s\n" "$pid" "$vmem" "$pname"
  done
  echo ""
fi

# ---------------------------------------------------------------------------
# Focused analysis: CodeSeek backend + Ollama processes
# ---------------------------------------------------------------------------
echo "--- Analysis ---"

BACKEND_USING_GPU=0
OLLAMA_USING_GPU=0

if [[ -n "$PROC_RAW" ]]; then
  while IFS=',' read -r pid pname vmem; do
    pname="${pname// /}"
    vmem="${vmem// /}"
    vmem_int="${vmem:-0}"
    # Treat anything > 100 MiB as "significant"
    if [[ "$vmem_int" -gt 100 ]] 2>/dev/null; then
      if echo "$pname" | grep -qE "\.venv/bin/python|uvicorn|rag_ingestion/main\.py"; then
        BACKEND_USING_GPU=1
        echo "  ⚠  Backend Python process using ${vmem_int} MiB GPU: $pname"
      fi
      if echo "$pname" | grep -qE "ollama|llama[-_]?server|llama\.cpp"; then
        OLLAMA_USING_GPU=1
        echo "  ℹ  Ollama/llama-server using ${vmem_int} MiB GPU: $pname"
      fi
    fi
  done <<< "$PROC_RAW"
fi

echo ""
if [[ "$BACKEND_USING_GPU" -eq 1 ]]; then
  echo "WARNING: backend Python is using GPU."
  echo "  Restart backend with scripts/run_backend_cpu_embeddings.sh to free VRAM."
  echo ""
  echo "  Kill existing backend first:"
  echo "    pkill -f 'uvicorn' 2>/dev/null || true"
  echo "    pkill -f 'python.*api_service' 2>/dev/null || true"
  echo "    pkill -f 'rag_ingestion/main.py' 2>/dev/null || true"
  echo ""
  echo "  Then restart:"
  echo "    ./scripts/run_backend_cpu_embeddings.sh"
elif [[ "$OLLAMA_USING_GPU" -eq 1 ]]; then
  echo "OK: only Ollama is using GPU (backend Python is GPU-free)"
  echo "  Ollama will be unloaded after indexing when UNLOAD_LOCAL_LLM_AFTER_INDEXING=1."
else
  echo "OK: backend is not using GPU"
fi

echo ""
echo "--- ollama ps ---"
if command -v ollama &>/dev/null; then
  ollama ps 2>/dev/null || echo "  (ollama command failed)"
else
  echo "  (ollama not found in PATH)"
fi
echo ""
echo "============================================================"
