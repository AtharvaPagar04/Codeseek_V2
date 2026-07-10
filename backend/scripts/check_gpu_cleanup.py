"""GPU cleanup validation script.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/check_gpu_cleanup.py

Optional env:
    LOCAL_LLM_UNLOAD_MODEL=qwen2.5-coder:3b-5k
    UNLOAD_OLLAMA=1       — actually send keep_alive=0 to Ollama
    CLEAR_CUDA=1          — run clear_python_cuda_cache()
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from rag_ingestion.utils.gpu_cleanup import (
    clear_python_cuda_cache,
    log_gpu_memory_snapshot,
    unload_ollama_model,
)

model = os.getenv("LOCAL_LLM_UNLOAD_MODEL", "").strip()
do_unload = os.getenv("UNLOAD_OLLAMA", "0").strip() in {"1", "true", "yes", "on"}
do_clear = os.getenv("CLEAR_CUDA", "0").strip() in {"1", "true", "yes", "on"}

print("\n=== GPU Cleanup Check ===\n")

# --- Before snapshot ---
print("[1] GPU memory BEFORE cleanup:")
log_gpu_memory_snapshot("before cleanup")

# --- Optional CUDA cache clear ---
if do_clear:
    print("\n[2] Clearing Python CUDA cache...")
    clear_python_cuda_cache("manual check")
else:
    print("\n[2] Skipping CUDA cache clear (set CLEAR_CUDA=1 to enable)")

# --- Optional Ollama unload ---
if do_unload and model:
    print(f"\n[3] Unloading Ollama model: {model}")
    unload_ollama_model(model)
elif do_unload and not model:
    print("\n[3] UNLOAD_OLLAMA=1 but LOCAL_LLM_UNLOAD_MODEL is not set — skipping")
else:
    print(f"\n[3] Skipping Ollama unload (set UNLOAD_OLLAMA=1 and LOCAL_LLM_UNLOAD_MODEL=<model>)")

# --- After snapshot ---
print("\n[4] GPU memory AFTER cleanup:")
log_gpu_memory_snapshot("after cleanup")

print("\n=== Done ===\n")
