"""GPU and VRAM cleanup utilities for the ingestion pipeline.

All functions are best-effort: they log warnings on failure but never
raise exceptions that would interrupt indexing.
"""

from __future__ import annotations

import gc
import logging
import subprocess

import httpx

logger = logging.getLogger(__name__)


def log_gpu_memory_snapshot(reason: str = "") -> None:
    """Log current GPU memory usage via nvidia-smi.

    Uses two nvidia-smi queries:
      1. Per-compute-process memory (pid, process_name, used_memory)
      2. Per-GPU totals (used, free, total)

    Never raises if nvidia-smi is unavailable.
    """
    tag = f" [{reason}]" if reason else ""
    try:
        proc_out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        proc_lines = proc_out.stdout.strip() or "(no compute processes)"
        gpu_lines = gpu_out.stdout.strip() or "(no GPU found)"
        logger.info(
            "GPU snapshot%s — gpu totals (used/free/total MiB): %s | compute procs: %s",
            tag,
            gpu_lines,
            proc_lines,
        )
    except FileNotFoundError:
        logger.debug("nvidia-smi not found; skipping GPU memory snapshot%s", tag)
    except Exception as exc:
        logger.warning("GPU memory snapshot%s failed: %s", tag, exc)


def clear_python_cuda_cache(reason: str = "") -> None:
    """Free Python-side CUDA memory allocations.

    Steps:
      1. gc.collect() — release any Python objects keeping tensors alive
      2. torch.cuda.empty_cache() — return cached blocks to the OS allocator
      3. torch.cuda.ipc_collect() if available

    Logs before/after allocated and reserved bytes.
    Never raises.
    """
    tag = f" [{reason}]" if reason else ""
    try:
        gc.collect()
    except Exception as exc:
        logger.warning("gc.collect() failed%s: %s", tag, exc)

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        logger.debug("torch not available; skipping CUDA cache clear%s", tag)
        return

    try:
        if not torch.cuda.is_available():
            logger.debug("CUDA not available; skipping cache clear%s", tag)
            return

        before_alloc = torch.cuda.memory_allocated()
        before_reserved = torch.cuda.memory_reserved()

        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

        after_alloc = torch.cuda.memory_allocated()
        after_reserved = torch.cuda.memory_reserved()

        logger.info(
            "CUDA cache cleared%s — allocated: %d→%d bytes, reserved: %d→%d bytes",
            tag,
            before_alloc,
            after_alloc,
            before_reserved,
            after_reserved,
        )
    except Exception as exc:
        logger.warning("CUDA cache clear%s failed: %s", tag, exc)


def unload_ollama_model(model: str, base_url: str = "http://localhost:11434") -> None:
    """Ask Ollama to evict a model from VRAM by sending keep_alive=0.

    Posts to /api/generate with {"model": model, "prompt": "", "keep_alive": 0}.
    Never raises if the request fails or Ollama is not running.
    """
    if not model or not model.strip():
        logger.debug("unload_ollama_model: no model name provided, skipping")
        return

    model = model.strip()
    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": "", "keep_alive": 0}

    try:
        response = httpx.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            logger.info("Ollama model '%s' unloaded (keep_alive=0)", model)
        else:
            logger.warning(
                "Ollama unload for '%s' returned status %s: %s",
                model,
                response.status_code,
                response.text[:200],
            )
    except Exception as exc:
        logger.warning("Failed to unload Ollama model '%s': %s", model, exc)


def cleanup_after_batch() -> None:
    """Free Python-side memory and empty CUDA cache if PyTorch is available."""
    import gc
    gc.collect()
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def ollama_stop_model(model: str, base_url: str = "http://localhost:11434") -> None:
    """Evict a model from Ollama. Calls 'ollama stop <model>' via subprocess,
    and fallback to unload_ollama_model API call.
    """
    if not model or not model.strip():
        return
    model = model.strip()
    
    try:
        # Run 'ollama stop' via subprocess
        subprocess.run(["ollama", "stop", model], capture_output=True, text=True, timeout=10)
        logger.info("Executed 'ollama stop %s'", model)
    except Exception as exc:
        logger.debug("Failed to run 'ollama stop %s': %s", model, exc)
        
    # Also fallback to unload_ollama_model API call (keep_alive=0)
    unload_ollama_model(model, base_url)

