"""Unit tests for GPU cleanup utilities and related config/embedder changes."""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_torch(cuda_available: bool = False):
    """Return a minimal fake torch module."""
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: cuda_available
    cuda.memory_allocated = lambda: 1024
    cuda.memory_reserved = lambda: 2048
    cuda.empty_cache = MagicMock()
    cuda.ipc_collect = MagicMock()
    torch.cuda = cuda
    return torch


# ---------------------------------------------------------------------------
# clear_python_cuda_cache
# ---------------------------------------------------------------------------

class TestClearPythonCudaCache:
    """clear_python_cuda_cache must never crash."""

    def test_no_crash_when_torch_unavailable(self):
        """If torch is not installed, function must return silently."""
        with patch.dict(sys.modules, {"torch": None}):
            # Re-import to pick up the patched modules map
            import importlib
            import rag_ingestion.utils.gpu_cleanup as mod
            importlib.reload(mod)
            # Should not raise
            mod.clear_python_cuda_cache("test: no torch")

    def test_no_crash_when_cuda_unavailable(self):
        """If torch is available but CUDA is not, function must return silently."""
        fake_torch = _make_fake_torch(cuda_available=False)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            import importlib
            import rag_ingestion.utils.gpu_cleanup as mod
            importlib.reload(mod)
            mod.clear_python_cuda_cache("test: no cuda")

    def test_calls_empty_cache_when_cuda_available(self):
        """If CUDA is available, empty_cache() and ipc_collect() must be called."""
        fake_torch = _make_fake_torch(cuda_available=True)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            import importlib
            import rag_ingestion.utils.gpu_cleanup as mod
            importlib.reload(mod)
            mod.clear_python_cuda_cache("test: with cuda")

        fake_torch.cuda.empty_cache.assert_called_once()
        fake_torch.cuda.ipc_collect.assert_called_once()

    def test_no_crash_on_unexpected_exception(self):
        """An unexpected exception inside the function must be swallowed."""
        fake_torch = _make_fake_torch(cuda_available=True)
        fake_torch.cuda.empty_cache.side_effect = RuntimeError("boom")
        with patch.dict(sys.modules, {"torch": fake_torch}):
            import importlib
            import rag_ingestion.utils.gpu_cleanup as mod
            importlib.reload(mod)
            # Must not propagate the RuntimeError
            mod.clear_python_cuda_cache("test: exception")


# ---------------------------------------------------------------------------
# unload_ollama_model
# ---------------------------------------------------------------------------

class TestUnloadOllamaModel:
    """unload_ollama_model must never crash and must send keep_alive=0."""

    def test_no_crash_on_connection_error(self):
        """Connection failures must be logged but not re-raised."""
        import rag_ingestion.utils.gpu_cleanup as mod

        with patch("rag_ingestion.utils.gpu_cleanup.httpx") as mock_httpx:
            mock_httpx.post.side_effect = ConnectionRefusedError("no server")
            # Should not raise
            mod.unload_ollama_model("qwen2.5-coder:3b-8k")

    def test_sends_keep_alive_zero(self):
        """Must POST to /api/generate with keep_alive=0."""
        import rag_ingestion.utils.gpu_cleanup as mod

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("rag_ingestion.utils.gpu_cleanup.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_response
            mod.unload_ollama_model(
                "qwen2.5-coder:3b-8k", base_url="http://localhost:11434"
            )
            mock_httpx.post.assert_called_once_with(
                "http://localhost:11434/api/generate",
                json={"model": "qwen2.5-coder:3b-8k", "prompt": "", "keep_alive": 0},
                timeout=30,
            )

    def test_empty_model_name_is_skipped(self):
        """An empty model string must return without making any HTTP call."""
        import rag_ingestion.utils.gpu_cleanup as mod

        with patch("rag_ingestion.utils.gpu_cleanup.httpx") as mock_httpx:
            mod.unload_ollama_model("")
            mod.unload_ollama_model("   ")
            mock_httpx.post.assert_not_called()

    def test_non_200_response_does_not_raise(self):
        """A non-200 HTTP response must be logged and not re-raised."""
        import rag_ingestion.utils.gpu_cleanup as mod

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "internal server error"

        with patch("rag_ingestion.utils.gpu_cleanup.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_response
            mod.unload_ollama_model("some-model")  # must not raise


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """GPU cleanup config keys must have safe defaults."""

    def test_embedding_device_default_is_cpu(self):
        """EMBEDDING_DEVICE must default to 'cpu'."""
        import os
        env_backup = os.environ.pop("EMBEDDING_DEVICE", None)
        try:
            import importlib
            import rag_ingestion.config as cfg
            importlib.reload(cfg)
            assert cfg.EMBEDDING_DEVICE == "cpu", f"Expected 'cpu', got {cfg.EMBEDDING_DEVICE!r}"
        finally:
            if env_backup is not None:
                os.environ["EMBEDDING_DEVICE"] = env_backup

    def test_embedding_batch_size_default_is_16(self):
        """EMBEDDING_BATCH_SIZE must default to 16."""
        import os
        env_backup = os.environ.pop("EMBEDDING_BATCH_SIZE", None)
        try:
            import importlib
            import rag_ingestion.config as cfg
            importlib.reload(cfg)
            assert cfg.EMBEDDING_BATCH_SIZE == 16, f"Expected 16, got {cfg.EMBEDDING_BATCH_SIZE}"
        finally:
            if env_backup is not None:
                os.environ["EMBEDDING_BATCH_SIZE"] = env_backup

    def test_gpu_cleanup_enabled_by_default(self):
        """ENABLE_GPU_CLEANUP_AFTER_STAGES must default to True."""
        import os
        env_backup = os.environ.pop("ENABLE_GPU_CLEANUP_AFTER_STAGES", None)
        try:
            import importlib
            import rag_ingestion.config as cfg
            importlib.reload(cfg)
            assert cfg.ENABLE_GPU_CLEANUP_AFTER_STAGES is True
        finally:
            if env_backup is not None:
                os.environ["ENABLE_GPU_CLEANUP_AFTER_STAGES"] = env_backup

    def test_unload_llm_after_descriptions_disabled_by_default(self):
        """UNLOAD_LOCAL_LLM_AFTER_DESCRIPTIONS must default to False."""
        import os
        env_backup = os.environ.pop("UNLOAD_LOCAL_LLM_AFTER_DESCRIPTIONS", None)
        try:
            import importlib
            import rag_ingestion.config as cfg
            importlib.reload(cfg)
            assert cfg.UNLOAD_LOCAL_LLM_AFTER_DESCRIPTIONS is False
        finally:
            if env_backup is not None:
                os.environ["UNLOAD_LOCAL_LLM_AFTER_DESCRIPTIONS"] = env_backup


# ---------------------------------------------------------------------------
# unload_embedding_model
# ---------------------------------------------------------------------------

class TestUnloadEmbeddingModel:
    """unload_embedding_model must clear the global cache and call gc."""

    def test_clears_global_model_ref(self):
        """After calling unload_embedding_model, _model must be None."""
        import rag_ingestion.stages.embedder as emb

        # Inject a fake model so there is something to unload
        emb._model = MagicMock()

        with patch("rag_ingestion.stages.embedder.clear_python_cuda_cache") as mock_clear:
            emb.unload_embedding_model()
            assert emb._model is None
            mock_clear.assert_called_once_with("after embedding model unload")

    def test_no_op_when_model_not_loaded(self):
        """unload_embedding_model must not crash when model is already None."""
        import rag_ingestion.stages.embedder as emb
        emb._model = None

        with patch("rag_ingestion.stages.embedder.clear_python_cuda_cache") as mock_clear:
            emb.unload_embedding_model()
            mock_clear.assert_not_called()


# ---------------------------------------------------------------------------
# log_gpu_memory_snapshot
# ---------------------------------------------------------------------------

class TestLogGpuMemorySnapshot:
    """log_gpu_memory_snapshot must never crash."""

    def test_no_crash_when_nvidia_smi_missing(self):
        """If nvidia-smi is not installed, function must return silently."""
        import subprocess
        import rag_ingestion.utils.gpu_cleanup as mod

        with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
            mod.log_gpu_memory_snapshot("test: no nvidia-smi")

    def test_no_crash_on_generic_error(self):
        """Any unexpected exception must be caught and logged."""
        import rag_ingestion.utils.gpu_cleanup as mod

        with patch("subprocess.run", side_effect=OSError("unexpected")):
            mod.log_gpu_memory_snapshot("test: generic error")
