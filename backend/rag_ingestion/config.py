"""Configuration constants for the local ingestion pipeline."""

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def _env_positive_int(name: str) -> int:
    value = os.getenv(name)
    if value is None:
        return 0

    try:
        parsed = int(value)
    except ValueError:
        return 0
    return parsed if parsed > 0 else 0


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = _env_int("QDRANT_PORT", 6333)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "repository_chunks")
RECREATE_COLLECTION_EACH_RUN = _env_bool("QDRANT_RECREATE_COLLECTION", False)

ENABLE_INCREMENTAL_FILE_SKIP = _env_bool(
    "INGESTION_ENABLE_INCREMENTAL_FILE_SKIP",
    True,
)
INGESTION_STATE_FILENAME = ".rag_ingestion_state.json"

EMBEDDING_MODEL = os.getenv("INGESTION_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = _env_int("INGESTION_EMBEDDING_DIM", 384)

MAX_CHUNK_TOKENS = _env_int("INGESTION_MAX_CHUNK_TOKENS", 2048)
BATCH_SIZE = _env_int("INGESTION_BATCH_SIZE", 128)

SLIDING_WINDOW_SIZE = _env_int("INGESTION_SLIDING_WINDOW_SIZE", 100)
SLIDING_OVERLAP = _env_int("INGESTION_SLIDING_OVERLAP", 20)

TEMP_CLONE_DIR = os.getenv("INGESTION_TEMP_CLONE_DIR", "/tmp/rag_ingestion")

ENABLE_LLM_CHUNK_DESCRIPTIONS = _env_bool(
    "ENABLE_LLM_CHUNK_DESCRIPTIONS",
    False,
)
CHUNK_DESCRIPTION_MAX_INPUT_CHARS = _env_int(
    "CHUNK_DESCRIPTION_MAX_INPUT_CHARS",
    1800,
)
CHUNK_DESCRIPTION_MAX_CHUNKS = _env_int(
    "CHUNK_DESCRIPTION_MAX_CHUNKS",
    -1,
)
CHUNK_DESCRIPTION_SLEEP_SECONDS = float(
    os.getenv("CHUNK_DESCRIPTION_SLEEP_SECONDS", "0")
)
CHUNK_DESCRIPTION_RETRY_ON_RATE_LIMIT = _env_bool(
    "CHUNK_DESCRIPTION_RETRY_ON_RATE_LIMIT",
    False,
)


EMBEDDING_INPUT_MAX_CODE_CHARS = _env_int("EMBEDDING_INPUT_MAX_CODE_CHARS", 6000)
EMBEDDING_INPUT_MAX_TOTAL_CHARS = _env_int("EMBEDDING_INPUT_MAX_TOTAL_CHARS", 10000)

ENABLE_CHUNK_LABELS = _env_bool("ENABLE_CHUNK_LABELS", True)

# ---------------------------------------------------------------------------
# GPU / VRAM cleanup after indexing stages
# ---------------------------------------------------------------------------
ENABLE_GPU_CLEANUP_AFTER_STAGES = _env_bool("ENABLE_GPU_CLEANUP_AFTER_STAGES", True)
UNLOAD_LOCAL_LLM_AFTER_INDEXING = _env_bool("UNLOAD_LOCAL_LLM_AFTER_INDEXING", True)
# Unload Ollama model immediately after description generation (before embedding).
# Defaults to False to avoid slow re-loads when descriptions run in batches.
UNLOAD_LOCAL_LLM_AFTER_DESCRIPTIONS = _env_bool("UNLOAD_LOCAL_LLM_AFTER_DESCRIPTIONS", False)
UNLOAD_EMBEDDING_MODEL_AFTER_INDEXING = _env_bool("UNLOAD_EMBEDDING_MODEL_AFTER_INDEXING", True)

# Embedding model device: "cpu" keeps the model off CUDA; "cuda" allows GPU embedding.
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Ollama model name to evict after indexing.  Defaults to the primary LLM model
# so the most common local setup works without extra configuration.
LOCAL_LLM_UNLOAD_MODEL = os.getenv(
    "LOCAL_LLM_UNLOAD_MODEL",
    os.getenv("RETRIEVAL_LOCAL_LLM_PRIMARY_MODEL", ""),
)

# --- CodeSeek Stability / Safe Ingestion Settings ---
CODESEEK_DESCRIPTION_MODEL = os.getenv("CODESEEK_DESCRIPTION_MODEL", "qwen2.5-coder:3b")
CODESEEK_LABEL_MODEL = os.getenv("CODESEEK_LABEL_MODEL", "qwen2.5-coder:3b")

CODESEEK_DESCRIPTION_BATCH_SIZE = _env_int("CODESEEK_DESCRIPTION_BATCH_SIZE", 1)
CODESEEK_EMBEDDING_BATCH_SIZE = _env_int("CODESEEK_EMBEDDING_BATCH_SIZE", 16)
CODESEEK_CHUNK_PROCESS_BATCH_SIZE = _env_int("CODESEEK_CHUNK_PROCESS_BATCH_SIZE", 32)

# Keep EMBEDDING_BATCH_SIZE for backwards compatibility
EMBEDDING_BATCH_SIZE = CODESEEK_EMBEDDING_BATCH_SIZE

CODESEEK_DESCRIPTION_MAX_CHARS = _env_int("CODESEEK_DESCRIPTION_MAX_CHARS", 600)
CODESEEK_DESCRIPTION_MAX_TOKENS = _env_int("CODESEEK_DESCRIPTION_MAX_TOKENS", 160)
CODESEEK_DESCRIPTION_NUM_CTX = _env_int("CODESEEK_DESCRIPTION_NUM_CTX", 4096)

CODESEEK_LLM_ENRICHMENT_NUM_PARALLEL = _env_int("CODESEEK_LLM_ENRICHMENT_NUM_PARALLEL", 1)
CODESEEK_LLM_BATCH_CLEANUP_EVERY = _env_int("CODESEEK_LLM_BATCH_CLEANUP_EVERY", 1)
CODESEEK_OLLAMA_KEEP_ALIVE = os.getenv("CODESEEK_OLLAMA_KEEP_ALIVE", "30s")
CODESEEK_OLLAMA_STOP_MODEL_EVERY = _env_int("CODESEEK_OLLAMA_STOP_MODEL_EVERY", 0)

CODESEEK_DESCRIPTION_COOLDOWN_EVERY = _env_int("CODESEEK_DESCRIPTION_COOLDOWN_EVERY", 200)
CODESEEK_DESCRIPTION_COOLDOWN_SECONDS = _env_int("CODESEEK_DESCRIPTION_COOLDOWN_SECONDS", 60)

CODESEEK_EMBEDDING_COOLDOWN_EVERY = _env_positive_int("CODESEEK_EMBEDDING_COOLDOWN_EVERY")
CODESEEK_EMBEDDING_COOLDOWN_SECONDS = _env_positive_int("CODESEEK_EMBEDDING_COOLDOWN_SECONDS")
