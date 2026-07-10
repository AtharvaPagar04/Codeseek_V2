"""Shared embedding provider configuration and execution helpers."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from typing import Protocol

import httpx
import logging

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_EMBEDDING_MODEL = os.getenv("CODESEEK_LOCAL_EMBEDDING_MODEL", "nomic-embed-text:latest").strip()
try:
    DEFAULT_LOCAL_EMBEDDING_DIMENSIONS = int(os.getenv("CODESEEK_LOCAL_EMBEDDING_DIMENSIONS", "768"))
except ValueError:
    DEFAULT_LOCAL_EMBEDDING_DIMENSIONS = 768
DEFAULT_EMBEDDING_PROVIDER = "local"
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 60.0
SUPPORTED_EMBEDDING_PROVIDERS = frozenset({"local", "openai_compatible"})

OPENAI_COMPATIBLE_EMBEDDING_MODELS = {
    "openai/text-embedding-3-small": {
        "id": "openai/text-embedding-3-small",
        "label": "OpenAI text-embedding-3-small — Recommended",
        "recommended": True,
        "default_dimensions": 1536,
        "allowed_dimensions": [512, 1536],
        "supports_dimension_parameter": True,
        "notes": "Recommended default for CodeSeek.",
    },
    "text-embedding-3-small": {
        "id": "text-embedding-3-small",
        "label": "OpenAI text-embedding-3-small — Secondary Fallback",
        "recommended": False,
        "default_dimensions": 1536,
        "allowed_dimensions": [512, 1536],
        "supports_dimension_parameter": True,
        "notes": "Secondary fallback.",
    },
    "openai/text-embedding-3-large": {
        "id": "openai/text-embedding-3-large",
        "label": "OpenAI text-embedding-3-large — Higher quality",
        "recommended": False,
        "default_dimensions": 3072,
        "allowed_dimensions": [256, 1024, 3072],
        "supports_dimension_parameter": True,
        "notes": "Higher quality, larger vectors.",
    },
    "text-embedding-3-large": {
        "id": "text-embedding-3-large",
        "label": "OpenAI text-embedding-3-large — Higher quality (Fallback)",
        "recommended": False,
        "default_dimensions": 3072,
        "allowed_dimensions": [256, 1024, 3072],
        "supports_dimension_parameter": True,
        "notes": "Higher quality, larger vectors (fallback).",
    },
    "openai/text-embedding-ada-002": {
        "id": "openai/text-embedding-ada-002",
        "label": "OpenAI text-embedding-ada-002 — Legacy",
        "recommended": False,
        "default_dimensions": 1536,
        "allowed_dimensions": [1536],
        "supports_dimension_parameter": False,
        "notes": "Legacy fallback.",
    },
    "text-embedding-ada-002": {
        "id": "text-embedding-ada-002",
        "label": "OpenAI text-embedding-ada-002 — Legacy (Fallback)",
        "recommended": False,
        "default_dimensions": 1536,
        "allowed_dimensions": [1536],
        "supports_dimension_parameter": False,
        "notes": "Legacy fallback.",
    },
    "google/gemini-embedding-001": {
        "id": "google/gemini-embedding-001",
        "label": "Google gemini-embedding-001",
        "recommended": False,
        "default_dimensions": 768,
        "allowed_dimensions": [],
        "supports_dimension_parameter": False,
        "notes": "Google embedding model. Auto dimensions recommended.",
    },
    "google/gemini-embedding-2-preview": {
        "id": "google/gemini-embedding-2-preview",
        "label": "Google gemini-embedding-2-preview",
        "recommended": False,
        "default_dimensions": 768,
        "allowed_dimensions": [],
        "supports_dimension_parameter": False,
        "notes": "Google embedding model. Auto dimensions recommended.",
    },
    "google/text-embedding-004": {
        "id": "google/text-embedding-004",
        "label": "Google text-embedding-004",
        "recommended": False,
        "default_dimensions": 768,
        "allowed_dimensions": [],
        "supports_dimension_parameter": False,
        "notes": "Google embedding model. Auto dimensions recommended.",
    },
}

def get_openai_compatible_embedding_model_options() -> list[dict]:
    return list(OPENAI_COMPATIBLE_EMBEDDING_MODELS.values())

_LOCAL_MODELS: dict[tuple[str, str], object] = {}
_LOCAL_MODEL_LOCK = threading.Lock()


class EmbeddingProviderError(RuntimeError):
    """Base error for embedding provider failures."""


class EmbeddingConfigurationError(EmbeddingProviderError):
    """Raised when the configured embedding provider is invalid."""


class EmbeddingRequestError(EmbeddingProviderError):
    """Raised when an embedding request fails upstream."""


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_positive_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_positive_float(name: str, default: float) -> float:
    value = _env_float(name, default)
    return value if value > 0 else default


def normalize_embedding_base_url(base_url: str) -> str:
    return str(base_url or "").strip().rstrip("/")


def build_embeddings_endpoint(base_url: str) -> str:
    normalized = normalize_embedding_base_url(base_url)
    if not normalized:
        return ""
    return f"{normalized}/embeddings"


def _sanitize_message(message: str, *, api_key: str) -> str:
    sanitized = str(message or "").strip()
    if api_key:
        sanitized = sanitized.replace(api_key, "*****")
    return sanitized


@dataclass(frozen=True)
class EmbeddingProviderConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    batch_size: int
    timeout_seconds: float
    dimensions: int
    local_model: str
    local_device: str
    source: str = "default/env"

    @property
    def effective_model(self) -> str:
        return self.local_model if self.provider == "local" else self.model

    @property
    def normalized_base_url(self) -> str:
        return self.base_url if self.provider != "local" else ""


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    dimensions: int

    def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        ...

    def embed_query(self, text: str, *, prefix: str = "") -> list[float]:
        ...


class SentenceTransformersEmbeddingProvider:
    provider_name = "local"

    def __init__(self, config: EmbeddingProviderConfig):
        self._config = config
        self.model_name = config.local_model
        self.dimensions = config.dimensions or DEFAULT_LOCAL_EMBEDDING_DIMENSIONS

    def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        model = _get_local_model(self._config.local_model, self._config.local_device)
        embeddings = model.encode(
            texts,
            batch_size=batch_size or self._config.batch_size,
            show_progress_bar=show_progress_bar,
        )
        vectors = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        normalized: list[list[float]] = []
        for vector in vectors:
            normalized.append([float(value) for value in vector])
        if normalized:
            self.dimensions = len(normalized[0])
        return normalized

    def embed_query(self, text: str, *, prefix: str = "") -> list[float]:
        return self.embed_texts([f"{prefix}{text}"], batch_size=1, show_progress_bar=False)[0]


class OpenAICompatibleEmbeddingProvider:
    provider_name = "openai_compatible"

    def __init__(self, config: EmbeddingProviderConfig):
        self._config = config
        self.model_name = config.model
        self.dimensions = config.dimensions
        self._endpoint = build_embeddings_endpoint(config.base_url)

    def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        del show_progress_bar
        if not texts:
            return []
        size = batch_size or self._config.batch_size
        if size < 1:
            size = 1
        outputs: list[list[float]] = []
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            outputs.extend(self._embed_batch(batch))
        return outputs

    def embed_query(self, text: str, *, prefix: str = "") -> list[float]:
        return self._embed_batch([f"{prefix}{text}"])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, object] = {
            "model": self._config.model,
            "input": texts,
        }
        
        logger.debug(
            "[embedding.test.debug] embedding_provider=openai_compatible base_url=%s endpoint=%s model=%s input_type=%s input_count=%d dimensions_sent=false api_key_present=%s",
            self._config.base_url,
            self._endpoint,
            self._config.model,
            type(texts).__name__,
            len(texts),
            str(bool(self._config.api_key)).lower(),
        )

        try:
            response = httpx.post(
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._config.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            try:
                body_text = exc.response.text
                try:
                    data = exc.response.json()
                    msg = data.get("error", {}).get("message")
                    if msg:
                        body_text = str(msg)
                except Exception:
                    pass
                body_text = body_text.strip()
                if len(body_text) > 500:
                    body_text = body_text[:497] + "..."
                safe_text = _sanitize_message(body_text, api_key=self._config.api_key)
                detail = f": {safe_text}" if safe_text else ""
            except Exception:
                detail = ""
                
            raise EmbeddingRequestError(
                f"OpenAI-compatible embedding request failed with status {status}{detail}"
            ) from exc
        except Exception as exc:
            raise EmbeddingRequestError(
                _sanitize_message(
                    f"OpenAI-compatible embedding request failed: {exc}",
                    api_key=self._config.api_key,
                )
            ) from exc

        try:
            data = response.json()
        except Exception as exc:
            raise EmbeddingRequestError("Embedding provider returned invalid JSON.") from exc

        items = data.get("data")
        if not isinstance(items, list):
            raise EmbeddingRequestError("Embedding provider response is missing a data array.")
        if len(items) != len(texts):
            raise EmbeddingRequestError(
                f"Embedding provider returned {len(items)} vectors for {len(texts)} inputs."
            )

        ordered = sorted(
            enumerate(items),
            key=lambda pair: int(pair[1].get("index", pair[0])),
        )
        vectors: list[list[float]] = []
        for _, item in ordered:
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise EmbeddingRequestError("Embedding provider returned an invalid embedding vector.")
            try:
                vector = [float(value) for value in embedding]
            except Exception as exc:
                raise EmbeddingRequestError("Embedding provider returned a non-numeric embedding vector.") from exc
            vectors.append(vector)

        if vectors:
            actual_dimensions = len(vectors[0])
            self.dimensions = actual_dimensions
        return vectors


def _fetch_saved_embedding_config() -> dict | None:
    try:
        from retrieval.db import db_cursor
        from retrieval.stores.crypto_store import decrypt_secret
        with db_cursor() as (_conn, cursor):
            row = cursor.execute("SELECT * FROM user_embedding_configs ORDER BY created_at ASC LIMIT 1").fetchone()
            if row:
                return {
                    "provider": row["provider"],
                    "base_url": row["base_url"],
                    "model": row["model"],
                    "api_key": decrypt_secret(row["encrypted_api_key"]) if row["encrypted_api_key"] else "",
                    "dimensions": row["dimensions"],
                    "timeout_seconds": row["timeout_seconds"],
                    "batch_size": row["batch_size"],
                }
    except Exception:
        pass
    return None


def resolve_embedding_config() -> EmbeddingProviderConfig:
    """
    Return effective embedding config from:
    1. saved setting if configured
    2. env
    3. local defaults
    """
    saved = _fetch_saved_embedding_config()
    
    local_model = os.getenv("INGESTION_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL).strip() or DEFAULT_LOCAL_EMBEDDING_MODEL
    local_dimensions = _env_positive_int("INGESTION_EMBEDDING_DIM", DEFAULT_LOCAL_EMBEDDING_DIMENSIONS)
    local_device = os.getenv("EMBEDDING_DEVICE", "cpu").strip() or "cpu"

    if saved:
        provider = saved["provider"]
        if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            provider = DEFAULT_EMBEDDING_PROVIDER
            
        config = EmbeddingProviderConfig(
            provider=provider,
            base_url=normalize_embedding_base_url(saved.get("base_url", "")),
            api_key=saved.get("api_key", ""),
            model=saved.get("model", ""),
            batch_size=saved.get("batch_size") if saved.get("batch_size") else _env_positive_int("CODESEEK_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE),
            timeout_seconds=saved.get("timeout_seconds") if saved.get("timeout_seconds") else _env_positive_float("CODESEEK_EMBEDDING_TIMEOUT_SECONDS", DEFAULT_EMBEDDING_TIMEOUT_SECONDS),
            dimensions=saved.get("dimensions") if saved.get("dimensions") else (local_dimensions if provider == "local" else 0),
            local_model=local_model,
            local_device=local_device,
            source="stored/ui",
        )
    else:
        config = get_embedding_provider_config()

    if config.provider == "openai_compatible":
        missing: list[str] = []
        if not config.base_url:
            missing.append("CODESEEK_EMBEDDING_BASE_URL (or saved base_url)")
        if not config.api_key:
            missing.append("CODESEEK_EMBEDDING_API_KEY (or saved api_key)")
        if not config.model:
            missing.append("CODESEEK_EMBEDDING_MODEL (or saved model)")
        if missing:
            raise EmbeddingConfigurationError(
                "OpenAI-compatible embeddings require: " + ", ".join(missing)
            )

        model_lower = config.model.lower()
        if "deepseek" in model_lower or "gpt-" in model_lower or "claude-" in model_lower or "gemini-" in model_lower:
            raise EmbeddingConfigurationError(
                f"Model '{config.model}' appears to be a chat model. OpenAI-compatible embeddings require a dedicated embedding model (e.g., text-embedding-3-small)."
            )

    return config


def get_embedding_provider_config() -> EmbeddingProviderConfig:
    provider = os.getenv("CODESEEK_EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER).strip().lower()
    if not provider:
        provider = DEFAULT_EMBEDDING_PROVIDER
    if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise EmbeddingConfigurationError(
            f"Unsupported CODESEEK_EMBEDDING_PROVIDER '{provider}'. "
            "Expected 'local' or 'openai_compatible'."
        )

    local_model = os.getenv("INGESTION_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL).strip() or DEFAULT_LOCAL_EMBEDDING_MODEL
    local_dimensions = _env_positive_int("INGESTION_EMBEDDING_DIM", DEFAULT_LOCAL_EMBEDDING_DIMENSIONS)
    local_device = os.getenv("EMBEDDING_DEVICE", "cpu").strip() or "cpu"

    config = EmbeddingProviderConfig(
        provider=provider,
        base_url=normalize_embedding_base_url(os.getenv("CODESEEK_EMBEDDING_BASE_URL", "")),
        api_key=os.getenv("CODESEEK_EMBEDDING_API_KEY", "").strip(),
        model=os.getenv("CODESEEK_EMBEDDING_MODEL", "").strip(),
        batch_size=_env_positive_int("CODESEEK_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE),
        timeout_seconds=_env_positive_float("CODESEEK_EMBEDDING_TIMEOUT_SECONDS", DEFAULT_EMBEDDING_TIMEOUT_SECONDS),
        dimensions=_env_positive_int(
            "CODESEEK_EMBEDDING_DIMENSIONS",
            local_dimensions if provider == "local" else 0,
        ),
        local_model=local_model,
        local_device=local_device,
        source="default/env",
    )

    if provider == "openai_compatible":
        missing: list[str] = []
        if not config.base_url:
            missing.append("CODESEEK_EMBEDDING_BASE_URL")
        if not config.api_key:
            missing.append("CODESEEK_EMBEDDING_API_KEY")
        if not config.model:
            missing.append("CODESEEK_EMBEDDING_MODEL")
        if missing:
            raise EmbeddingConfigurationError(
                "OpenAI-compatible embeddings require: " + ", ".join(missing)
            )

        model_lower = config.model.lower()
        if "deepseek" in model_lower or "gpt-" in model_lower or "claude-" in model_lower or "gemini-" in model_lower:
            raise EmbeddingConfigurationError(
                f"Model '{config.model}' appears to be a chat model. OpenAI-compatible embeddings require a dedicated embedding model (e.g., text-embedding-3-small)."
            )

    return config


def _is_ollama_local_config(config: EmbeddingProviderConfig) -> bool:
    base_url = (config.base_url or "").lower()
    return bool(base_url) and (
        "localhost:11434" in base_url
        or "127.0.0.1:11434" in base_url
        or base_url.endswith(":11434")
        or "ollama" in base_url
    )

class OllamaEmbeddingProvider:
    provider_name = "local"

    def __init__(self, config: EmbeddingProviderConfig):
        self.base_url = config.base_url.rstrip("/")
        self.model_name = config.model or config.local_model
        self.dimensions = config.dimensions or 0
        self.timeout_seconds = config.timeout_seconds
        self.batch_size = config.batch_size or 1

    def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        # Ollama /api/embed takes a list of strings
        payload = {
            "model": self.model_name,
            "input": texts,
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/api/embed", json=payload)
            response.raise_for_status()
            data = response.json()

        vectors = data.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            # Fallback to older /api/embeddings endpoint sequentially if /api/embed fails
            vectors = [self.embed_query(text) for text in texts]

        if self.dimensions and len(vectors) > 0 and len(vectors[0]) != self.dimensions:
            raise EmbeddingConfigurationError(
                f"Ollama embedding dimension mismatch: expected {self.dimensions}, got {len(vectors[0])}"
            )

        if len(vectors) > 0:
            self.dimensions = len(vectors[0])
        return vectors

    def embed_query(self, text: str, *, prefix: str = "") -> list[float]:
        payload = {
            "model": self.model_name,
            "prompt": f"{prefix}{text}",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()

        vector = data.get("embedding")
        if not vector:
            raise EmbeddingProviderError("Ollama embedding response missing embedding")

        if self.dimensions and len(vector) != self.dimensions:
            raise EmbeddingConfigurationError(
                f"Ollama embedding dimension mismatch: expected {self.dimensions}, got {len(vector)}"
            )

        self.dimensions = len(vector)
        return vector

def get_embedding_provider(config: EmbeddingProviderConfig | None = None) -> EmbeddingProvider:
    resolved = config or resolve_embedding_config()
    if resolved.provider == "local":
        if _is_ollama_local_config(resolved):
            return OllamaEmbeddingProvider(resolved)
        return SentenceTransformersEmbeddingProvider(resolved)
    if resolved.provider == "openai_compatible":
        return OpenAICompatibleEmbeddingProvider(resolved)
    raise EmbeddingConfigurationError(f"Unsupported embedding provider '{resolved.provider}'.")


def build_embedding_config_hash(
    *,
    provider: str,
    base_url: str,
    model: str,
    dimensions: int,
) -> str:
    payload = {
        "provider": str(provider or "").strip().lower(),
        "base_url": normalize_embedding_base_url(base_url),
        "model": str(model or "").strip(),
        "dimensions": int(dimensions or 0),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def current_embedding_metadata(
    *,
    resolved_dimensions: int | None = None,
    dimensions_fallback: int | None = None,
) -> dict[str, object]:
    config = resolve_embedding_config()
    dimensions = int(resolved_dimensions or 0)
    if dimensions <= 0:
        dimensions = int(config.dimensions or 0)
    if dimensions <= 0:
        dimensions = int(dimensions_fallback or 0)

    metadata = {
        "embedding_provider": config.provider,
        "embedding_base_url": config.normalized_base_url,
        "embedding_model": config.effective_model,
        "embedding_dimensions": dimensions,
        "embedding_config_source": getattr(config, "source", "unknown"),
    }
    metadata["embedding_config_hash"] = build_embedding_config_hash(
        provider=str(metadata["embedding_provider"]),
        base_url=str(metadata["embedding_base_url"]),
        model=str(metadata["embedding_model"]),
        dimensions=int(metadata["embedding_dimensions"] or 0),
    )
    return metadata


def unload_local_embedding_model() -> None:
    with _LOCAL_MODEL_LOCK:
        _LOCAL_MODELS.clear()


def _get_local_model(model_name: str, device: str):
    key = (model_name, device)
    with _LOCAL_MODEL_LOCK:
        model = _LOCAL_MODELS.get(key)
        if model is not None:
            return model
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model = SentenceTransformer(model_name, device=device)
        _LOCAL_MODELS[key] = model
        return model
