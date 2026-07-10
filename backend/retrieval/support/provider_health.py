"""Provider readiness validation for optional LLM features (e.g. chunk descriptions).

This module is intentionally independent of the ingestion pipeline.
It only performs validation — never starts model loading on its own for remote providers.
For local/Ollama providers it does a lightweight availability probe.
"""

from __future__ import annotations

import httpx

from retrieval.config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_TIMEOUT_SECONDS,
)
from retrieval.stores.provider_store import get_active_provider_credential


class ProviderNotConfiguredError(RuntimeError):
    """Raised when no usable LLM provider is configured for the requesting user."""


class ProviderNotReadyError(RuntimeError):
    """Raised when the provider is configured but not reachable / not loaded."""


def _is_auto_model(model: str | None) -> bool:
    return (model or "").strip().lower() in {"", "auto", "default"}


def _ollama_api_root() -> str:
    base = LOCAL_LLM_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    return base


def _check_ollama_available() -> None:
    """Probe Ollama's /api/ps endpoint. Raises ProviderNotReadyError on failure."""
    url = f"{_ollama_api_root()}/api/ps"
    try:
        response = httpx.get(url, timeout=LOCAL_LLM_TIMEOUT_SECONDS)
        response.raise_for_status()
    except Exception as exc:
        raise ProviderNotReadyError(
            "Local LLM provider is selected but Ollama is not reachable. "
            "Start Ollama and make sure it is listening on the configured base URL "
            f"({LOCAL_LLM_BASE_URL})."
        ) from exc


def _model_matches(requested: str, available: str) -> bool:
    requested = requested.lower()
    available = available.lower()
    if requested == available:
        return True
    req_norm = requested.replace(":latest", "")
    av_norm = available.replace(":latest", "")
    if req_norm == av_norm:
        return True
    if ":" in requested and ":" in available:
        req_base, req_tag = requested.split(":", 1)
        av_base, av_tag = available.split(":", 1)
        if req_base == av_base:
            if req_tag in av_tag or av_tag in req_tag:
                return True
    return False


def _get_ollama_pulled_models() -> list[str]:
    url = f"{_ollama_api_root()}/api/tags"
    try:
        response = httpx.get(url, timeout=LOCAL_LLM_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        return [m.get("name") for m in models if m.get("name")]
    except Exception as exc:
        raise ProviderNotReadyError(
            "Local LLM provider is selected but Ollama is not reachable. "
            "Start Ollama and make sure it is listening on the configured base URL "
            f"({LOCAL_LLM_BASE_URL})."
        ) from exc


def _is_model_available(requested: str, pulled_models: list[str]) -> bool:
    for av in pulled_models:
        if _model_matches(requested, av):
            return True
    return False


def require_llm_ready_for_user(user_id: str) -> dict:
    """Validate that the user's active LLM provider is configured and reachable.

    Returns the decrypted provider credential dict on success.
    Raises ProviderNotConfiguredError or ProviderNotReadyError on failure.
    Never mutates global config.
    """
    credential = get_active_provider_credential(user_id)

    if not credential:
        raise ProviderNotConfiguredError(
            "No active LLM provider configured. "
            "Configure one in API Tokens before enabling LLM chunk descriptions."
        )

    provider = (credential.get("provider") or "").strip().lower()
    model = (credential.get("model") or "").strip()
    api_key = (credential.get("api_key") or "").strip()

    if not provider:
        raise ProviderNotConfiguredError(
            "Active LLM provider credential is missing the provider name."
        )

    if provider == "local":
        # Local/Ollama — API key is not required.
        _check_ollama_available()

        # Decouple chat model check from ingestion check.
        # We verify that the ingestion-safe models are pulled.
        from rag_ingestion.config import CODESEEK_DESCRIPTION_MODEL

        pulled = _get_ollama_pulled_models()

        if not _is_model_available(CODESEEK_DESCRIPTION_MODEL, pulled):
            raise ProviderNotReadyError(
                f"Local ingestion model '{CODESEEK_DESCRIPTION_MODEL}' is not available in Ollama.\n"
                f"Run:\n"
                f"ollama pull {CODESEEK_DESCRIPTION_MODEL}\n"
                f"or choose an installed 3B model in CODESEEK_DESCRIPTION_MODEL."
            )

        return credential

    # Remote providers require an API key.
    if not api_key:
        raise ProviderNotConfiguredError(
            f"Provider '{provider}' is selected but no API key is configured. "
            "Add the API key in API Tokens."
        )

    return credential
