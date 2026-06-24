from __future__ import annotations

import os

import pytest

from retrieval.support.embedding_provider import (
    EmbeddingConfigurationError,
    LocalEmbeddingProvider,
    build_embedding_config_hash,
    current_embedding_metadata,
    get_embedding_provider,
    get_embedding_provider_config,
    normalize_embedding_base_url,
)


def test_default_embedding_provider_is_local(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CODESEEK_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("CODESEEK_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("CODESEEK_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("CODESEEK_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("INGESTION_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("INGESTION_EMBEDDING_DIM", "384")

    config = get_embedding_provider_config()

    assert config.provider == "local"
    assert config.local_model == "BAAI/bge-small-en-v1.5"
    assert config.dimensions == 384
    assert isinstance(get_embedding_provider(config), LocalEmbeddingProvider)


@pytest.mark.parametrize(
    ("missing_key", "expected_fragment"),
    [
        ("CODESEEK_EMBEDDING_BASE_URL", "CODESEEK_EMBEDDING_BASE_URL"),
        ("CODESEEK_EMBEDDING_API_KEY", "CODESEEK_EMBEDDING_API_KEY"),
        ("CODESEEK_EMBEDDING_MODEL", "CODESEEK_EMBEDDING_MODEL"),
    ],
)
def test_cloud_provider_requires_base_url_api_key_and_model(
    monkeypatch: pytest.MonkeyPatch,
    missing_key: str,
    expected_fragment: str,
):
    monkeypatch.setenv("CODESEEK_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("CODESEEK_EMBEDDING_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("CODESEEK_EMBEDDING_API_KEY", "secret-key")
    monkeypatch.setenv("CODESEEK_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.delenv(missing_key, raising=False)

    with pytest.raises(EmbeddingConfigurationError) as exc_info:
        get_embedding_provider_config()

    assert expected_fragment in str(exc_info.value)
    assert "secret-key" not in str(exc_info.value)


def test_base_url_normalization():
    assert normalize_embedding_base_url("https://api.example.com/v1/") == "https://api.example.com/v1"
    assert normalize_embedding_base_url(" https://api.example.com/v1 ") == "https://api.example.com/v1"


def test_config_hash_excludes_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CODESEEK_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("CODESEEK_EMBEDDING_BASE_URL", "https://api.example.com/v1/")
    monkeypatch.setenv("CODESEEK_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setenv("CODESEEK_EMBEDDING_DIMENSIONS", "1536")

    monkeypatch.setenv("CODESEEK_EMBEDDING_API_KEY", "first-secret")
    first = current_embedding_metadata()
    monkeypatch.setenv("CODESEEK_EMBEDDING_API_KEY", "second-secret")
    second = current_embedding_metadata()

    assert first["embedding_config_hash"] == second["embedding_config_hash"]
    assert "secret" not in first["embedding_config_hash"]


def test_config_hash_changes_for_provider_model_and_dimensions():
    base = build_embedding_config_hash(
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        model="openai/text-embedding-3-small",
        dimensions=1536,
    )

    changed_provider = build_embedding_config_hash(
        provider="local",
        base_url="",
        model="BAAI/bge-small-en-v1.5",
        dimensions=384,
    )
    changed_model = build_embedding_config_hash(
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        model="text-embedding-3-large",
        dimensions=1536,
    )
    changed_dimensions = build_embedding_config_hash(
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        model="openai/text-embedding-3-small",
        dimensions=3072,
    )

    assert base != changed_provider
    assert base != changed_model
    assert base != changed_dimensions


def test_local_provider_still_selected_by_default(monkeypatch: pytest.MonkeyPatch):
    for key in (
        "CODESEEK_EMBEDDING_PROVIDER",
        "CODESEEK_EMBEDDING_BASE_URL",
        "CODESEEK_EMBEDDING_API_KEY",
        "CODESEEK_EMBEDDING_MODEL",
        "CODESEEK_EMBEDDING_DIMENSIONS",
    ):
        monkeypatch.delenv(key, raising=False)

    provider = get_embedding_provider()

    assert provider.provider_name == "local"
