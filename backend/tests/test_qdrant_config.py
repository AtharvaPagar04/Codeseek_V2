import os
from unittest.mock import patch, MagicMock

import pytest

from retrieval.support.qdrant_config import get_qdrant_settings, create_qdrant_client


@pytest.fixture
def clean_env():
    # Remove all Qdrant related environment variables
    env_keys = ["QDRANT_URL", "QDRANT_API_KEY", "QDRANT_HOST", "QDRANT_PORT", "QDRANT_TIMEOUT_SECONDS"]
    original_env = {}
    for key in env_keys:
        if key in os.environ:
            original_env[key] = os.environ.pop(key)
    yield
    # Restore environment
    for key in env_keys:
        if key in original_env:
            os.environ[key] = original_env[key]
        elif key in os.environ:
            del os.environ[key]


def test_qdrant_settings_local_default(clean_env):
    settings = get_qdrant_settings()
    assert settings.url is None
    assert settings.api_key is None
    assert settings.host == "localhost"
    assert settings.port == 6333
    assert settings.timeout is None
    assert not settings.is_cloud


def test_qdrant_settings_cloud_precedence(clean_env):
    os.environ["QDRANT_URL"] = "https://cluster.qdrant.io:6333"
    os.environ["QDRANT_API_KEY"] = "secret-key"
    os.environ["QDRANT_HOST"] = "192.168.1.5"  # Should be ignored
    os.environ["QDRANT_PORT"] = "1234"         # Should be ignored

    settings = get_qdrant_settings()
    assert settings.url == "https://cluster.qdrant.io:6333"
    assert settings.api_key == "secret-key"
    assert settings.is_cloud


def test_qdrant_settings_custom_local(clean_env):
    os.environ["QDRANT_HOST"] = "qdrant-db"
    os.environ["QDRANT_PORT"] = "6334"
    os.environ["QDRANT_TIMEOUT_SECONDS"] = "20.5"

    settings = get_qdrant_settings()
    assert settings.url is None
    assert settings.host == "qdrant-db"
    assert settings.port == 6334
    assert settings.timeout == 20.5
    assert not settings.is_cloud


@patch("retrieval.support.qdrant_config.QdrantClient")
def test_create_qdrant_client_local(mock_qdrant_client, clean_env):
    # Ensure no URL is set
    client = create_qdrant_client(check_compatibility=False)

    mock_qdrant_client.assert_called_once_with(
        host="localhost",
        port=6333,
        check_compatibility=False,
    )


@patch("retrieval.support.qdrant_config.QdrantClient")
def test_create_qdrant_client_cloud(mock_qdrant_client, clean_env):
    os.environ["QDRANT_URL"] = "https://test.qdrant.cloud:6333"
    os.environ["QDRANT_API_KEY"] = "my-api-key"
    
    # Optional kwarg override
    client = create_qdrant_client(timeout=15.0)

    mock_qdrant_client.assert_called_once_with(
        url="https://test.qdrant.cloud:6333",
        timeout=15.0,  # Overridden by kwargs
        api_key="my-api-key",
    )
