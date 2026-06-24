from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from retrieval.support.embedding_provider import (
    EmbeddingConfigurationError,
    EmbeddingProviderConfig,
    EmbeddingRequestError,
    OpenAICompatibleEmbeddingProvider,
    build_embeddings_endpoint,
)


def _cloud_config(**overrides) -> EmbeddingProviderConfig:
    base = EmbeddingProviderConfig(
        provider="openai_compatible",
        base_url="https://api.example.com/v1/",
        api_key="super-secret-key",
        model="openai/text-embedding-3-small",
        batch_size=2,
        timeout_seconds=30.0,
        dimensions=0,
        local_model="BAAI/bge-small-en-v1.5",
        local_device="cpu",
    )
    return base.__class__(**{**base.__dict__, **overrides})


def test_base_url_builds_embeddings_endpoint():
    assert build_embeddings_endpoint("https://api.example.com/v1/") == "https://api.example.com/v1/embeddings"
    assert build_embeddings_endpoint("https://api.example.com/custom") == "https://api.example.com/custom/embeddings"


def test_cloud_embedding_request_shape(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4]},
                ]
            },
        )

    monkeypatch.setattr("retrieval.support.embedding_provider.httpx.post", fake_post)

    provider = OpenAICompatibleEmbeddingProvider(_cloud_config())
    vectors = provider.embed_texts(["text 1", "text 2"])

    assert captured["url"] == "https://api.example.com/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer super-secret-key"
    assert captured["json"] == {
        "model": "openai/text-embedding-3-small",
        "input": ["text 1", "text 2"],
    }
    assert captured["timeout"] == 30.0
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_cloud_embedding_response_parsing_preserves_order(monkeypatch: pytest.MonkeyPatch):
    def fake_post(url, *, headers, json, timeout):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
        )

    monkeypatch.setattr("retrieval.support.embedding_provider.httpx.post", fake_post)
    provider = OpenAICompatibleEmbeddingProvider(_cloud_config())

    vectors = provider.embed_texts(["first", "second"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert provider.dimensions == 2


def test_cloud_embedding_response_count_mismatch_raises(monkeypatch: pytest.MonkeyPatch):
    def fake_post(url, *, headers, json, timeout):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
        )

    monkeypatch.setattr("retrieval.support.embedding_provider.httpx.post", fake_post)
    provider = OpenAICompatibleEmbeddingProvider(_cloud_config())

    with pytest.raises(EmbeddingRequestError) as exc_info:
        provider.embed_texts(["first", "second"])

    assert "returned 1 vectors for 2 inputs" in str(exc_info.value)


def test_api_key_is_not_included_in_error_text(monkeypatch: pytest.MonkeyPatch):
    def fake_post(url, *, headers, json, timeout):
        raise RuntimeError("upstream failure for super-secret-key")

    monkeypatch.setattr("retrieval.support.embedding_provider.httpx.post", fake_post)
    provider = OpenAICompatibleEmbeddingProvider(_cloud_config())

    with pytest.raises(EmbeddingRequestError) as exc_info:
        provider.embed_texts(["first"])

    assert "super-secret-key" not in str(exc_info.value)
    assert "*****" in str(exc_info.value)


def test_cloud_embedding_batches_inputs(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_post(url, *, headers, json, timeout):
        calls.append(list(json["input"]))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "data": [
                    {"index": idx, "embedding": [float(idx), float(idx + 1)]}
                    for idx, _ in enumerate(json["input"])
                ]
            },
        )

    monkeypatch.setattr("retrieval.support.embedding_provider.httpx.post", fake_post)
    provider = OpenAICompatibleEmbeddingProvider(_cloud_config(batch_size=2))

    vectors = provider.embed_texts(["a", "b", "c"])

    assert calls == [["a", "b"], ["c"]]
    assert vectors == [[0.0, 1.0], [1.0, 2.0], [0.0, 1.0]]


def test_http_status_error_extracts_and_sanitizes_body(monkeypatch: pytest.MonkeyPatch):
    class FakeResponse:
        status_code = 400
        text = '{"error": {"message": "Invalid model text-embedding for api key super-secret-key"}}'
        
        def json(self):
            import json
            return json.loads(self.text)
            
    class FakeException(httpx.HTTPStatusError):
        def __init__(self):
            self.response = FakeResponse()
            super().__init__("error", request=None, response=self.response)

    def fake_post(url, *, headers, json, timeout):
        raise FakeException()

    monkeypatch.setattr("retrieval.support.embedding_provider.httpx.post", fake_post)
    provider = OpenAICompatibleEmbeddingProvider(_cloud_config())

    with pytest.raises(EmbeddingRequestError) as exc_info:
        provider.embed_texts(["first"])

    assert "super-secret-key" not in str(exc_info.value)
    assert "*****" in str(exc_info.value)
    assert "Invalid model" in str(exc_info.value)
    assert "status 400" in str(exc_info.value)
