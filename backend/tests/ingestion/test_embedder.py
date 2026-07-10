"""Unit tests for the embedding input formatting."""

from __future__ import annotations

import pytest
from rag_ingestion.models.chunk import Chunk
from rag_ingestion.stages.embedder import _embedding_input


def test_basic_fields():
    chunk = Chunk(
        relative_path="src/main.py",
        language="python",
        chunk_type="function",
        symbol_name="hello_world",
        summary="Simple hello world function",
        content="print('Hello World')",
    )
    result = _embedding_input(chunk)
    
    assert "File: src/main.py" in result
    assert "Language: python" in result
    assert "Type: function" in result
    assert "Symbol: hello_world" in result
    assert "Summary: Simple hello world function" in result
    assert "Code:" in result
    assert "print('Hello World')" in result
    # Assert other fields are omitted (avoid empty lines)
    assert "Dependencies:" not in result
    assert "Frameworks:" not in result


def test_description_and_facts():
    chunk = Chunk(
        relative_path="src/main.py",
        language="python",
        chunk_type="function",
        summary="A simple hello world",
        description="This function prints greeting messages to stdout.",
        summary_facts=["Prints 'Hello World'", "Uses standard print"],
        content="print('Hello World')",
    )
    result = _embedding_input(chunk)
    assert "Description: This function prints greeting messages to stdout." in result
    assert "Facts: Prints 'Hello World', Uses standard print" in result


def test_repo_and_config_metadata():
    chunk = Chunk(
        relative_path="package.json",
        language="json",
        chunk_type="file",
        file_type="package_json",
        detected_frameworks=["React", "Next.js"],
        dependencies=["next", "react"],
        dev_dependencies=["eslint", "typescript"],
        scripts={"dev": "next dev", "build": "next build"},
        services=["web"],
        ports=["3000"],
        env_keys=["PORT", "NODE_ENV"],
        entrypoints=["next start"],
        config_tools=["eslint", "typescript"],
        package_manager="pnpm",
        purpose="A next.js portfolio application",
        setup_steps=["pnpm install"],
        usage_commands=["pnpm dev"],
        architecture_notes=["Frontend is Next.js; hosted on Vercel"],
        content='{"name": "test-app"}',
    )
    result = _embedding_input(chunk)
    
    assert "File Type: package_json" in result
    assert "Frameworks: React, Next.js" in result
    assert "Dependencies: next, react" in result
    assert "Dev Dependencies: eslint, typescript" in result
    assert "Scripts: dev=next dev; build=next build" in result
    assert "Services: web" in result
    assert "Ports: 3000" in result
    assert "Environment Keys: PORT, NODE_ENV" in result
    assert "Entrypoints: next start" in result
    assert "Config Tools: eslint, typescript" in result
    assert "Package Manager: pnpm" in result
    assert "Purpose: A next.js portfolio application" in result
    assert "Setup Steps: pnpm install" in result
    assert "Usage Commands: pnpm dev" in result
    assert "Architecture Notes: Frontend is Next.js; hosted on Vercel" in result


def test_skips_empty_lines():
    chunk = Chunk(
        relative_path="src/utils.py",
        language="python",
        chunk_type="function",
        content="def add(a, b): return a + b",
    )
    result = _embedding_input(chunk)
    lines = result.splitlines()
    for line in lines:
        assert not line.endswith(": ")
        assert not line.endswith(":") or line == "Code:"
        
    assert "Dependencies" not in result
    assert "Services" not in result
    assert "Environment Keys" not in result


def test_code_truncation(monkeypatch):
    monkeypatch.setattr("rag_ingestion.stages.embedder.EMBEDDING_INPUT_MAX_CODE_CHARS", 10)
    chunk = Chunk(
        relative_path="src/main.py",
        language="python",
        chunk_type="file",
        content="1234567890abcdef",
    )
    result = _embedding_input(chunk)
    assert "Code:" in result
    assert "1234567890... [truncated]" in result


def test_total_input_truncation(monkeypatch):
    monkeypatch.setattr("rag_ingestion.stages.embedder.EMBEDDING_INPUT_MAX_TOTAL_CHARS", 30)
    chunk = Chunk(
        relative_path="src/main.py",
        language="python",
        chunk_type="file",
        content="abc",
    )
    result = _embedding_input(chunk)
    assert len(result) <= 45  # 30 + length of suffix "... [truncated]"
    assert result.endswith("... [truncated]")


def test_dict_fields_formatting():
    chunk = Chunk(
        relative_path="docker-compose.yml",
        language="yaml",
        chunk_type="file",
        scripts={"start": "docker compose up"},
        service_dependencies={"api": ["db", "redis"], "web": ["api"]},
        content="services: ...",
    )
    result = _embedding_input(chunk)
    assert "Scripts: start=docker compose up" in result
    assert "Service Dependencies: api depends on db, redis; web depends on api" in result


def test_no_raw_bracket_strings():
    chunk = Chunk(
        relative_path="src/main.py",
        language="python",
        chunk_type="file",
        detected_frameworks=[],
        dependencies=[],
        scripts={},
        service_dependencies={},
        content="print(1)",
    )
    result = _embedding_input(chunk)
    assert "[]" not in result
    assert "{}" not in result
    assert "Frameworks:" not in result
    assert "Dependencies:" not in result
    assert "Scripts:" not in result
    assert "Service Dependencies:" not in result


def test_embed_chunks_batching_and_cleanup():
    from unittest.mock import patch, MagicMock
    from rag_ingestion.stages.embedder import embed_chunks
    from rag_ingestion.utils.counters import PipelineCounters
    
    chunks = [
        Chunk(relative_path="file1.py", content="def a(): pass"),
        Chunk(relative_path="file2.py", content="def b(): pass"),
        Chunk(relative_path="file3.py", content="def c(): pass"),
    ]
    counters = PipelineCounters()
    
    mock_provider = MagicMock()
    
    with patch(
        "rag_ingestion.stages.embedder._get_provider",
        return_value=(MagicMock(provider="local", effective_model="BAAI/bge-small-en-v1.5", dimensions=0), mock_provider),
    ), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_BATCH_SIZE", 2), \
         patch("rag_ingestion.utils.gpu_cleanup.cleanup_after_batch") as mock_cleanup:

        mock_provider.embed_texts.side_effect = [
            [[0.1] * 384, [0.2] * 384],
            [[0.3] * 384],
        ]
        
        result = embed_chunks(chunks, counters)
        
        assert len(result) == 3
        assert result[0].embedding == [0.1]*384
        assert result[1].embedding == [0.2]*384
        assert result[2].embedding == [0.3]*384
        assert counters.embeddings_generated == 3
        
        # mock_cleanup should be called twice (once after each batch)
        assert mock_cleanup.call_count == 2


def test_embed_chunks_fails_cleanly_on_exception():
    from unittest.mock import patch, MagicMock
    from rag_ingestion.stages.embedder import embed_chunks
    from rag_ingestion.utils.counters import PipelineCounters
    
    chunks = [Chunk(relative_path="file1.py", content="def a(): pass")]
    counters = PipelineCounters()
    
    mock_provider = MagicMock()
    mock_provider.embed_texts.side_effect = RuntimeError("CUDA out of memory")
    
    with patch(
        "rag_ingestion.stages.embedder._get_provider",
        return_value=(MagicMock(provider="local", effective_model="BAAI/bge-small-en-v1.5", dimensions=0), mock_provider),
    ), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_BATCH_SIZE", 16):
         
        with pytest.raises(RuntimeError) as exc_info:
            embed_chunks(chunks, counters)
        
        assert "CUDA out of memory" in str(exc_info.value)


def test_embedding_cooldown_triggers_after_300(capsys):
    from unittest.mock import patch, MagicMock
    from rag_ingestion.stages.embedder import embed_chunks
    from rag_ingestion.utils.counters import PipelineCounters
    import numpy as np

    chunks = [Chunk(relative_path=f"file{i}.py", content="pass") for i in range(400)]
    counters = PipelineCounters()

    mock_provider = MagicMock()
    mock_provider.embed_texts.side_effect = lambda inputs, **kwargs: np.zeros((len(inputs), 384)).tolist()

    sleep_calls = []
    def fake_sleep(secs):
        sleep_calls.append(secs)

    with patch(
        "rag_ingestion.stages.embedder._get_provider",
        return_value=(MagicMock(provider="local", effective_model="BAAI/bge-small-en-v1.5", dimensions=0), mock_provider),
    ), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_BATCH_SIZE", 100), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_EVERY", 300), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_SECONDS", 30), \
         patch("rag_ingestion.stages.embedder._sleep", side_effect=fake_sleep):

        embed_chunks(chunks, counters)

    assert sleep_calls == [30]
    captured = capsys.readouterr()
    assert "[embedding.cooldown] embedded=300 remaining=100 sleeping=30s" in captured.out


def test_embedding_cooldown_explicit_default_sleep_duration(capsys):
    from unittest.mock import patch, MagicMock
    from rag_ingestion.stages.embedder import embed_chunks
    from rag_ingestion.utils.counters import PipelineCounters
    import numpy as np

    chunks = [Chunk(relative_path=f"file{i}.py", content="pass") for i in range(350)]
    counters = PipelineCounters()

    mock_provider = MagicMock()
    mock_provider.embed_texts.side_effect = lambda inputs, **kwargs: np.zeros((len(inputs), 384)).tolist()

    sleep_calls = []
    def fake_sleep(secs):
        sleep_calls.append(secs)

    with patch(
        "rag_ingestion.stages.embedder._get_provider",
        return_value=(MagicMock(provider="local", effective_model="BAAI/bge-small-en-v1.5", dimensions=0), mock_provider),
    ), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_BATCH_SIZE", 300), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_EVERY", 300), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_SECONDS", 30), \
         patch("rag_ingestion.stages.embedder._sleep", side_effect=fake_sleep):

        embed_chunks(chunks, counters)

    assert sleep_calls == [30]
    captured = capsys.readouterr()
    assert "sleeping=30s" in captured.out


def test_embedding_cooldown_disabled_with_every_zero(capsys):
    from unittest.mock import patch, MagicMock
    from rag_ingestion.stages.embedder import embed_chunks
    from rag_ingestion.utils.counters import PipelineCounters
    import numpy as np

    chunks = [Chunk(relative_path=f"file{i}.py", content="pass") for i in range(400)]
    counters = PipelineCounters()

    mock_provider = MagicMock()
    mock_provider.embed_texts.side_effect = lambda inputs, **kwargs: np.zeros((len(inputs), 384)).tolist()

    sleep_calls = []
    def fake_sleep(secs):
        sleep_calls.append(secs)

    with patch(
        "rag_ingestion.stages.embedder._get_provider",
        return_value=(MagicMock(provider="local", effective_model="BAAI/bge-small-en-v1.5", dimensions=0), mock_provider),
    ), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_BATCH_SIZE", 100), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_EVERY", 0), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_SECONDS", 30), \
         patch("rag_ingestion.stages.embedder._sleep", side_effect=fake_sleep):

        embed_chunks(chunks, counters)

    assert len(sleep_calls) == 0
    captured = capsys.readouterr()
    assert "[embedding.cooldown]" not in captured.out


def test_embedding_cooldown_disabled_with_seconds_zero(capsys):
    from unittest.mock import patch, MagicMock
    from rag_ingestion.stages.embedder import embed_chunks
    from rag_ingestion.utils.counters import PipelineCounters
    import numpy as np

    chunks = [Chunk(relative_path=f"file{i}.py", content="pass") for i in range(400)]
    counters = PipelineCounters()

    mock_provider = MagicMock()
    mock_provider.embed_texts.side_effect = lambda inputs, **kwargs: np.zeros((len(inputs), 384)).tolist()

    sleep_calls = []
    def fake_sleep(secs):
        sleep_calls.append(secs)

    with patch(
        "rag_ingestion.stages.embedder._get_provider",
        return_value=(MagicMock(provider="local", effective_model="BAAI/bge-small-en-v1.5", dimensions=0), mock_provider),
    ), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_BATCH_SIZE", 100), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_EVERY", 300), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_SECONDS", 0), \
         patch("rag_ingestion.stages.embedder._sleep", side_effect=fake_sleep):

        embed_chunks(chunks, counters)

    assert len(sleep_calls) == 0
    captured = capsys.readouterr()
    assert "[embedding.cooldown]" not in captured.out


def test_embedding_cooldown_does_not_sleep_after_final_batch(capsys):
    from unittest.mock import patch, MagicMock
    from rag_ingestion.stages.embedder import embed_chunks
    from rag_ingestion.utils.counters import PipelineCounters
    import numpy as np

    chunks = [Chunk(relative_path=f"file{i}.py", content="pass") for i in range(300)]
    counters = PipelineCounters()

    mock_provider = MagicMock()
    mock_provider.embed_texts.side_effect = lambda inputs, **kwargs: np.zeros((len(inputs), 384)).tolist()

    sleep_calls = []
    def fake_sleep(secs):
        sleep_calls.append(secs)

    with patch(
        "rag_ingestion.stages.embedder._get_provider",
        return_value=(MagicMock(provider="local", effective_model="BAAI/bge-small-en-v1.5", dimensions=0), mock_provider),
    ), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_BATCH_SIZE", 100), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_EVERY", 300), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_SECONDS", 30), \
         patch("rag_ingestion.stages.embedder._sleep", side_effect=fake_sleep):

        embed_chunks(chunks, counters)

    assert len(sleep_calls) == 0
    captured = capsys.readouterr()
    assert "[embedding.cooldown]" not in captured.out


def test_embedding_cooldown_cleanup_before_sleep():
    from unittest.mock import patch, MagicMock
    from rag_ingestion.stages.embedder import embed_chunks
    from rag_ingestion.utils.counters import PipelineCounters
    import numpy as np

    chunks = [Chunk(relative_path=f"file{i}.py", content="pass") for i in range(400)]
    counters = PipelineCounters()

    mock_provider = MagicMock()
    mock_provider.embed_texts.side_effect = lambda inputs, **kwargs: np.zeros((len(inputs), 384)).tolist()

    actions = []

    def fake_cleanup():
        actions.append("cleanup")

    def fake_sleep(secs):
        actions.append("sleep")

    with patch(
        "rag_ingestion.stages.embedder._get_provider",
        return_value=(MagicMock(provider="local", effective_model="BAAI/bge-small-en-v1.5", dimensions=0), mock_provider),
    ), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_BATCH_SIZE", 300), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_EVERY", 300), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_SECONDS", 30), \
         patch("rag_ingestion.stages.embedder._sleep", side_effect=fake_sleep), \
         patch("rag_ingestion.utils.gpu_cleanup.cleanup_after_batch", side_effect=fake_cleanup):

        embed_chunks(chunks, counters)

    assert actions == ["cleanup", "cleanup", "sleep", "cleanup"]


def test_embedding_cooldown_log_message_contents(capsys):
    from unittest.mock import patch, MagicMock
    from rag_ingestion.stages.embedder import embed_chunks
    from rag_ingestion.utils.counters import PipelineCounters
    import numpy as np

    chunks = [Chunk(relative_path=f"file{i}.py", content="pass") for i in range(350)]
    counters = PipelineCounters()

    mock_provider = MagicMock()
    mock_provider.embed_texts.side_effect = lambda inputs, **kwargs: np.zeros((len(inputs), 384)).tolist()

    with patch(
        "rag_ingestion.stages.embedder._get_provider",
        return_value=(MagicMock(provider="local", effective_model="BAAI/bge-small-en-v1.5", dimensions=0), mock_provider),
    ), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_BATCH_SIZE", 300), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_EVERY", 300), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_SECONDS", 30), \
         patch("rag_ingestion.stages.embedder._sleep"):

        embed_chunks(chunks, counters)

    captured = capsys.readouterr()
    assert "[embedding.cooldown] embedded=300 remaining=50 sleeping=30s" in captured.out


def test_embedding_cooldown_variable_batch_sizes(capsys):
    from unittest.mock import patch, MagicMock
    from rag_ingestion.stages.embedder import embed_chunks
    from rag_ingestion.utils.counters import PipelineCounters
    import numpy as np

    chunks = [Chunk(relative_path=f"file{i}.py", content="pass") for i in range(1000)]
    counters = PipelineCounters()

    mock_provider = MagicMock()
    mock_provider.embed_texts.side_effect = lambda inputs, **kwargs: np.zeros((len(inputs), 384)).tolist()

    sleep_calls = []
    def fake_sleep(secs):
        sleep_calls.append(secs)

    with patch(
        "rag_ingestion.stages.embedder._get_provider",
        return_value=(MagicMock(provider="local", effective_model="BAAI/bge-small-en-v1.5", dimensions=0), mock_provider),
    ), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_BATCH_SIZE", 400), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_EVERY", 300), \
         patch("rag_ingestion.config.CODESEEK_EMBEDDING_COOLDOWN_SECONDS", 15), \
         patch("rag_ingestion.stages.embedder._sleep", side_effect=fake_sleep):

        embed_chunks(chunks, counters)

    assert sleep_calls == [15, 15]

    captured = capsys.readouterr()
    assert "[embedding.cooldown] embedded=400 remaining=600 sleeping=15s" in captured.out
    assert "[embedding.cooldown] embedded=800 remaining=200 sleeping=15s" in captured.out
