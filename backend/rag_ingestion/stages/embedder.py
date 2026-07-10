"""Embedding generation stage."""

from __future__ import annotations

import gc
import logging

from rag_ingestion.config import (
    EMBEDDING_INPUT_MAX_CODE_CHARS,
    EMBEDDING_INPUT_MAX_TOTAL_CHARS,
)
from rag_ingestion.models.chunk import Chunk
from rag_ingestion.utils.counters import PipelineCounters
from rag_ingestion.utils.gpu_cleanup import clear_python_cuda_cache
from retrieval.support.embedding_provider import (
    current_embedding_metadata,
    get_embedding_provider,
    get_embedding_provider_config,
    unload_local_embedding_model,
)

logger = logging.getLogger(__name__)


def _sleep(seconds: float) -> None:
    import time
    time.sleep(seconds)


def _get_provider():
    from retrieval.support.embedding_provider import resolve_embedding_config
    config = resolve_embedding_config()
    return config, get_embedding_provider(config)

KNOWN_LABELS = {
    "File",
    "Language",
    "Type",
    "File Type",
    "Symbol",
    "Qualified Symbol",
    "Parent Symbol",
    "Signature",
    "Labels",
    "Code Intent",
    "Summary",
    "Description",
    "Purpose",
    "Facts",
    "Frameworks",
    "Dependencies",
    "Dev Dependencies",
    "Scripts",
    "Services",
    "Ports",
    "Environment Keys",
    "Feature Flags",
    "Provider Keys",
    "Entrypoints",
    "Config Tools",
    "Build System",
    "Base Image",
    "Workdir",
    "Package Manager",
    "Volumes",
    "Service Dependencies",
    "Setup Steps",
    "Usage Commands",
    "Architecture Notes",
    "Parameters",
    "Methods",
    "File Symbols",
    "Docstring",
}


def embed_chunks(
    chunks: list[Chunk], counters: PipelineCounters, event_callback=None
) -> list[Chunk]:
    """Generate embeddings for chunks in batches."""
    from rag_ingestion.config import (
        CODESEEK_EMBEDDING_BATCH_SIZE,
        CODESEEK_EMBEDDING_COOLDOWN_EVERY,
        CODESEEK_EMBEDDING_COOLDOWN_SECONDS,
    )
    from rag_ingestion.utils.gpu_cleanup import cleanup_after_batch

    config, provider = _get_provider()

    logger.info(
        "[embedding] provider=%s model=%s dimensions=%s source=%s",
        config.provider,
        config.effective_model,
        config.dimensions if config.dimensions > 0 else "auto/infer",
        getattr(config, "source", "unknown"),
    )
    logger.info(
        "Embedding %d chunks — provider=%s model=%s batch_size=%d",
        len(chunks),
        config.provider,
        config.effective_model,
        CODESEEK_EMBEDDING_BATCH_SIZE,
    )

    batch_size = CODESEEK_EMBEDDING_BATCH_SIZE
    if batch_size < 1:
        batch_size = 1

    total_chunks = len(chunks)
    embedded_count = 0
    next_cooldown_at = CODESEEK_EMBEDDING_COOLDOWN_EVERY

    try:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            inputs = [_embedding_input(chunk) for chunk in batch]
            embeddings = provider.embed_texts(
                inputs,
                batch_size=batch_size,
                show_progress_bar=True,
            )
            for chunk, embedding in zip(batch, embeddings, strict=True):
                chunk.embedding = list(embedding)
                counters.embeddings_generated += 1
            
            
            embedded_count += len(batch)
            if event_callback:
                event_callback(
                    stage="embedding",
                    message=f"Embedded {embedded_count} of {total_chunks} chunks...",
                    progress=embedded_count,
                    total=total_chunks,
                )

            # Free temporary inputs/embeddings and collect Python memory
            del inputs
            del embeddings
            cleanup_after_batch()

            remaining = total_chunks - embedded_count
            if (
                CODESEEK_EMBEDDING_COOLDOWN_EVERY > 0
                and CODESEEK_EMBEDDING_COOLDOWN_SECONDS > 0
                and remaining > 0
                and embedded_count >= next_cooldown_at
            ):
                cleanup_after_batch()
                print(
                    f"[embedding.cooldown] embedded={embedded_count} remaining={remaining} sleeping={CODESEEK_EMBEDDING_COOLDOWN_SECONDS}s"
                )
                _sleep(CODESEEK_EMBEDDING_COOLDOWN_SECONDS)
                while next_cooldown_at <= embedded_count:
                    next_cooldown_at += CODESEEK_EMBEDDING_COOLDOWN_EVERY
    except Exception as exc:
        logger.error(
            "Embedding generation failed: %s. "
            "This may be caused by provider configuration issues, upstream request failures, "
            "CUDA OOM, or system RAM limits. Try reducing CODESEEK_EMBEDDING_BATCH_SIZE or "
            "switching CODESEEK_EMBEDDING_PROVIDER.",
            exc,
        )
        raise exc

    resolved_dimensions = 0
    for chunk in chunks:
        if chunk.embedding:
            resolved_dimensions = len(chunk.embedding)
            break
    if config.dimensions > 0 and resolved_dimensions > 0 and resolved_dimensions != config.dimensions:
        logger.warning(
            "Provider returned %d dimensions although config expected %d; using returned provider dimension %d for this index.",
            resolved_dimensions, config.dimensions, resolved_dimensions
        )
    elif config.dimensions <= 0 and resolved_dimensions > 0:
        logger.info("[embedding] resolved dimensions=%d", resolved_dimensions)
        
    setattr(
        counters,
        "embedding_provider_metadata",
        current_embedding_metadata(resolved_dimensions=resolved_dimensions),
    )

    return chunks


def unload_embedding_model() -> None:
    """Release the cached SentenceTransformer reference and free memory.

    After this call the model will be re-loaded on the next embed_chunks()
    invocation.  This is intentional: it allows the OS to reclaim any CUDA
    or CPU memory that was held by the model weights.
    """
    unload_local_embedding_model()
    gc.collect()
    clear_python_cuda_cache("after embedding model unload")
    logger.info("Embedding model reference released")


def _line(label: str, value: str | None) -> list[str]:
    if not value or not str(value).strip():
        return []
    return [f"{label}: {str(value).strip()}"]


def _list_line(label: str, values: list[str], limit: int = 20) -> list[str]:
    if not values:
        return []
    cleaned = [str(v).strip() for v in values if v and str(v).strip()]
    if not cleaned:
        return []
    return [f"{label}: {', '.join(cleaned[:limit])}"]


def _dict_line(label: str, values: dict, limit: int = 20) -> list[str]:
    if not values:
        return []
    parts = []
    for k, v in list(values.items())[:limit]:
        if not k or v is None:
            continue
        if isinstance(v, list):
            if not v:
                continue
            parts.append(f"{k} depends on {', '.join(str(item) for item in v if item)}")
        else:
            parts.append(f"{k}={v}")
    if not parts:
        return []
    return [f"{label}: {'; '.join(parts)}"]


def _embedding_input(chunk: Chunk) -> str:
    lines = []

    lines += _line("File", chunk.relative_path)
    lines += _line("Language", chunk.language)
    lines += _line("Type", chunk.chunk_type)
    lines += _line("File Type", chunk.file_type)
    lines += _line("Symbol", chunk.symbol_name)
    lines += _line("Qualified Symbol", chunk.qualified_symbol)
    lines += _line("Parent Symbol", chunk.parent_symbol)
    lines += _line("Signature", chunk.signature)
    lines += _list_line("Labels", getattr(chunk, "labels", []))
    lines += _line("Code Intent", getattr(chunk, "code_intent", ""))
    lines += _line("Summary", chunk.summary)
    lines += _line("Description", chunk.description)
    lines += _line("Purpose", chunk.purpose)
    lines += _list_line("Facts", chunk.summary_facts)
    lines += _list_line("Frameworks", chunk.detected_frameworks)
    lines += _list_line("Dependencies", chunk.dependencies, limit=30)
    lines += _list_line("Dev Dependencies", chunk.dev_dependencies)
    lines += _dict_line("Scripts", chunk.scripts)
    lines += _list_line("Services", chunk.services)
    lines += _list_line("Ports", chunk.ports)
    lines += _list_line("Environment Keys", chunk.env_keys, limit=30)
    lines += _list_line("Feature Flags", chunk.feature_flags)
    lines += _list_line("Provider Keys", chunk.provider_keys)
    lines += _list_line("Entrypoints", chunk.entrypoints)
    lines += _list_line("Config Tools", chunk.config_tools)
    lines += _line("Build System", chunk.build_system)
    lines += _line("Base Image", chunk.base_image)
    lines += _line("Workdir", chunk.workdir)
    lines += _line("Package Manager", chunk.package_manager)
    lines += _list_line("Volumes", chunk.volumes)
    lines += _dict_line("Service Dependencies", chunk.service_dependencies)
    lines += _list_line("Setup Steps", chunk.setup_steps)
    lines += _list_line("Usage Commands", chunk.usage_commands)
    lines += _list_line("Architecture Notes", chunk.architecture_notes)
    lines += _list_line("Parameters", chunk.parameters)
    lines += _list_line("Methods", chunk.methods)
    lines += _list_line("File Symbols", chunk.file_symbols)
    lines += _line("Docstring", chunk.docstring)

    code = chunk.content or ""
    if code.strip():
        if len(code) > EMBEDDING_INPUT_MAX_CODE_CHARS:
            code = code[:EMBEDDING_INPUT_MAX_CODE_CHARS] + "... [truncated]"
        lines.append("Code:")
        lines.append(code)

    final_input = "\n".join(lines)
    if len(final_input) > EMBEDDING_INPUT_MAX_TOTAL_CHARS:
        final_input = final_input[:EMBEDDING_INPUT_MAX_TOTAL_CHARS] + "... [truncated]"

    return final_input
