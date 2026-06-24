"""Entry point for the RAG ingestion pipeline."""

import argparse
import logging

logger = logging.getLogger(__name__)


from rag_ingestion.config import (
    COLLECTION_NAME,
    ENABLE_GPU_CLEANUP_AFTER_STAGES,
    ENABLE_INCREMENTAL_FILE_SKIP,
    LOCAL_LLM_UNLOAD_MODEL,
    RECREATE_COLLECTION_EACH_RUN,
    UNLOAD_EMBEDDING_MODEL_AFTER_INDEXING,
    UNLOAD_LOCAL_LLM_AFTER_DESCRIPTIONS,
    UNLOAD_LOCAL_LLM_AFTER_INDEXING,
)
from rag_ingestion.utils.gpu_cleanup import (
    clear_python_cuda_cache,
    log_gpu_memory_snapshot,
    unload_ollama_model,
    cleanup_after_batch,
)
from retrieval.support.isolation import expected_collection_name, validate_collection_binding
from rag_ingestion.stages.chunker import generate_chunks
from rag_ingestion.stages.discovery import discover_files
from rag_ingestion.stages.embedder import embed_chunks, unload_embedding_model
from rag_ingestion.stages.filtering import filter_files
from rag_ingestion.stages.language import detect_languages
from rag_ingestion.stages.loader import load_repository
from rag_ingestion.stages.metadata import build_metadata
from rag_ingestion.stages.overflow import handle_overflow
from rag_ingestion.stages.parser import parse_file
from rag_ingestion.stages.repo_summary import (
    build_repo_summary_chunk,
    is_repo_summary_evidence_path,
)
from rag_ingestion.stages.storage import delete_chunks_for_paths, store_chunks
from rag_ingestion.stages.summary import generate_summary
from rag_ingestion.utils.counters import PipelineCounters
from rag_ingestion.utils.logger import log_skip, skipped_files
from rag_ingestion.utils.state import (
    build_file_signature,
    is_file_unchanged,
    load_ingestion_state,
    save_ingestion_state,
)


def main() -> None:
    """Parse CLI arguments and run the ingestion pipeline."""
    args = _parse_args()
    run_pipeline(args.source, collection_name=args.collection)


def run_pipeline(
    source: str,
    collection_name: str | None = None,
    enable_chunk_descriptions: bool | None = None,
    provider_config: dict | None = None,
    event_callback=None,
    recreate_collection: bool | None = None,
    session_id: str | None = None,
    commit_sha: str | None = None,
    branch_name: str | None = None,
) -> PipelineCounters:
    """Run all ingestion stages in order."""
    from rag_ingestion.config import (
        CODESEEK_DESCRIPTION_MODEL,
        CODESEEK_LABEL_MODEL,
        CODESEEK_DESCRIPTION_BATCH_SIZE,
        CODESEEK_EMBEDDING_BATCH_SIZE,
        CODESEEK_CHUNK_PROCESS_BATCH_SIZE,
        CODESEEK_DESCRIPTION_MAX_CHARS,
        CODESEEK_DESCRIPTION_MAX_TOKENS,
        CODESEEK_OLLAMA_KEEP_ALIVE,
        CODESEEK_OLLAMA_STOP_MODEL_EVERY,
    )

    logger.info("[ingestion.config] description_model=%s", CODESEEK_DESCRIPTION_MODEL)
    logger.info("[ingestion.config] label_model=%s", CODESEEK_LABEL_MODEL)
    logger.info("[ingestion.config] description_batch_size=%d", CODESEEK_DESCRIPTION_BATCH_SIZE)
    logger.info("[ingestion.config] embedding_batch_size=%d", CODESEEK_EMBEDDING_BATCH_SIZE)
    logger.info("[ingestion.config] chunk_process_batch_size=%d", CODESEEK_CHUNK_PROCESS_BATCH_SIZE)
    logger.info("[ingestion.config] description_max_chars=%d", CODESEEK_DESCRIPTION_MAX_CHARS)
    logger.info("[ingestion.config] description_max_tokens=%d", CODESEEK_DESCRIPTION_MAX_TOKENS)
    logger.info("[ingestion.config] ollama_keep_alive=%s", CODESEEK_OLLAMA_KEEP_ALIVE)
    logger.info("[ingestion.config] ollama_stop_model_every=%d", CODESEEK_OLLAMA_STOP_MODEL_EVERY)

    should_recreate_collection = (
        RECREATE_COLLECTION_EACH_RUN
        if recreate_collection is None
        else recreate_collection
    )

    counters = PipelineCounters()

    def emit(stage, message, level="info", progress=None, total=None, metadata=None):
        if event_callback:
            event_callback(
                stage=stage, message=message, level=level,
                progress=progress, total=total, metadata=metadata,
            )

    repository = load_repository(source)
    selected_collection = collection_name or expected_collection_name(
        repository["repository_root"]
    )
    validate_collection_binding(selected_collection, repository["repository_root"])

    # --- Discovery ---
    discovered_files = discover_files(repository["repository_root"], counters)
    emit("discovery", f"Discovered {counters.files_discovered} files in the repository.",
         progress=counters.files_discovered)

    # --- Filtering ---
    filtered_files = filter_files(
        discovered_files,
        repository["repository_root"],
        counters,
    )
    emit("filtering", f"Filtered repository — ignored {counters.files_ignored} generated or noise files.",
         progress=len(filtered_files))

    # --- Language detection ---
    language_files = detect_languages(filtered_files, counters)
    processable = [f for f in language_files if not f.skipped]
    emit("language",
         f"Detected supported languages for {len(processable)} files. "
         f"{counters.files_skipped_unsupported} files skipped as unsupported.",
         progress=len(processable))

    # --- Incremental state ---
    previous_state: dict[str, dict[str, int]] = {}
    next_state: dict[str, dict[str, int]] = {}
    modified_paths: list[str] = []  # files that were re-parsed (changed) in incremental mode
    use_incremental_skip = ENABLE_INCREMENTAL_FILE_SKIP and not should_recreate_collection
    if use_incremental_skip:
        previous_state = load_ingestion_state(repository["repository_root"])

    # --- Parse + chunk ---
    all_chunks = []
    parsed_count = 0

    batch_size = CODESEEK_CHUNK_PROCESS_BATCH_SIZE
    if batch_size < 1:
        batch_size = 1

    file_batches = [processable[i : i + batch_size] for i in range(0, len(processable), batch_size)]

    for file_batch in file_batches:
        batch_chunks = []
        for file in file_batch:
            signature = build_file_signature(file)
            file_was_unchanged = use_incremental_skip and is_file_unchanged(
                file.relative_path, signature, previous_state
            )
            if file_was_unchanged:
                next_state[file.relative_path] = signature
                if not is_repo_summary_evidence_path(file.relative_path):
                    log_skip(file.relative_path, "unchanged_file", "skipped")
                    continue
                log_skip(file.relative_path, "repo_summary_evidence_refresh", "parsed")

            # Only track as modified if the file actually changed (not a forced evidence refresh)
            if (use_incremental_skip and not should_recreate_collection
                    and not file_was_unchanged
                    and file.relative_path in previous_state):
                modified_paths.append(file.relative_path)

            parsed = parse_file(file, counters)
            chunks = generate_chunks(parsed, file)
            chunks = handle_overflow(chunks)

            for chunk in chunks:
                build_metadata(chunk)
                chunk.summary = generate_summary(chunk)

            # Copy file_type to all chunks of the same file
            file_type = next((c.file_type for c in chunks if c.file_type), "")
            if file_type:
                for chunk in chunks:
                    chunk.file_type = file_type

            counters.chunks_generated += len(chunks)
            batch_chunks.extend(chunks)
            if ENABLE_INCREMENTAL_FILE_SKIP:
                next_state[file.relative_path] = signature

            parsed_count += 1
            if parsed_count % 10 == 0:
                emit("parser", f"Parsed {parsed_count} files so far…",
                     progress=parsed_count, total=len(processable))

        all_chunks.extend(batch_chunks)
        del batch_chunks
        cleanup_after_batch()

    emit("parser", f"Parsed {counters.files_parsed_ok} files successfully.",
         progress=counters.files_parsed_ok, total=len(processable), level="success")

    # --- Repo summary ---
    repo_summary = build_repo_summary_chunk(all_chunks, repository)
    if repo_summary is not None:
        build_metadata(repo_summary)
        all_chunks.append(repo_summary)
        counters.chunks_generated += 1

    emit("chunker", f"Generated {counters.chunks_generated} searchable chunks.",
         progress=counters.chunks_generated, total=counters.chunks_generated)

    if all_chunks:
        # --- Descriptions ---
        from rag_ingestion.stages.description import describe_chunks
        all_chunks = describe_chunks(
            all_chunks,
            enabled=enable_chunk_descriptions,
            provider_config=provider_config,
            event_callback=event_callback,
        )

        # --- GPU cleanup after description generation ---
        if ENABLE_GPU_CLEANUP_AFTER_STAGES:
            clear_python_cuda_cache("after chunk description generation")
            log_gpu_memory_snapshot("after chunk description generation")

        # --- Optional: unload Ollama model after descriptions (before embedding) ---
        if UNLOAD_LOCAL_LLM_AFTER_DESCRIPTIONS and LOCAL_LLM_UNLOAD_MODEL:
            unload_ollama_model(LOCAL_LLM_UNLOAD_MODEL)
            clear_python_cuda_cache("after ollama unload post-descriptions")
            log_gpu_memory_snapshot("after ollama unload post-descriptions")

        # --- Labeling ---
        from rag_ingestion.config import ENABLE_CHUNK_LABELS
        if ENABLE_CHUNK_LABELS:
            from rag_ingestion.stages.labeler import label_chunks
            repo_name = repository.get("repository_name", "")
            repo_root = repository.get("repository_root", "")
            all_chunks = label_chunks(all_chunks, repo_name=repo_name, repo_root=repo_root)

            labeled_count = sum(1 for c in all_chunks if getattr(c, "labels", None))
            logger.info(
                "Labeled %s/%s chunks before embedding",
                labeled_count,
                len(all_chunks),
            )
            for chunk in all_chunks[:5]:
                logger.debug(
                    "Labeled chunk sample: path=%s type=%s labels=%s code_intent=%s",
                    chunk.relative_path,
                    chunk.chunk_type,
                    chunk.labels,
                    chunk.code_intent,
                )

        # --- Embedding ---
        emit("embedding", f"Embedding {len(all_chunks)} chunks…")
        embedded_chunks = embed_chunks(all_chunks, counters, event_callback=emit)
        embedding_metadata = getattr(counters, "embedding_provider_metadata", {})
        embedding_dimensions = int(embedding_metadata.get("embedding_dimensions", 0) or 0)

        emit("embedding",
             f"Generated embeddings for {counters.embeddings_generated} chunks.",
             level="success", progress=counters.embeddings_generated,
             total=len(all_chunks))

        # --- GPU cleanup after embedding ---
        if ENABLE_GPU_CLEANUP_AFTER_STAGES:
            clear_python_cuda_cache("after embedding generation")
            log_gpu_memory_snapshot("after embedding generation")

        # --- Storage: delete stale chunks for modified files first ---
        if modified_paths:
            emit("storage", f"Deleting stale chunks for {len(modified_paths)} modified file(s)…")
            delete_chunks_for_paths(modified_paths, collection_name=selected_collection)

        emit("storage", f"Storing {len(embedded_chunks)} chunks in Qdrant…")
        store_chunks(
            embedded_chunks,
            counters,
            collection_name=selected_collection,
            recreate_collection=should_recreate_collection,
            embedding_dimensions=embedding_dimensions,
        )
        emit("storage",
             f"Stored {counters.embeddings_stored} chunks in Qdrant.",
             level="success", progress=counters.embeddings_stored,
             total=counters.embeddings_stored)

        # --- Passive Metadata Recording ---
        if session_id:
            try:
                import hashlib
                from retrieval.db import upsert_session_file, replace_session_file_chunks
                from datetime import datetime, timezone
                import os

                chunks_by_file = {}
                for chunk in embedded_chunks:
                    if not chunk.relative_path or chunk.chunk_type == "repo_summary":
                        continue
                    chunks_by_file.setdefault(chunk.relative_path, []).append(chunk)

                last_indexed_at = datetime.now(timezone.utc).isoformat()

                for rel_path, file_chunks in chunks_by_file.items():
                    first_chunk = file_chunks[0]
                    abs_path = first_chunk.file_path or os.path.join(repository["repository_root"], rel_path)

                    file_hash = ""
                    try:
                        if os.path.isfile(abs_path):
                            with open(abs_path, "rb") as f:
                                file_hash = hashlib.sha256(f.read()).hexdigest()
                        else:
                            file_hash = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()
                    except Exception:
                        file_hash = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()

                    file_record = upsert_session_file(
                        session_id=session_id,
                        repo_path=rel_path,
                        file_hash=file_hash,
                        indexed_commit_sha=commit_sha or "",
                        indexed_branch=branch_name or "",
                        status="indexed",
                        last_indexed_at=last_indexed_at,
                        deleted_at=None,
                    )

                    chunk_mappings = []
                    for c in file_chunks:
                        chunk_mappings.append({
                            "chunk_id": c.chunk_id,
                            "vector_id": c.chunk_id,
                            "symbol": c.symbol_name or None,
                            "start_line": c.start_line if c.start_line > 0 else None,
                            "end_line": c.end_line if c.end_line > 0 else None,
                        })

                    replace_session_file_chunks(file_record["id"], chunk_mappings)

            except Exception as e:
                logger.error("Failed to record passive incremental metadata: %s", e)
                raise RuntimeError(f"Metadata recording failed: {e}") from e

        # --- Final cleanup: unload models and free VRAM ---
        if UNLOAD_EMBEDDING_MODEL_AFTER_INDEXING:
            unload_embedding_model()

        if UNLOAD_LOCAL_LLM_AFTER_INDEXING and LOCAL_LLM_UNLOAD_MODEL:
            unload_ollama_model(LOCAL_LLM_UNLOAD_MODEL)

        if ENABLE_GPU_CLEANUP_AFTER_STAGES:
            clear_python_cuda_cache("after indexing complete")
            log_gpu_memory_snapshot("after indexing complete")

    if ENABLE_INCREMENTAL_FILE_SKIP:
        if not should_recreate_collection:
            removed_paths = sorted(set(previous_state) - set(next_state))
            if removed_paths:
                emit("storage", f"Deleting chunks for {len(removed_paths)} removed file(s)…")
                delete_chunks_for_paths(removed_paths, collection_name=selected_collection)
                emit("storage",
                     f"Deleted chunks for {len(removed_paths)} removed file(s).",
                     level="success")
        save_ingestion_state(repository["repository_root"], next_state)

    _print_report(repository, counters, collection_name=selected_collection or COLLECTION_NAME)
    return counters


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a local or public GitHub repository into Qdrant."
    )
    parser.add_argument(
        "source",
        help="Absolute local repository path or public GitHub URL.",
    )
    parser.add_argument(
        "--collection",
        default="",
        help="Optional Qdrant collection override for this run.",
    )
    return parser.parse_args()


def _print_report(repository: dict, counters: PipelineCounters, collection_name: str) -> None:
    print("========================================")
    print("Ingestion Complete")
    print("========================================")
    print(f"Repository:             {repository['repository_name']}")
    print(f"Repository source:      {repository['source_type']}")
    print()
    print(f"Files discovered:       {counters.files_discovered}")
    print(f"Files ignored:          {counters.files_ignored}")
    print(f"Files skipped:          {counters.files_skipped_unsupported}")
    print(f"Files parsed OK:        {counters.files_parsed_ok}")
    print(f"Files parse failed:     {counters.files_parse_failed}")
    print()
    print(f"Chunks generated:       {counters.chunks_generated}")
    print(f"Embeddings stored:      {counters.embeddings_stored}")
    print()
    metadata = getattr(counters, "embedding_provider_metadata", {})
    if metadata:
        print(f"Embedding provider:     {metadata.get('embedding_provider', 'unknown')}")
        print(f"Embedding model:        {metadata.get('embedding_model', 'unknown')}")
        print(f"Embedding dimensions:   {metadata.get('embedding_dimensions', 'unknown')}")
        print(f"Embedding config:       {metadata.get('embedding_config_source', 'unknown')}")
        print()
    print(f"Collection:             {collection_name}")
    print("========================================")

    if skipped_files:
        print()
        print("Skipped files:")
        for item in skipped_files:
            print(f"- {item['file']} | {item['reason']} | {item['action']}")


def run_incremental_pipeline(
    source: str,
    collection_name: str | None = None,
    enable_chunk_descriptions: bool | None = None,
    provider_config: dict | None = None,
    event_callback=None,
    session_id: str | None = None,
    commit_sha: str | None = None,
    branch_name: str | None = None,
    added_files: list[str] | None = None,
    modified_files: list[str] | None = None,
    deleted_files: list[str] | None = None,
) -> PipelineCounters:
    """
    Executes a partial/incremental reindexing pipeline for the specified session
    and subset of files.
    """
    from rag_ingestion.utils.counters import PipelineCounters
    from rag_ingestion.stages.loader import load_repository
    from retrieval.support.isolation import expected_collection_name, validate_collection_binding
    from rag_ingestion.stages.discovery import discover_files
    from rag_ingestion.stages.filtering import filter_files
    from rag_ingestion.stages.language import detect_languages
    from rag_ingestion.stages.parser import parse_file
    from rag_ingestion.stages.chunker import generate_chunks
    from rag_ingestion.stages.overflow import handle_overflow
    from rag_ingestion.stages.metadata import build_metadata
    from rag_ingestion.stages.summary import generate_summary
    from rag_ingestion.stages.embedder import embed_chunks
    import os
    import hashlib
    from pathlib import Path
    from datetime import datetime, timezone
    from rag_ingestion.models.file import FileRecord
    import logging

    logger = logging.getLogger(__name__)
    counters = PipelineCounters()

    def emit(stage, message, level="info", progress=None, total=None, metadata=None):
        if event_callback:
            event_callback(
                stage=stage, message=message, level=level,
                progress=progress, total=total, metadata=metadata,
            )

    repository = load_repository(source)
    selected_collection = collection_name or expected_collection_name(
        repository["repository_root"]
    )
    validate_collection_binding(selected_collection, repository["repository_root"])

    targets = set(added_files or []) | set(modified_files or [])
    if not targets and not (deleted_files or []):
        logger.info("No files added, modified, or deleted. Incremental update is a no-op.")
        return counters

    # --- Construct FileRecords ---
    root = Path(repository["repository_root"]).resolve()
    processable = []
    for rel_path in targets:
        abs_path = root / rel_path
        if not abs_path.exists():
            continue
        stat = abs_path.stat()
        file_rec = FileRecord(
            path=str(abs_path.resolve()),
            relative_path=rel_path,
            extension=abs_path.suffix,
            size_bytes=stat.st_size,
        )
        processable.append(file_rec)
        counters.files_discovered += 1

    emit("discovery", f"Discovered {counters.files_discovered} files to update.", progress=counters.files_discovered)

    # --- Filtering ---
    filtered_files = filter_files(
        processable,
        repository["repository_root"],
        counters,
    )
    emit("filtering", f"Filtered target list — ignored {counters.files_ignored} files.", progress=len(filtered_files))

    # --- Language detection ---
    language_files = detect_languages(filtered_files, counters)
    target_processable = [f for f in language_files if not f.skipped]
    emit("language", f"Processing {len(target_processable)} files after language support checks.")

    # --- Parse + chunk ---
    all_chunks = []
    for file in target_processable:
        emit("parser", f"Processing file {file.relative_path}...", progress=counters.files_parsed_ok)
        try:
            parsed = parse_file(file, counters)
            chunks = generate_chunks(parsed, file)
            chunks = handle_overflow(chunks)

            for chunk in chunks:
                build_metadata(chunk)
                chunk.summary = generate_summary(chunk)

            # Copy file_type to all chunks of the same file
            file_type = next((c.file_type for c in chunks if c.file_type), "")
            if file_type:
                for chunk in chunks:
                    chunk.file_type = file_type

            counters.chunks_generated += len(chunks)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error("Failed to parse/chunk file %s: %s", file.relative_path, e)
            raise RuntimeError(f"Incremental indexing failed for file {file.relative_path}: {e}") from e

    emit("parser", f"Parsed {counters.files_parsed_ok} files successfully.", progress=counters.files_parsed_ok)

    embedded_chunks = []
    if all_chunks:
        # --- Descriptions ---
        if enable_chunk_descriptions:
            from rag_ingestion.stages.description import describe_chunks
            all_chunks = describe_chunks(
                all_chunks,
                enabled=enable_chunk_descriptions,
                provider_config=provider_config,
                event_callback=event_callback,
            )

        # --- Labeling ---
        from rag_ingestion.config import ENABLE_CHUNK_LABELS
        if ENABLE_CHUNK_LABELS:
            from rag_ingestion.stages.labeler import label_chunks
            repo_name = repository.get("repository_name", "")
            repo_root = repository.get("repository_root", "")
            all_chunks = label_chunks(all_chunks, repo_name=repo_name, repo_root=repo_root)

        # --- Embedding ---
        emit("embedding", f"Embedding {len(all_chunks)} chunks…")
        embedded_chunks = embed_chunks(all_chunks, counters, event_callback=emit)
        embedding_metadata = getattr(counters, "embedding_provider_metadata", {})
        embedding_dimensions = int(embedding_metadata.get("embedding_dimensions", 0) or 0)
        emit("embedding", f"Generated embeddings for {counters.embeddings_generated} chunks.")

    # --- Qdrant Deletion Pre-fetch ---
    old_vector_ids_to_delete = []
    if session_id:
        from retrieval.db import list_session_files
        db_files = list_session_files(session_id, include_deleted=True)
        db_files_by_path = {f["repo_path"]: f for f in db_files}

        for mod_file in (modified_files or []):
            if mod_file in db_files_by_path:
                for chunk in db_files_by_path[mod_file]["chunks"]:
                    if chunk.get("vector_id"):
                        old_vector_ids_to_delete.append(chunk["vector_id"])

        for del_file in (deleted_files or []):
            if del_file in db_files_by_path:
                for chunk in db_files_by_path[del_file]["chunks"]:
                    if chunk.get("vector_id"):
                        old_vector_ids_to_delete.append(chunk["vector_id"])

    # Cooperative cancel check before storage changes
    emit("storage", "Preparing storage modifications...", progress=0)

    # --- Qdrant Upsert ---
    if all_chunks:
        from rag_ingestion.stages.storage import store_chunks
        emit("storage", f"Storing {len(embedded_chunks)} chunks in Qdrant…")
        store_chunks(
            embedded_chunks,
            counters,
            collection_name=selected_collection,
            recreate_collection=False,
            embedding_dimensions=embedding_dimensions,
        )

    # --- Qdrant Deletion ---
    if old_vector_ids_to_delete:
        from rag_ingestion.stages.storage import delete_vectors_by_ids
        delete_vectors_by_ids(old_vector_ids_to_delete, collection_name=selected_collection)

    # --- DB Metadata Update ---
    if session_id:
        try:
            from retrieval.db import db_cursor, upsert_session_file, replace_session_file_chunks, mark_session_files_deleted
            last_indexed_at = datetime.now(timezone.utc).isoformat()

            chunks_by_file = {}
            for chunk in (embedded_chunks if all_chunks else []):
                if not chunk.relative_path or chunk.chunk_type == "repo_summary":
                    continue
                chunks_by_file.setdefault(chunk.relative_path, []).append(chunk)

            with db_cursor() as (conn, cursor):
                for rel_path in targets:
                    abs_path = root / rel_path
                    file_hash = ""
                    try:
                        if abs_path.is_file():
                            with open(abs_path, "rb") as f:
                                file_hash = hashlib.sha256(f.read()).hexdigest()
                        else:
                            file_hash = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()
                    except Exception:
                        file_hash = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()

                    file_record = upsert_session_file(
                        session_id=session_id,
                        repo_path=rel_path,
                        file_hash=file_hash,
                        indexed_commit_sha=commit_sha or "",
                        indexed_branch=branch_name or "",
                        status="indexed",
                        last_indexed_at=last_indexed_at,
                        deleted_at=None,
                        cursor=cursor,
                    )

                    file_chunks = chunks_by_file.get(rel_path, [])
                    chunk_mappings = []
                    for c in file_chunks:
                        chunk_mappings.append({
                            "chunk_id": c.chunk_id,
                            "vector_id": c.chunk_id,
                            "symbol": c.symbol_name or None,
                            "start_line": c.start_line if c.start_line > 0 else None,
                            "end_line": c.end_line if c.end_line > 0 else None,
                        })

                    replace_session_file_chunks(file_record["id"], chunk_mappings, cursor=cursor)

                if deleted_files:
                    mark_session_files_deleted(session_id, deleted_files, cursor=cursor)

        except Exception as e:
            logger.error("Failed to record incremental reindex metadata: %s", e)
            raise RuntimeError(f"Metadata recording failed: {e}") from e

    return counters


if __name__ == "__main__":
    main()
