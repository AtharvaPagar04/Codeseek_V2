#!/usr/bin/env python
"""
Runtime Qdrant storage integrity validation script.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/check_storage_integrity.py [COLLECTION_NAME]

Checks:
  1. Qdrant collection exists.
  2. points_count matches scrolled points.
  3. Every point ID equals payload chunk_id.
  4. Every payload has all required fields.
  5. content_excerpt exists and length <= CONTENT_EXCERPT_CHARS.
  6. No full `content` field stored in payload.
  7. No `embedding` field stored in payload.
  8. No .rag_ingestion_state.json is indexed.
  9. No duplicate chunk_id.
  10. chunk_part <= total_parts (0,0 allowed for synthetic chunks).
  11. Line ranges valid (0,0 allowed).
  12. repo_summary exists.
  13. Vector dimension matches EMBEDDING_DIM for a small sample.
"""

from __future__ import annotations

import os
import sys

from retrieval.support.qdrant_config import create_qdrant_client

from rag_ingestion.config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
)
from rag_ingestion.stages.storage import CONTENT_EXCERPT_CHARS
from retrieval.db import db_cursor

REQUIRED_PAYLOAD_KEYS = [
    "chunk_id", "file_path", "relative_path", "language", "chunk_type",
    "symbol_name", "qualified_symbol", "parent_symbol", "signature",
    "start_line", "end_line", "chunk_part", "total_parts", "token_count",
    "imports", "calls", "parameters", "methods", "file_symbols",
    "docstring", "summary", "description", "file_type", "summary_facts",
    "detected_frameworks", "dependencies", "dev_dependencies", "scripts",
    "services", "ports", "env_keys", "entrypoints", "config_tools",
    "build_system", "volumes", "service_dependencies", "base_image",
    "workdir", "package_manager", "feature_flags", "provider_keys",
    "purpose", "setup_steps", "usage_commands", "architecture_notes",
    "content_excerpt",
]


def get_collection_name() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    try:
        with db_cursor() as (conn, cursor):
            row = cursor.execute(
                "SELECT collection FROM repo_sessions ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            if row and row["collection"]:
                coll = row["collection"]
                print(f"Using latest collection from database: {coll}")
                return coll
    except Exception as e:
        print(f"Could not fetch collection from database: {e}")
    coll = os.getenv("QDRANT_COLLECTION_NAME", COLLECTION_NAME)
    print(f"Using fallback config/env collection: {coll}")
    return coll


def main():
    collection_name = get_collection_name()
    print(f"Connecting to Qdrant...")
    client = create_qdrant_client(check_compatibility=False)

    errors: list[str] = []

    # --- Check 1: Collection exists ---
    try:
        info = client.get_collection(collection_name)
    except Exception as e:
        print(f"FATAL: Collection '{collection_name}' not found: {e}")
        sys.exit(1)

    reported_count = info.points_count

    # --- Scroll all points (no vectors) ---
    print(f"Scrolling all points in collection '{collection_name}'...")
    all_points = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=500,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        all_points.extend(points)
        if offset is None:
            break

    scrolled_count = len(all_points)
    print(f"Reported points_count: {reported_count} | Scrolled: {scrolled_count}")

    # --- Check 2: points_count matches scrolled ---
    if reported_count != scrolled_count:
        errors.append(
            f"points_count mismatch: Qdrant reports {reported_count}, scrolled {scrolled_count}."
        )

    # --- Scroll sample with vectors for dimension check ---
    sample_points, _ = client.scroll(
        collection_name=collection_name,
        limit=5,
        with_payload=False,
        with_vectors=True,
    )

    # --- Check 13: vector dimensions ---
    for pt in sample_points:
        vec = pt.vector
        if vec is None:
            errors.append(f"Point {pt.id}: vector is None.")
        elif len(vec) != EMBEDDING_DIM:
            errors.append(
                f"Point {pt.id}: vector dim={len(vec)}, expected {EMBEDDING_DIM}."
            )

    # --- Per-point checks ---
    seen_chunk_ids: set[str] = set()
    repo_summary_count = 0

    for point in all_points:
        payload = point.payload or {}
        point_id = str(point.id)
        chunk_id = payload.get("chunk_id", "")
        relative_path = payload.get("relative_path", "")

        # Check 3: point ID == payload chunk_id
        # Qdrant may format UUIDs with dashes; chunk_id is stored as raw hex without dashes.
        if point_id.replace("-", "") != chunk_id.replace("-", ""):
            errors.append(
                f"Point {point_id}: id does not match payload chunk_id='{chunk_id}'."
            )

        # Check 4: required fields
        for key in REQUIRED_PAYLOAD_KEYS:
            if key not in payload:
                errors.append(f"Chunk '{chunk_id}' ({relative_path}): missing payload key '{key}'.")

        # Check 5: content_excerpt bounded
        excerpt = payload.get("content_excerpt")
        if excerpt is None:
            errors.append(f"Chunk '{chunk_id}' ({relative_path}): content_excerpt is None.")
        elif len(excerpt) > CONTENT_EXCERPT_CHARS:
            errors.append(
                f"Chunk '{chunk_id}' ({relative_path}): content_excerpt too long "
                f"({len(excerpt)} > {CONTENT_EXCERPT_CHARS})."
            )

        # Check 6: no full content field
        if "content" in payload:
            errors.append(f"Chunk '{chunk_id}' ({relative_path}): full 'content' stored in payload.")

        # Check 7: no embedding in payload
        if "embedding" in payload:
            errors.append(f"Chunk '{chunk_id}' ({relative_path}): 'embedding' stored in payload.")

        # Check 8: no state file indexed
        if ".rag_ingestion_state.json" in relative_path:
            errors.append(f"State file indexed: '{relative_path}'.")

        # Check 9: no duplicate chunk_ids
        if chunk_id in seen_chunk_ids:
            errors.append(f"Duplicate chunk_id detected: '{chunk_id}'.")
        else:
            seen_chunk_ids.add(chunk_id)

        # Check 10: chunk_part <= total_parts and both >= 1
        chunk_part = payload.get("chunk_part", 0)
        total_parts = payload.get("total_parts", 0)
        if chunk_part < 1 or total_parts < 1 or chunk_part > total_parts:
            errors.append(
                f"Chunk '{chunk_id}' ({relative_path}): "
                f"bad chunk part metadata: {chunk_part}/{total_parts}."
            )

        # Check 11: line ranges valid (allow 0,0)
        start_line = payload.get("start_line", 0)
        end_line = payload.get("end_line", 0)
        if not (start_line == 0 and end_line == 0):
            if end_line < start_line:
                errors.append(
                    f"Chunk '{chunk_id}' ({relative_path}): "
                    f"end_line={end_line} < start_line={start_line}."
                )

        # Check 12: track repo_summary
        if payload.get("chunk_type") == "repo_summary":
            repo_summary_count += 1

    # Check 12: repo_summary must exist
    if repo_summary_count == 0:
        errors.append("No repo_summary chunk found in collection.")

    unique_ids = len(seen_chunk_ids)

    # --- Report ---
    print()
    print("=" * 60)
    print("Storage Integrity Validation")
    print("=" * 60)
    print(f"Collection:            {collection_name}")
    print(f"Qdrant points_count:   {reported_count}")
    print(f"Scrolled points:       {scrolled_count}")
    print(f"Unique chunk_ids:      {unique_ids}")
    print(f"Repo summary chunks:   {repo_summary_count}")
    print(f"EMBEDDING_DIM:         {EMBEDDING_DIM}")
    print(f"CONTENT_EXCERPT_CHARS: {CONTENT_EXCERPT_CHARS}")
    print(f"Errors:                {len(errors)}")
    print("=" * 60)

    if errors:
        print("\n❌ Storage Integrity FAILED! Details (up to 40 shown):")
        for err in errors[:40]:
            print(f"  - {err}")
        if len(errors) > 40:
            print(f"  ... and {len(errors) - 40} more.")
        print("\nVERDICT: FAIL ❌")
        sys.exit(1)
    else:
        print("\nVERDICT: SUCCESS ✅")
        sys.exit(0)


if __name__ == "__main__":
    main()
