#!/usr/bin/env python
"""
Runtime Qdrant validation script to verify metadata and payloads are correct after ingestion.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/check_metadata_payloads.py [COLLECTION_NAME]
"""

import sys
import os
from retrieval.support.qdrant_config import create_qdrant_client
from rag_ingestion.config import COLLECTION_NAME
from retrieval.db import db_cursor


def get_collection_name() -> str:
    # 1. Check CLI argument
    if len(sys.argv) > 1:
        return sys.argv[1]

    # 2. Check Database for latest collection
    try:
        with db_cursor() as (conn, cursor):
            # Sort by updated_at desc to find latest session
            row = cursor.execute(
                "SELECT collection FROM repo_sessions ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            if row and row["collection"]:
                coll = row["collection"]
                print(f"Using latest collection from database: {coll}")
                return coll
    except Exception as e:
        print(f"Could not fetch collection from database: {e}")

    # 3. Fallback to default from config
    coll = os.getenv("QDRANT_COLLECTION_NAME", COLLECTION_NAME)
    print(f"Using fallback config/env collection: {coll}")
    return coll


def main():
    collection_name = get_collection_name()
    print(f"Connecting to Qdrant...")
    client = create_qdrant_client(check_compatibility=False)

    try:
        # Check if collection exists
        client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: Collection '{collection_name}' not found or Qdrant connection failed: {e}")
        sys.exit(1)

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

    total_points = len(all_points)
    print(f"Retrieved {total_points} points.")

    errors = []
    chunk_ids = set()
    repo_summary_count = 0
    state_file_count = 0

    required_fields = [
        "chunk_id",
        "relative_path",
        "language",
        "chunk_type",
        "qualified_symbol",
        "start_line",
        "end_line",
        "chunk_part",
        "total_parts",
        "token_count",
        "summary",
    ]

    for idx, point in enumerate(all_points):
        payload = point.payload or {}
        point_id_str = f"Point {point.id}"

        # 1. Validate required payload fields
        for field in required_fields:
            if field not in payload:
                errors.append(f"{point_id_str}: Missing required field '{field}'")
            elif payload[field] is None:
                errors.append(f"{point_id_str}: Field '{field}' is None")

        # Get values for further validation
        chunk_id = payload.get("chunk_id")
        relative_path = payload.get("relative_path")
        chunk_type = payload.get("chunk_type")
        start_line = payload.get("start_line")
        end_line = payload.get("end_line")
        chunk_part = payload.get("chunk_part")
        total_parts = payload.get("total_parts")

        # 2. Check chunk_id uniqueness
        if chunk_id:
            if chunk_id in chunk_ids:
                errors.append(f"{point_id_str}: Duplicate chunk_id '{chunk_id}'")
            chunk_ids.add(chunk_id)

        # 3. Check line ranges
        if start_line is not None and end_line is not None:
            try:
                start_line_val = int(start_line)
                end_line_val = int(end_line)
                if start_line_val < 0:
                    errors.append(f"{point_id_str}: start_line is negative: {start_line_val}")
                if end_line_val < 0:
                    errors.append(f"{point_id_str}: end_line is negative: {end_line_val}")
                # Tolerate start_line=1, end_line=0 as a legacy empty-file representation
                if start_line_val > end_line_val and not (start_line_val == 1 and end_line_val == 0):
                    errors.append(
                        f"{point_id_str}: start_line ({start_line_val}) > end_line ({end_line_val})"
                    )
            except (ValueError, TypeError):
                errors.append(f"{point_id_str}: line range contains non-integer values: {start_line}, {end_line}")

        # 4. Check chunk_part/total_parts validity (1-based indexing for standard chunks, 0/1 for repo_summary)
        if chunk_part is not None and total_parts is not None:
            try:
                chunk_part_val = int(chunk_part)
                total_parts_val = int(total_parts)
                if chunk_type == "repo_summary":
                    if chunk_part_val not in (0, 1) or total_parts_val not in (0, 1):
                        errors.append(
                            f"{point_id_str}: repo_summary chunk has unexpected part/total: chunk_part={chunk_part_val}, total_parts={total_parts_val}"
                        )
                else:
                    if total_parts_val < 1:
                        errors.append(f"{point_id_str}: total_parts < 1: {total_parts_val}")
                    if not (1 <= chunk_part_val <= total_parts_val):
                        errors.append(
                            f"{point_id_str}: Invalid part range (1-based expected): chunk_part={chunk_part_val}, total_parts={total_parts_val}"
                        )
            except (ValueError, TypeError):
                errors.append(f"{point_id_str}: part numbers contain non-integer values: {chunk_part}, {total_parts}")

        # 5. Check if repo_summary
        if chunk_type == "repo_summary":
            repo_summary_count += 1

        # 6. Check state files are absent
        if relative_path and ".rag_ingestion_state.json" in str(relative_path):
            state_file_count += 1
            errors.append(f"{point_id_str}: State file found in index: '{relative_path}'")

    # Final validation for existence of repo_summary
    if repo_summary_count == 0:
        errors.append("Collection: No 'repo_summary' chunk found in the collection.")

    # Print Report
    print("=" * 60)
    print("METADATA & PAYLOAD VALIDATION REPORT")
    print("=" * 60)
    print(f"Collection Name:     {collection_name}")
    print(f"Total Chunks/Points: {total_points}")
    print(f"Unique Chunk IDs:    {len(chunk_ids)}")
    print(f"Repo Summary Chunks: {repo_summary_count}")
    print(f"State Files Found:   {state_file_count}")
    print(f"Total Validation Errs:{len(errors)}")
    print("=" * 60)

    if errors:
        print("\n❌ Validation Failed! Detail of errors (up to 50 shown):")
        for err in errors[:50]:
            print(f"- {err}")
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more errors.")
        print("\nVERDICT: FAIL ❌")
        sys.exit(1)
    else:
        print("\nVERDICT: SUCCESS ✅")
        sys.exit(0)


if __name__ == "__main__":
    main()
