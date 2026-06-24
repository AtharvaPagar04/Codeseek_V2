#!/usr/bin/env python
"""
Runtime Qdrant validation script to verify embedding inputs after ingestion.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/check_embedding_inputs.py [COLLECTION_NAME]
"""

from __future__ import annotations

import os
import sys
from retrieval.support.qdrant_config import create_qdrant_client

from rag_ingestion.config import (
    COLLECTION_NAME,
    EMBEDDING_INPUT_MAX_TOTAL_CHARS,
)
from rag_ingestion.models.chunk import Chunk
from rag_ingestion.stages.embedder import _embedding_input, KNOWN_LABELS
from retrieval.db import db_cursor


def get_collection_name() -> str:
    # 1. Check CLI argument
    if len(sys.argv) > 1:
        return sys.argv[1]

    # 2. Check Database for latest collection
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

    # 3. Fallback to default from config
    coll = os.getenv("QDRANT_COLLECTION_NAME", COLLECTION_NAME)
    print(f"Using fallback config/env collection: {coll}")
    return coll


def main():
    collection_name = get_collection_name()
    print(f"Connecting to Qdrant...")
    client = create_qdrant_client(check_compatibility=False)

    try:
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

    samples = {
        "readme": None,
        "package_json": None,
        "docker_compose": None,
        "dockerfile": None,
        "repo_summary": None,
        "function": None,
    }

    errors = []

    # Map payload key to expected label and a converter function for comparison
    fields_to_check = {
        "relative_path": "File",
        "language": "Language",
        "chunk_type": "Type",
        "file_type": "File Type",
        "symbol_name": "Symbol",
        "qualified_symbol": "Qualified Symbol",
        "parent_symbol": "Parent Symbol",
        "signature": "Signature",
        "summary": "Summary",
        "description": "Description",
        "purpose": "Purpose",
        "build_system": "Build System",
        "base_image": "Base Image",
        "workdir": "Workdir",
        "package_manager": "Package Manager",
        "docstring": "Docstring",
    }

    list_fields_to_check = {
        "summary_facts": "Facts",
        "detected_frameworks": "Frameworks",
        "dependencies": "Dependencies",
        "dev_dependencies": "Dev Dependencies",
        "services": "Services",
        "ports": "Ports",
        "env_keys": "Environment Keys",
        "feature_flags": "Feature Flags",
        "provider_keys": "Provider Keys",
        "entrypoints": "Entrypoints",
        "config_tools": "Config Tools",
        "volumes": "Volumes",
        "setup_steps": "Setup Steps",
        "usage_commands": "Usage Commands",
        "architecture_notes": "Architecture Notes",
        "parameters": "Parameters",
        "methods": "Methods",
        "file_symbols": "File Symbols",
    }

    for point in all_points:
        payload = point.payload or {}
        
        # Reconstruct Chunk
        chunk_dict = {}
        for field_name in Chunk.__dataclass_fields__:
            if field_name == "content":
                chunk_dict["content"] = payload.get("content_excerpt", "")
            elif field_name in payload:
                chunk_dict[field_name] = payload[field_name]
        chunk = Chunk(**chunk_dict)

        # Generate embedding input
        embedding_input = _embedding_input(chunk)

        # Track samples
        rel_path = chunk.relative_path.lower()
        chunk_type = chunk.chunk_type
        
        if chunk_type == "repo_summary" and not samples["repo_summary"]:
            samples["repo_summary"] = (chunk, embedding_input)
        elif chunk_type == "function" and not samples["function"]:
            samples["function"] = (chunk, embedding_input)
        elif chunk_type == "file":
            if "readme" in rel_path and not samples["readme"]:
                samples["readme"] = (chunk, embedding_input)
            elif "package.json" in rel_path and not samples["package_json"]:
                samples["package_json"] = (chunk, embedding_input)
            elif "docker-compose" in rel_path and not samples["docker_compose"]:
                samples["docker_compose"] = (chunk, embedding_input)
            elif "dockerfile" in rel_path and not samples["dockerfile"]:
                samples["dockerfile"] = (chunk, embedding_input)

        # AUTOMATED VALIDATION
        # Check 1: total length <= configured max + small margin
        limit = EMBEDDING_INPUT_MAX_TOTAL_CHARS + 25
        if len(embedding_input) > limit:
            errors.append(
                f"Chunk '{chunk.relative_path}' embedding input too long "
                f"({len(embedding_input)} chars, limit is {limit})."
            )

        # Check 2: metadata appears before Code:
        lines = embedding_input.splitlines()
        code_idx = -1
        for idx, line in enumerate(lines):
            if line.strip() == "Code:":
                code_idx = idx
                break

        metadata_lines = lines[:code_idx] if code_idx != -1 else lines

        # Group lines back together for multi-line fields
        grouped_metadata = []
        for line in metadata_lines:
            if not line.strip():
                continue
            has_known_prefix = False
            for kl in KNOWN_LABELS:
                if line.startswith(f"{kl}:"):
                    has_known_prefix = True
                    break
            if has_known_prefix:
                grouped_metadata.append(line)
            else:
                if grouped_metadata:
                    grouped_metadata[-1] += "\n" + line
                else:
                    grouped_metadata.append(line)

        # Check 3: no empty labels, and no unmapped labels
        for idx, line in enumerate(grouped_metadata):
            if ":" not in line:
                errors.append(f"Chunk '{chunk.relative_path}': line '{line}' missing ':' divider.")
                continue
            label, val = line.split(":", 1)
            label = label.strip()
            val = val.strip()
            if label not in KNOWN_LABELS:
                errors.append(f"Chunk '{chunk.relative_path}': line '{line}' has unknown label '{label}'.")
            if not val:
                errors.append(f"Chunk '{chunk.relative_path}': line '{line}' has empty value.")

        # Check 4: structured fields are present when payload has them
        # Simple string fields
        for payload_key, label_name in fields_to_check.items():
            val = payload.get(payload_key)
            if val and str(val).strip():
                expected_prefix = f"{label_name}:"
                if expected_prefix not in embedding_input:
                    errors.append(
                        f"Chunk '{chunk.relative_path}': missing '{label_name}' "
                        f"field in embedding input despite having value '{val}'."
                    )

        # List fields
        for payload_key, label_name in list_fields_to_check.items():
            val_list = payload.get(payload_key)
            if val_list and any(str(item).strip() for item in val_list):
                expected_prefix = f"{label_name}:"
                if expected_prefix not in embedding_input:
                    errors.append(
                        f"Chunk '{chunk.relative_path}': missing '{label_name}' "
                        f"field in embedding input despite having list elements."
                    )

        # Dict fields
        if payload.get("scripts"):
            if "Scripts:" not in embedding_input:
                errors.append(f"Chunk '{chunk.relative_path}': missing 'Scripts' label in embedding input.")
        if payload.get("service_dependencies"):
            if "Service Dependencies:" not in embedding_input:
                errors.append(f"Chunk '{chunk.relative_path}': missing 'Service Dependencies' label in embedding input.")

    # PRINT SAMPLES
    print("=" * 60)
    print("SAMPLE EMBEDDING INPUTS")
    print("=" * 60)
    for name, sample_tuple in samples.items():
        print(f"\n--- SAMPLE: {name.upper()} ---")
        if not sample_tuple:
            print("[Not found in collection]")
            continue
        chunk, embedding_input = sample_tuple
        print(f"Path: {chunk.relative_path} | Type: {chunk.chunk_type}")
        print("-" * 40)
        # Show first 800 chars and last 200 chars if long
        if len(embedding_input) > 1000:
            print(embedding_input[:800])
            print("...\n[omitted content]\n...")
            print(embedding_input[-200:])
        else:
            print(embedding_input)
        print("-" * 40)

    # FINAL REPORT
    print("\n" + "=" * 60)
    print("EMBEDDING INPUTS VALIDATION REPORT")
    print("=" * 60)
    print(f"Collection Name:       {collection_name}")
    print(f"Total Points scrolled: {total_points}")
    print(f"Total Quality Errors:  {len(errors)}")
    print("=" * 60)

    if errors:
        print("\n❌ Embedding Input Validation Failed! Details (up to 30 shown):")
        for err in errors[:30]:
            print(f"- {err}")
        if len(errors) > 30:
            print(f"... and {len(errors) - 30} more errors.")
        print("\nVERDICT: FAIL ❌")
        sys.exit(1)
    else:
        print("\nVERDICT: SUCCESS ✅")
        sys.exit(0)


if __name__ == "__main__":
    main()
