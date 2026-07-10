#!/usr/bin/env python
"""
Runtime Qdrant validation script to verify summary quality and structured facts after ingestion.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/check_summary_quality.py [COLLECTION_NAME]
"""

import os
import sys
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

    errors = []
    repo_summaries = []
    readme_files = []
    config_files = []
    other_files = 0

    # Config file type detectors based on typical names
    config_suffixes = (
        "package.json", "requirements.txt", "pyproject.toml", "docker-compose.yml",
        "docker-compose.yaml", "dockerfile", "tsconfig.json", "jsconfig.json",
        "next.config.js", "next.config.ts", "next.config.mjs",
        "vite.config.js", "vite.config.ts", "vite.config.mjs",
        "tailwind.config.js", "tailwind.config.ts", "tailwind.config.mjs",
        "postcss.config.js", "postcss.config.mjs",
        "eslint.config.js", "eslint.config.mjs",
        "vercel.json", "netlify.toml", "turbo.json",
        "caddyfile", "nginx.conf", "pnpm-workspace.yaml",
        "render.yaml", "railway.json", ".env.example"
    )

    for point in all_points:
        payload = point.payload or {}
        chunk_type = payload.get("chunk_type")
        relative_path = payload.get("relative_path", "")
        rel_lower = relative_path.lower()

        if chunk_type == "repo_summary":
            repo_summaries.append(payload)
        elif chunk_type == "file":
            if "readme" in rel_lower:
                readme_files.append(payload)
            elif rel_lower.endswith(config_suffixes) or payload.get("file_type") in ("config", "json", "yaml", "toml"):
                config_files.append(payload)
            else:
                other_files += 1

    # 1. Validate repo_summary
    if not repo_summaries:
        errors.append("Missing repo_summary: No repository summary chunk found.")
    else:
        for idx, rs in enumerate(repo_summaries):
            facts = rs.get("summary_facts", [])
            summary_text = rs.get("summary", "")
            if not summary_text:
                errors.append(f"repo_summary[{idx}]: Summary is empty.")
            if not facts:
                errors.append(f"repo_summary[{idx}]: summary_facts list is empty.")

    # 2. Validate README files
    for rdm in readme_files:
        path = rdm.get("relative_path")
        purpose = rdm.get("purpose")
        facts = rdm.get("summary_facts", [])
        if not purpose:
            errors.append(f"README '{path}': Missing purpose/description.")
        if not facts:
            errors.append(f"README '{path}': summary_facts is empty.")

    # 3. Validate Config files
    for cfg in config_files:
        path = cfg.get("relative_path")
        file_type = cfg.get("file_type")
        facts = cfg.get("summary_facts", [])
        if not file_type:
            errors.append(f"Config '{path}': Missing 'file_type' field.")
        if not facts:
            errors.append(f"Config '{path}': summary_facts is empty.")

    # Print Samples
    print("=" * 60)
    print("SAMPLE DETECTED SUMMARIES & METADATA")
    print("=" * 60)

    if repo_summaries:
        print(f"\n[REPO SUMMARY] ({len(repo_summaries)} chunk(s)):")
        for idx, rs in enumerate(repo_summaries[:2]):
            print(f"--- Chunk {idx + 1} ---")
            print(f"Summary: {rs.get('summary')[:300]}...")
            print(f"Facts: {rs.get('summary_facts')[:5]}")

    if readme_files:
        print(f"\n[README FILES] ({len(readme_files)} file(s)):")
        for idx, rdm in enumerate(readme_files[:2]):
            print(f"--- {rdm.get('relative_path')} ---")
            print(f"Purpose: {rdm.get('purpose')}")
            print(f"Setup Steps: {rdm.get('setup_steps')}")
            print(f"Usage Commands: {rdm.get('usage_commands')}")
            print(f"Facts: {rdm.get('summary_facts')[:4]}")

    if config_files:
        print(f"\n[CONFIG FILES] ({len(config_files)} file(s)):")
        for idx, cfg in enumerate(config_files[:4]):
            print(f"--- {cfg.get('relative_path')} ---")
            print(f"File Type: {cfg.get('file_type')}")
            print(f"Frameworks: {cfg.get('detected_frameworks')}")
            print(f"Tooling: {cfg.get('config_tools')}")
            print(f"Facts: {cfg.get('summary_facts')[:4]}")

    # Summary Report
    print("\n" + "=" * 60)
    print("SUMMARY QUALITY VALIDATION REPORT")
    print("=" * 60)
    print(f"Collection Name:       {collection_name}")
    print(f"Total Points scrolled: {total_points}")
    print(f"Repo Summary Chunks:   {len(repo_summaries)}")
    print(f"README Chunks:         {len(readme_files)}")
    print(f"Config Chunks:         {len(config_files)}")
    print(f"Other File Chunks:     {other_files}")
    print(f"Total Quality Errors:  {len(errors)}")
    print("=" * 60)

    if errors:
        print("\n❌ Quality Validation Failed! Details (up to 30 shown):")
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
