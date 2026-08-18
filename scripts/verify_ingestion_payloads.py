#!/usr/bin/env python3
"""Verify the metadata payloads produced by a clean RAG ingestion run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = SCRIPT_ROOT / "backend"
if not BACKEND_ROOT.is_dir():
    # Supports running the script from the backend container, where /app is
    # already the import root and the repository-level wrapper is absent.
    SCRIPT_ROOT = Path.cwd()
    BACKEND_ROOT = SCRIPT_ROOT
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from qdrant_client.models import FieldCondition, Filter, MatchValue

from retrieval.support.isolation import expected_collection_name
from retrieval.support.qdrant_config import create_qdrant_client


def _collection_name(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.exists() and candidate.is_dir():
        return expected_collection_name(str(candidate.resolve()))
    return value


def _scroll(client, collection: str, *, field: str, value: str) -> list:
    points: list = []
    offset = None
    scroll_filter = Filter(
        must=[FieldCondition(key=field, match=MatchValue(value=value))]
    )
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        points.extend(batch)
        if offset is None:
            return points


def _payloads(client, collection: str, **conditions: str) -> list[dict]:
    payloads = None
    for field, value in conditions.items():
        matches = _scroll(client, collection, field=field, value=value)
        current = [point.payload or {} for point in matches]
        payloads = current if payloads is None else [
            payload for payload in payloads if payload in current
        ]
    return payloads or []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("collection_or_repo", help="Qdrant collection name or indexed repo path")
    args = parser.parse_args()
    collection = _collection_name(args.collection_or_repo)

    print(f"Collection: {collection}")
    client = create_qdrant_client(check_compatibility=False)
    client.get_collection(collection)

    failures: list[str] = []

    env_payloads = _payloads(client, collection, relative_path=".env.example")
    env_keys = sorted({
        key
        for payload in env_payloads
        for key in (payload.get("env_keys") or [])
    })
    print(f"[env] payloads={len(env_payloads)}")
    print(f"[env] env_keys={json.dumps(env_keys)}")
    boilerplate = {"DATABASE_URL", "OAUTH_SECRET"}
    remaining = sorted(boilerplate.intersection(env_keys))
    if remaining:
        failures.append(f"unused boilerplate keys remain: {remaining}")
    else:
        print("[env] PASS: DATABASE_URL and OAUTH_SECRET are absent")

    summary_payloads = _payloads(client, collection, chunk_type="repo_summary")
    print(f"[repo_summary] payloads={len(summary_payloads)}")
    for payload in summary_payloads:
        content = str(payload.get("content") or payload.get("summary") or "")
        print(content[:500])
        print(f"[repo_summary] summary_facts={json.dumps(payload.get('summary_facts') or [])}")
    if not summary_payloads:
        failures.append("repo_summary payload is missing")
    elif not any(
        "## Project Directory Structure" in str(
            payload.get("content") or payload.get("summary") or ""
        )
        for payload in summary_payloads
    ):
        failures.append("repo_summary directory tree heading is missing")
    else:
        print("[repo_summary] PASS: directory tree heading is present")

    logging_payloads = _payloads(
        client,
        collection,
        relative_path="bot/logging_config.py",
    )
    logging_facts = [
        fact
        for payload in logging_payloads
        for fact in (payload.get("summary_facts") or [])
    ]
    print(f"[logging] payloads={len(logging_payloads)}")
    print(f"[logging] summary_facts={json.dumps(logging_facts)}")
    if not logging_payloads:
        failures.append("bot/logging_config.py payload is missing")
    elif not any(fact.startswith("Log Level:") for fact in logging_facts):
        failures.append("logging level facts are missing")
    elif not any(fact.startswith("Handler:") for fact in logging_facts):
        failures.append("logging handler facts are missing")
    elif not any(fact.startswith("File:") for fact in logging_facts):
        failures.append("logging file facts are missing")
    else:
        print("[logging] PASS: levels, handlers, and file facts are present")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: ingestion payload verification completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
