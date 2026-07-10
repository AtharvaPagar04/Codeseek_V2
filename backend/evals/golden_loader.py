"""Loader and validator for golden query evaluation datasets."""

import os
import sys
from pathlib import Path
import yaml

def load_golden_queries(yaml_path: str | Path) -> list[dict]:
    """Load, validate, and normalize golden queries from a YAML file."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Golden queries file not found: {yaml_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse YAML file {yaml_path}: {str(e)}")

    if not isinstance(data, list):
        raise ValueError(f"Golden queries file must contain a list of queries, got {type(data)}")

    seen_ids = set()
    validated_queries = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} is not a dictionary")

        qid = item.get("id")
        if not qid:
            raise ValueError(f"Item at index {idx} is missing required field 'id'")
        if qid in seen_ids:
            raise ValueError(f"Duplicate query ID found: '{qid}'")
        seen_ids.add(qid)

        query = item.get("query")
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError(f"Query '{qid}' has missing or invalid 'query' field")

        category = item.get("category")
        if not category or not isinstance(category, str) or not category.strip():
            raise ValueError(f"Query '{qid}' has missing or invalid 'category' field")

        # Intent fields validation
        expected_intent = item.get("expected_intent")
        if expected_intent and not isinstance(expected_intent, str):
            raise ValueError(f"Query '{qid}' has invalid 'expected_intent'")

        expected_reranker_intent = item.get("expected_reranker_intent")
        if expected_reranker_intent and not isinstance(expected_reranker_intent, str):
            raise ValueError(f"Query '{qid}' has invalid 'expected_reranker_intent'")

        # Validate that at least one verification list is present
        has_expectations = any([
            item.get("expected_files"),
            item.get("expected_symbols"),
            item.get("expected_file_types"),
            item.get("expected_labels_in_top1"),
            item.get("expected_labels_in_top3"),
            item.get("expected_labels_in_top5"),
        ])
        if not has_expectations:
            raise ValueError(f"Query '{qid}' has no expectations (expected_files, expected_symbols, expected_labels, etc.)")

        # Normalize paths in expected_files
        expected_files = item.get("expected_files")
        if expected_files:
            if not isinstance(expected_files, list):
                raise ValueError(f"Query '{qid}' expected_files must be a list")
            item["expected_files"] = [str(Path(f).as_posix()).strip().lstrip("/") for f in expected_files]

        # Normalize symbols
        expected_symbols = item.get("expected_symbols")
        if expected_symbols:
            if not isinstance(expected_symbols, list):
                raise ValueError(f"Query '{qid}' expected_symbols must be a list")
            item["expected_symbols"] = [str(s).strip() for s in expected_symbols]

        # must_hit_top_k validation
        must_hit_top_k = item.get("must_hit_top_k", 5)
        try:
            must_hit_top_k = int(must_hit_top_k)
            if must_hit_top_k < 1:
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError(f"Query '{qid}' must_hit_top_k must be an integer >= 1")
        item["must_hit_top_k"] = must_hit_top_k

        validated_queries.append(item)

    return validated_queries

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python golden_loader.py <path_to_yaml>")
        sys.exit(1)

    try:
        queries = load_golden_queries(sys.argv[1])
        print(f"Successfully loaded and validated {len(queries)} golden queries.")
        for q in queries[:3]:
            print(f"- [{q['id']}] ({q['category']}): {q['query']}")
        if len(queries) > 3:
            print("...")
    except Exception as e:
        print(f"Validation FAILED: {str(e)}")
        sys.exit(1)
