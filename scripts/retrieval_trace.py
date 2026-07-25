#!/usr/bin/env python3
"""
retrieval_trace.py
==================
A CLI diagnostic script that executes the CodeSeek V2 retrieval pipeline
for a given query (avoiding LLM answer generation) and prints a structured,
human-readable summary of the routing decisions, extracted entities,
layer-by-layer search results, and retrieved display vs. reasoning sources.
"""

import sys
import argparse
import os
from pathlib import Path

# Add appropriate search paths to sys.path to find retrieval module
parent_path = Path(__file__).resolve().parent.parent
if str(parent_path) not in sys.path:
    sys.path.insert(0, str(parent_path))
backend_path = parent_path / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

try:
    from retrieval.query.query_processor import process_query
    from retrieval.search.searcher import (
        search,
        _dense_search,
        _lexical_search,
        _metadata_search,
        _exact_entity_search,
        _dependency_search,
        _local_content_match_candidates,
        _inject_direct_topics_candidates,
        _inject_code_topic_routing_candidates,
        _semantic_boost_discovery,
        _feature_recall_discovery,
        _framework_aware_discovery,
        match_code_topic_route,
    )
    from retrieval.search.expander import expand
    from retrieval.search.source_filter import (
        prune_exact_file_context,
        apply_feature_location_gate,
        apply_wrong_evidence_guard,
        prioritize_final_sources,
        split_sources_two_layer,
    )
    from retrieval.main import _align_display_sources_with_reasoning
    from retrieval.generation.assembler import assemble
    from retrieval.config import (
        get_collection_name,
        get_repo_root,
        ENABLE_TWO_LAYER_SOURCES,
        DISPLAY_SOURCES_CAP,
        ENABLE_LEXICAL_RETRIEVAL,
    )
    from retrieval.support.isolation import validate_collection_binding
    from retrieval.generation.code_answers import (
        is_overview_request,
        is_architecture_request,
    )
except ImportError as exc:
    print(f"[ERROR] Failed to import retrieval modules: {exc}", file=sys.stderr)
    print("Please make sure you run this script in an environment where 'backend' is in Python path.", file=sys.stderr)
    sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="retrieval_trace.py",
        description="Execute only the CodeSeek V2 retrieval pipeline and print trace diagnostics.",
    )
    p.add_argument("query", help="The raw query string to trace.")
    p.add_argument("--collection", help="Override active Qdrant collection name.")
    p.add_argument("--repo-root", help="Override active repository root path.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Override config environment variables if provided
    if args.collection:
        os.environ["QDRANT_COLLECTION_NAME"] = args.collection
    if args.repo_root:
        os.environ["RETRIEVAL_REPO_ROOT"] = args.repo_root

    collection = get_collection_name()
    repo_root = get_repo_root()

    print(f"Active collection: {collection}")
    print(f"Active repo root: {repo_root}")

    try:
        validate_collection_binding(collection, repo_root)
    except Exception as exc:
        print(f"[ERROR] Collection binding validation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 80)
    print("QUERY PROCESSING & INTENT CLASSIFICATION")
    print("=" * 80)

    # 1. Process query (classification and entity extraction)
    query_info = process_query(args.query)

    print(f"Raw Query      : {query_info.get('raw_query')}")
    print(f"Legacy Intent  : {query_info.get('intent')}")
    print(f"Primary Intent : {query_info.get('primary_intent')}")
    print(f"Response Mode  : {query_info.get('response_mode')}")
    print(f"Source Intent  : {query_info.get('source_intent')}")
    print(f"Confidence     : {query_info.get('confidence', 'N/A')}")
    print(f"Classifier Mode: {query_info.get('classifier_mode', 'N/A')}")

    entities = query_info.get("entities", {})
    print("\nExtracted Entities:")
    has_entities = False
    for k, v in entities.items():
        if v:
            print(f"  - {k:<15}: {v}")
            has_entities = True
    if not has_entities:
        print("  (none)")

    # 2. Detailed Layer Breakdown
    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER SEARCH RESULTS")
    print("=" * 80)

    # Run individual search layers
    dense_res = _dense_search(args.query)
    print(f"1. Dense Search Layer: retrieved {len(dense_res)} items")
    for idx, (p, score, src) in enumerate(dense_res[:5], start=1):
        print(f"   [{idx}] {p.get('relative_path')}:{p.get('start_line')}-{p.get('end_line')} | Score: {score:.4f}")

    lex_res = _lexical_search(args.query) if ENABLE_LEXICAL_RETRIEVAL else []
    print(f"\n2. Lexical Search Layer: retrieved {len(lex_res)} items")
    for idx, (p, score, src) in enumerate(lex_res[:5], start=1):
        print(f"   [{idx}] {p.get('relative_path')}:{p.get('start_line')}-{p.get('end_line')} | Score: {score:.4f}")

    meta_res = _metadata_search(args.query, entities, query_info)
    print(f"\n3. Metadata Search Layer: retrieved {len(meta_res)} items")
    for idx, (p, score, src) in enumerate(meta_res[:5], start=1):
        print(f"   [{idx}] {p.get('relative_path')}:{p.get('start_line')}-{p.get('end_line')} | Score: {score:.4f}")

    exact_res = _exact_entity_search(entities)
    print(f"\n4. Exact Entity Search Layer: retrieved {len(exact_res)} items")
    for idx, (p, score, src) in enumerate(exact_res[:5], start=1):
        print(f"   [{idx}] {p.get('relative_path')}:{p.get('start_line')}-{p.get('end_line')} | Score: {score:.4f}")

    direct_res = _inject_direct_topics_candidates(args.query, query_info.get("primary_intent"))
    print(f"\n5. Direct Topic Injection: retrieved {len(direct_res)} items")
    for idx, (p, score, src) in enumerate(direct_res[:10], start=1):
        print(f"   [{idx}] {p.get('relative_path')}:{p.get('start_line')}-{p.get('end_line')} | Score: {score:.4f} | Type: {p.get('chunk_type')}")

    matched_route = match_code_topic_route(args.query, query_info.get("primary_intent"))
    route_res = _inject_code_topic_routing_candidates(args.query, query_info.get("primary_intent"), matched_route)
    print(f"\n6. Code Topic Routing Injection: retrieved {len(route_res)} items")
    for idx, (p, score, src) in enumerate(route_res[:5], start=1):
        print(f"   [{idx}] {p.get('relative_path')}:{p.get('start_line')}-{p.get('end_line')} | Score: {score:.4f}")

    boost_res = _semantic_boost_discovery(args.query, entities, query_info)
    print(f"\n7. Semantic Boost Discovery: retrieved {len(boost_res)} items")
    for idx, (p, score, src) in enumerate(boost_res[:5], start=1):
        print(f"   [{idx}] {p.get('relative_path')}:{p.get('start_line')}-{p.get('end_line')} | Score: {score:.4f}")

    # 3. Retrieve initial candidates
    print("\n" + "=" * 80)
    print("FINAL MERGED & SORTED CANDIDATES (Capped at TOP_K_AFTER_MERGE)")
    print("=" * 80)
    candidates = search(query_info)

    if not candidates:
        print("  No initial candidates retrieved.")
    for idx, c in enumerate(candidates, 1):
        path = c.get("relative_path", "")
        start = c.get("start_line", 0)
        end = c.get("end_line", 0)
        ctype = c.get("chunk_type", "")
        score = c.get("retrieval_score", 0.0) or c.get("exact_entity_score", 0.0) or c.get("fusion_score", 0.0)
        source = c.get("support_kind", "") or c.get("retrieval_source", "search")
        symbol = c.get("symbol_name", "")
        symbol_str = f" | Symbol: {symbol}" if symbol else ""
        print(f"  {idx:2d}. {path}:{start}-{end} | Type: {ctype}{symbol_str} | Score: {score:.4f} | Source: {source}")

    # 4. Expansion, Pruning, Assembly, and Gating
    expanded = expand(candidates, query_info)
    expanded, pruning_diag = prune_exact_file_context(args.query, query_info, expanded)

    assemble_result = assemble(
        expanded,
        "",  # Empty history block for isolation
        primary_intent=query_info.get("primary_intent"),
        raw_query=args.query,
    )
    if len(assemble_result) == 4:
        _, sources, _, _ = assemble_result
    else:
        _, sources, _ = assemble_result

    # Apply standard post-retrieval pipeline stages
    sources, gate_diag = apply_feature_location_gate(args.query, sources)
    sources, guard_diag = apply_wrong_evidence_guard(args.query, sources, query_info)
    sources = prioritize_final_sources(args.query, sources, query_info)

    # 5. Display vs. Reasoning sources split
    display_sources, reasoning_sources = split_sources_two_layer(
        args.query, sources, enabled=ENABLE_TWO_LAYER_SOURCES
    )

    # Align display sources with reasoning
    display_sources = _align_display_sources_with_reasoning(
        display_sources,
        reasoning_sources,
        display_cap=8 if (is_overview_request(args.query) or is_architecture_request(args.query)) else DISPLAY_SOURCES_CAP,
        query_info=query_info,
    )

    print("\n" + "=" * 80)
    print("TWO-LAYER SOURCE ROUTING GATES")
    print("=" * 80)
    print(f"Two-Layer Sources Enabled: {ENABLE_TWO_LAYER_SOURCES}")
    print(f"Display Sources Cap      : {DISPLAY_SOURCES_CAP}")

    print(f"\nFinal Display Sources ({len(display_sources)} items):")
    display_keys = set()
    for idx, s in enumerate(display_sources, 1):
        path = s.get("relative_path", "")
        start = s.get("start_line", 0)
        end = s.get("end_line", 0)
        ctype = s.get("chunk_type", "")
        symbol = s.get("symbol_name", "")
        symbol_str = f" | Symbol: {symbol}" if symbol else ""
        score = s.get("score", 0.0)
        display_keys.add((path, start, end))
        print(f"  {idx:2d}. {path}:{start}-{end} | Type: {ctype}{symbol_str} | Score: {score:.4f}")

    print(f"\nReasoning Sources (context included but excluded from display citations):")
    reasoning_idx = 1
    for s in reasoning_sources:
        path = s.get("relative_path", "")
        start = s.get("start_line", 0)
        end = s.get("end_line", 0)
        if (path, start, end) in display_keys:
            continue
        ctype = s.get("chunk_type", "")
        symbol = s.get("symbol_name", "")
        symbol_str = f" | Symbol: {symbol}" if symbol else ""
        score = s.get("score", 0.0)
        print(f"  {reasoning_idx:2d}. {path}:{start}-{end} | Type: {ctype}{symbol_str} | Score: {score:.4f}")
        reasoning_idx += 1
    if reasoning_idx == 1:
        print("  (none)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
