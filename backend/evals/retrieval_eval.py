"""Evaluation runner for CodeSeek retrieval pipeline."""


import os
import sys
import json
import time
from pathlib import Path
import argparse

# Load .env file before importing retrieval config
def _load_env_file():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                key = k.strip()
                if key not in os.environ:
                    os.environ[key] = v.strip()

_load_env_file()

# Ensure backend directory is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.query.query_processor import process_query
from retrieval.search.searcher import (
    search, 
    _dense_search, 
    _lexical_search, 
    _metadata_search, 
    _exact_entity_search, 
    _dependency_search
)
from retrieval.query.query_intent import classify_query_intent, map_label_intent_to_reranker_intent
from retrieval.db import db_cursor

from evals.golden_loader import load_golden_queries
from evals.metrics import (
    compute_file_hit,
    compute_symbol_hit,
    compute_label_hit,
    compute_protected_exact_hit_preserved,
    audit_exact_hit_preservation,
    compute_duplicate_context_rate
)

def format_point(payload: dict, score: float, source: str) -> dict:
    """Format candidate tuple to standard JSON serializable dict."""
    return {
        "chunk_id": payload.get("chunk_id", ""),
        "relative_path": payload.get("relative_path", ""),
        "symbol_name": payload.get("symbol_name", ""),
        "qualified_symbol": payload.get("qualified_symbol", ""),
        "chunk_type": payload.get("chunk_type", ""),
        "file_type": payload.get("file_type", ""),
        "labels": payload.get("labels", []),
        "code_intent": payload.get("code_intent", ""),
        "score": score,
        "source_layer": source
    }

def format_final_point(p: dict) -> dict:
    """Format candidate dict returned by search() to standard JSON serializable dict."""
    return {
        "chunk_id": p.get("chunk_id", ""),
        "relative_path": p.get("relative_path", ""),
        "symbol_name": p.get("symbol_name", ""),
        "qualified_symbol": p.get("qualified_symbol", ""),
        "chunk_type": p.get("chunk_type", ""),
        "file_type": p.get("file_type", ""),
        "labels": p.get("labels", []),
        "code_intent": p.get("code_intent", ""),
        "score": p.get("retrieval_score", 0.0),
        "source_layer": "merged",
        "file_symbols": p.get("file_symbols", []),
        "calls": p.get("calls", []),
        "imports": p.get("imports", []),
        "content": p.get("content", ""),
        "content_excerpt": p.get("content_excerpt", ""),
        "summary": p.get("summary", ""),
    }
def _safe_round(value, digits: int = 6):
    if isinstance(value, (int, float)):
        return round(value, digits)
    return value


def _candidate_excerpt(candidate: dict, limit: int = 200) -> str:
    text = (
        candidate.get("content")
        or candidate.get("content_excerpt")
        or candidate.get("summary")
        or candidate.get("code_intent")
        or ""
    )
    text = str(text).replace("\n", " ").strip()
    return text[:limit]


def _compact_candidate(candidate: dict, rank: int | None = None, include_excerpt: bool = False) -> dict:
    item = {
        "rank": rank,
        "chunk_id": candidate.get("chunk_id"),
        "relative_path": candidate.get("relative_path"),
        "symbol_name": candidate.get("symbol_name"),
        "qualified_symbol": candidate.get("qualified_symbol"),
        "chunk_type": candidate.get("chunk_type"),
        "file_type": candidate.get("file_type"),
        "labels": candidate.get("labels") or [],
        "source_layers": candidate.get("source_layers") or [],
        "vector_score": _safe_round(candidate.get("vector_score")),
        "exact_match_score": _safe_round(candidate.get("exact_match_score")),
        "label_boost": _safe_round(candidate.get("label_boost")),
        "path_symbol_boost": _safe_round(candidate.get("path_symbol_boost")),
        "final_score": _safe_round(
            candidate.get("final_score")
            or candidate.get("rerank_score")
            or candidate.get("score")
        ),
    }

    if include_excerpt:
        item["content_excerpt"] = _candidate_excerpt(candidate)

    return item


def _compact_results(results: list[dict], *, limit: int = 10, include_excerpt: bool = False) -> list[dict]:
    return [
        _compact_candidate(candidate, rank=i, include_excerpt=include_excerpt)
        for i, candidate in enumerate((results or [])[:limit], 1)
    ]


def _likely_failure_type(
    *,
    file_hit: bool,
    symbol_hit: bool,
    label_hit: bool,
    expected_files: list[str],
    expected_symbols: list[str],
    final_results: list[dict],
) -> str:
    if not final_results:
        return "empty_results"

    if expected_files and not file_hit and expected_symbols and not symbol_hit:
        return "expected_file_and_symbol_not_in_top5"

    if expected_files and not file_hit:
        return "expected_file_not_in_top5"

    if expected_symbols and not symbol_hit:
        return "expected_symbol_not_in_top5"

    if not label_hit:
        return "expected_labels_not_in_top5"

    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for CodeSeek.")
    parser.add_argument("--session-id", help="Session ID from database.")
    parser.add_argument("--golden", default="evals/golden/golden_queries.yaml", help="Path to golden queries YAML file.")
    parser.add_argument("--output", default="evals/reports/latest.json", help="Path to write the evaluation JSON report.")
    parser.add_argument("--k", type=int, default=5, help="K value for evaluation (default: 5).")
    parser.add_argument(
        "--debug-query-ids",
        default="",
        help="Comma-separated golden query IDs to include detailed layer debug output for, e.g. q004,q007",
    )

    args = parser.parse_args()

    debug_query_ids = {
        q.strip()
        for q in (args.debug_query_ids or "").split(",")
        if q.strip()
    }

    session_id = args.session_id
    collection_name = None
    repo_root = None

    # 1. Resolve collection & repo root from database
    if session_id:
        print(f"Loading session {session_id} from database...")
        with db_cursor() as (conn, cursor):
            cursor.execute(
                "SELECT collection, repo_root FROM repo_sessions WHERE id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                db_session = dict(row)
                collection_name = db_session["collection"]
                repo_root = db_session["repo_root"]
                
                os.environ["QDRANT_COLLECTION_NAME"] = collection_name
                os.environ["RETRIEVAL_REPO_ROOT"] = repo_root
                print(f"Bound to collection: {collection_name}")
                print(f"Bound to repo root:  {repo_root}")
            else:
                print(f"Error: Session ID {session_id} not found.")
                sys.exit(1)
    else:
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "repository_chunks")
        repo_root = os.getenv("RETRIEVAL_REPO_ROOT", str(Path.cwd()))
        print(f"Using default collection: {collection_name}")
        print(f"Using default repo root:  {repo_root}")

    # Load golden queries
    try:
        golden_queries = load_golden_queries(args.golden)
    except Exception as e:
        print(f"Error loading golden queries: {str(e)}")
        sys.exit(1)

    print(f"Loaded {len(golden_queries)} golden queries. Running evaluation...")

    traces = []
    total_queries = len(golden_queries)

    # Metrics accumulators
    intent_matches = 0
    reranker_intent_matches = 0
    file_hits_at_1 = 0
    file_hits_at_3 = 0
    file_hits_at_5 = 0
    symbol_hits_at_5 = 0
    label_hits_at_5 = 0
    protected_exact_hits_preserved = 0
    total_eligible_exact_preserved = 0
    exact_hit_regression_count = 0
    total_duplicates_count = 0
    total_chunks_count = 0
    wrong_top1_count = 0
    empty_result_count = 0
    total_latency_ms = 0
    reranker_intent_failures = []
    global_protected_hits_total = 0
    global_protected_hits_preserved = 0
    global_protected_hits_dropped = 0
    all_dropped_exact_hits = []
    known_failed_queries = []
    blocking_failures = []
    non_blocking_failures = []
    debug_queries = {}

    for idx, gq in enumerate(golden_queries, 1):
        query_id = gq["id"]
        raw_query = gq["query"]
        expected_intent = gq.get("expected_intent")
        expected_reranker_intent = gq.get("expected_reranker_intent")
        expected_files = gq.get("expected_files", [])
        expected_symbols = gq.get("expected_symbols", [])
        expected_labels = (
            gq.get("expected_labels_in_top1", []) + 
            gq.get("expected_labels_in_top3", []) + 
            gq.get("expected_labels_in_top5", [])
        )
        # De-duplicate expected labels
        expected_labels = sorted(list(set(expected_labels)))

        print(f"[{idx}/{total_queries}] Processing: {raw_query}")

        t0 = time.perf_counter()
        
        # 2. Run query processor
        # Run label classifier first
        label_profile = classify_query_intent(raw_query)
        label_intent = label_profile["intent"]

        q_info = process_query(raw_query)
        primary_intent = q_info["primary_intent"]
        from retrieval.query.query_intent import identify_followup_or_low_context
        is_followup, is_low_context = identify_followup_or_low_context(raw_query, conversation_state=None)
        
        # Map label classifier intent to reranker intent
        mapped_reranker_intent = map_label_intent_to_reranker_intent(
            label_intent,
            query=raw_query,
            is_followup=is_followup,
            is_low_context=is_low_context,
            extracted_entities=q_info.get("entities")
        )

        # 3. Run individual search layers for tracing
        try:
            dense_tuples = _dense_search(raw_query)
            dense_results = [format_point(p, s, src) for p, s, src in dense_tuples]
        except Exception:
            dense_results = []

        try:
            lexical_tuples = _lexical_search(raw_query)
            bm25_results = [format_point(p, s, src) for p, s, src in lexical_tuples]
        except Exception:
            bm25_results = []

        try:
            metadata_tuples = _metadata_search(raw_query, q_info["entities"])
            metadata_results = [format_point(p, s, src) for p, s, src in metadata_tuples]
        except Exception:
            metadata_results = []

        try:
            exact_tuples = _exact_entity_search(q_info["entities"])
            exact_results = [format_point(p, s, src) for p, s, src in exact_tuples]
        except Exception:
            exact_results = []

        try:
            dep_tuples = _dependency_search(q_info["entities"]) if mapped_reranker_intent == "DEPENDENCY" else []
            dependency_results = [format_point(p, s, src) for p, s, src in dep_tuples]
        except Exception:
            dependency_results = []

        # 4. Run merged final search
        t_search_start = time.perf_counter()
        final_tuples = search(q_info)
        t_search_end = time.perf_counter()
        latency_ms = int((t_search_end - t0) * 1000)
        total_latency_ms += latency_ms

        final_results = [format_final_point(p) for p in final_tuples]

        # Calculate metrics for this query
        q_intent_match = (label_intent == expected_intent) if expected_intent else True
        q_reranker_intent_match = (mapped_reranker_intent == expected_reranker_intent) if expected_reranker_intent else True

        if q_intent_match:
            intent_matches += 1
        if q_reranker_intent_match:
            reranker_intent_matches += 1
        else:
            if expected_reranker_intent:
                reason_if_detectable = "N/A"
                if expected_reranker_intent == "DEPENDENCY" and mapped_reranker_intent != "DEPENDENCY":
                    reason_if_detectable = "Dependency patterns not matched"
                elif expected_reranker_intent == "CONFIG" and mapped_reranker_intent != "CONFIG":
                    reason_if_detectable = "Config patterns / env keys not matched"
                elif expected_reranker_intent == "FOLLOWUP" and mapped_reranker_intent != "FOLLOWUP":
                    reason_if_detectable = "Not identified as followup"
                elif expected_reranker_intent == "LOW_CONTEXT" and mapped_reranker_intent != "LOW_CONTEXT":
                    reason_if_detectable = "Not identified as low context"
                
                reranker_intent_failures.append({
                    "query_id": query_id,
                    "query": raw_query,
                    "expected_intent": expected_intent or "N/A",
                    "actual_intent": label_intent,
                    "expected_reranker_intent": expected_reranker_intent,
                    "actual_reranker_intent": mapped_reranker_intent,
                    "boost_labels": q_info.get("boost_labels", []),
                    "reason_if_detectable": reason_if_detectable
                })

        q_file_hit1 = compute_file_hit(final_results, expected_files, 1)
        q_file_hit3 = compute_file_hit(final_results, expected_files, 3)
        q_file_hit5 = compute_file_hit(final_results, expected_files, 5)

        if q_file_hit1:
            file_hits_at_1 += 1
        if q_file_hit3:
            file_hits_at_3 += 1
        if q_file_hit5:
            file_hits_at_5 += 1

        q_symbol_hit5 = compute_symbol_hit(
            final_results,
            expected_symbols,
            k=5,
            reranker_intent=mapped_reranker_intent,
        )
        if q_symbol_hit5:
            symbol_hits_at_5 += 1

        q_label_hit5 = compute_label_hit(final_results, expected_labels, 5)
        if q_label_hit5:
            label_hits_at_5 += 1

        if not q_file_hit5 or not q_symbol_hit5 or not q_label_hit5:
            failure_entry = {
                "query_id": query_id,
                "query": raw_query,
                "category": gq.get("category"),
                "expected_intent": expected_intent,
                "actual_intent": label_intent,
                "expected_reranker_intent": expected_reranker_intent,
                "actual_reranker_intent": mapped_reranker_intent,
                "file_hit@5": q_file_hit5,
                "symbol_hit@5": q_symbol_hit5,
                "label_hit@5": q_label_hit5,
                "expected_files": expected_files,
                "expected_symbols": expected_symbols,
                "expected_labels": expected_labels,
                "actual_top5": _compact_results(final_results, limit=5, include_excerpt=False),
                "likely_failure_type": _likely_failure_type(
                    file_hit=q_file_hit5,
                    symbol_hit=q_symbol_hit5,
                    label_hit=q_label_hit5,
                    expected_files=expected_files,
                    expected_symbols=expected_symbols,
                    final_results=final_results,
                ),
            }

            known_failed_queries.append(failure_entry)
            
        if query_id in debug_query_ids:
            debug_queries[query_id] = {
                "query_id": query_id,
                "query": raw_query,
                "category": gq.get("category"),
                "expected_files": expected_files,
                "expected_symbols": expected_symbols,
                "expected_labels": expected_labels,
                "expected_intent": expected_intent,
                "actual_intent": label_intent,
                "expected_reranker_intent": expected_reranker_intent,
                "actual_reranker_intent": mapped_reranker_intent,
                "metrics": {
                    "file_hit@5": q_file_hit5,
                    "symbol_hit@5": q_symbol_hit5,
                    "label_hit@5": q_label_hit5,
                },
                "dense_results_top10": _compact_results(dense_results, limit=10, include_excerpt=True),
                "bm25_results_top10": _compact_results(bm25_results, limit=10, include_excerpt=True),
                "metadata_results_top10": _compact_results(metadata_results, limit=10, include_excerpt=True),
                "exact_results_top10": _compact_results(exact_results, limit=10, include_excerpt=True),
                "dependency_results_top10": _compact_results(dependency_results, limit=10, include_excerpt=True),
                "final_results_top10": _compact_results(final_results, limit=10, include_excerpt=True),
            }

        exact_audit = audit_exact_hit_preservation(final_results, exact_results, 5)
        q_exact_preserved = "N/A"
        if exact_audit["eligible"]:
            q_exact_preserved = (exact_audit["protected_hits_dropped"] == 0)
            total_eligible_exact_preserved += 1
            if q_exact_preserved:
                protected_exact_hits_preserved += 1
            else:
                exact_hit_regression_count += 1
            
            global_protected_hits_total += exact_audit["protected_hits_total"]
            global_protected_hits_preserved += exact_audit["protected_hits_preserved"]
            global_protected_hits_dropped += exact_audit["protected_hits_dropped"]
            
            for item in exact_audit["dropped_details"]:
                all_dropped_exact_hits.append({
                    "query_id": query_id,
                    "query": raw_query,
                    "chunk_id": item["chunk_id"],
                    "relative_path": item["relative_path"],
                    "exact_layer_rank": item["exact_layer_rank"]
                })

        q_dup_rate = compute_duplicate_context_rate(final_results)
        total_duplicates_count += int(q_dup_rate * len(final_results))
        total_chunks_count += len(final_results)

        if len(final_results) == 0:
            empty_result_count += 1

        # Check wrong top-1: if query has expected files/symbols, does top-1 match them?
        if expected_files or expected_symbols:
            has_top1_match = compute_file_hit(
                final_results,
                expected_files,
                1,
            ) or compute_symbol_hit(
                final_results,
                expected_symbols,
                k=1,
                reranker_intent=mapped_reranker_intent,
            )

            if not has_top1_match:
                wrong_top1_count += 1

        # Collect trace
        trace = {
            "query_id": query_id,
            "query": raw_query,
            "category": gq["category"],
            "label_classifier_intent": label_intent,
            "expected_intent": expected_intent,
            "reranker_intent": mapped_reranker_intent,
            "expected_reranker_intent": expected_reranker_intent,
            "is_followup": is_followup,
            "is_low_context": is_low_context,
            "extracted_entities": q_info.get("entities", {}),
            "boost_labels": q_info.get("boost_labels", []),
            "dense_results": dense_results[:5],
            "bm25_results": bm25_results[:5],
            "metadata_results": metadata_results[:5],
            "exact_results": exact_results[:5],
            "dependency_results": dependency_results[:5],
            "final_results": final_results[:5],
            "latency_ms": latency_ms,
            "metrics": {
                "file_hit@1": q_file_hit1,
                "file_hit@3": q_file_hit3,
                "file_hit@5": q_file_hit5,
                "symbol_hit@5": q_symbol_hit5,
                "label_hit@5": q_label_hit5,
                "exact_hit_preserved": q_exact_preserved,
                "duplicate_rate": round(q_dup_rate, 4),
            }
        }
        traces.append(trace)

    # Compute summary aggregates
    intent_accuracy = (intent_matches / total_queries) * 100.0 if total_queries else 0.0
    reranker_intent_accuracy = (reranker_intent_matches / total_queries) * 100.0 if total_queries else 0.0
    
    file_hit_at_1 = (file_hits_at_1 / total_queries) * 100.0 if total_queries else 0.0
    file_hit_at_3 = (file_hits_at_3 / total_queries) * 100.0 if total_queries else 0.0
    file_hit_at_5 = (file_hits_at_5 / total_queries) * 100.0 if total_queries else 0.0
    symbol_hit_at_5 = (symbol_hits_at_5 / total_queries) * 100.0 if total_queries else 0.0
    label_hit_at_5 = (label_hits_at_5 / total_queries) * 100.0 if total_queries else 0.0
    
    protected_exact_hit_preserved_rate = (
        (protected_exact_hits_preserved / total_eligible_exact_preserved * 100.0)
        if total_eligible_exact_preserved else 100.0
    )
    
    duplicate_context_rate = (total_duplicates_count / total_chunks_count * 100.0) if total_chunks_count else 0.0
    wrong_top1_rate = (wrong_top1_count / total_queries * 100.0) if total_queries else 0.0
    empty_result_rate = (empty_result_count / total_queries * 100.0) if total_queries else 0.0
    avg_latency = (total_latency_ms / total_queries) if total_queries else 0.0

    status = "PASS" if file_hit_at_5 >= 80.0 else "FAIL"

    pass_thresholds = {
        "file_hit@5": 0.80,
        "symbol_hit@5": "reported",
        "label_hit@5": "reported",
        "reranker_intent_accuracy": "reported",
        "duplicate_context_rate": "reported",
    }

    if status == "PASS":
        non_blocking_failures = known_failed_queries
        blocking_failures = []
    else:
        blocking_failures = known_failed_queries
        non_blocking_failures = []

    summary = {
        "intent_accuracy": round(intent_accuracy, 2),
        "reranker_intent_accuracy": round(reranker_intent_accuracy, 2),
        "file_hit@1": round(file_hit_at_1, 2),
        "file_hit@3": round(file_hit_at_3, 2),
        "file_hit@5": round(file_hit_at_5, 2),
        "symbol_hit@5": round(symbol_hit_at_5, 2),
        "label_hit@5": round(label_hit_at_5, 2),
        "protected_exact_hit_preserved@5": round(protected_exact_hit_preserved_rate, 2),
        "exact_hit_regression_count": exact_hit_regression_count,
        "protected_hits_total": global_protected_hits_total,
        "protected_hits_preserved": global_protected_hits_preserved,
        "protected_hits_dropped": global_protected_hits_dropped,
        "duplicate_context_rate": round(duplicate_context_rate, 2),
        "wrong_top1_rate": round(wrong_top1_rate, 2),
        "empty_result_rate": round(empty_result_rate, 2),
        "avg_latency_ms": round(avg_latency, 2),
    }

    report = {
        "status": status,
        "session_id": session_id or "N/A",
        "collection": collection_name,
        "repo_root": repo_root,
        "total_queries": total_queries,
        "summary": summary,
        "pass_thresholds": pass_thresholds,
        "known_failed_queries": known_failed_queries,
        "blocking_failures": blocking_failures,
        "non_blocking_failures": non_blocking_failures,
        "debug_queries": debug_queries,
        "reranker_intent_failures": reranker_intent_failures,
        "dropped_exact_hits": all_dropped_exact_hits,
        "query_traces": traces,
    }

    # Write report file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 40)
    print("         EVALUATION REPORT SUMMARY")
    print("=" * 40)
    print(f"Status:                      {report['status']}")
    print(f"Total Queries:               {report['total_queries']}")
    print(f"Avg Latency:                 {report['summary']['avg_latency_ms']} ms")
    print(f"Intent Accuracy:             {report['summary']['intent_accuracy']}%")
    print(f"Reranker Intent Accuracy:    {report['summary']['reranker_intent_accuracy']}%")
    print(f"File Hit@5:                  {report['summary']['file_hit@5']}%")
    print(f"Symbol Hit@5:                {report['summary']['symbol_hit@5']}%")
    print(f"Label Hit@5:                 {report['summary']['label_hit@5']}%")
    print(f"Protected Hits Preserved:    {report['summary']['protected_exact_hit_preserved@5']}%")
    print(f"Exact Hit Regressions:       {report['summary']['exact_hit_regression_count']}")
    print(f"Total Protected Hits:        {report['summary']['protected_hits_total']}")
    print(f"Protected Hits Preserved:    {report['summary']['protected_hits_preserved']}")
    print(f"Protected Hits Dropped:      {report['summary']['protected_hits_dropped']}")
    print(f"Duplicate Context Rate:      {report['summary']['duplicate_context_rate']}%")
    print(f"Wrong Top-1 Rate:            {report['summary']['wrong_top1_rate']}%")
    print(f"Empty Result Rate:           {report['summary']['empty_result_rate']}%")
    print("=" * 40)
    
    if reranker_intent_failures:
        print("\n" + "=" * 80)
        print("                 RERANKER INTENT MISMATCHES")
        print("=" * 80)
        print(f"{'ID':<6} | {'Query':<30} | {'Expected':<12} | {'Actual':<12} | {'Reason':<25}")
        print("-" * 80)
        for val in reranker_intent_failures:
            q_trunc = val['query']
            if len(q_trunc) > 28:
                q_trunc = q_trunc[:25] + "..."
            print(f"{val['query_id']:<6} | {q_trunc:<30} | {val['expected_reranker_intent']:<12} | {val['actual_reranker_intent']:<12} | {val['reason_if_detectable']:<25}")
        print("=" * 80)

    if all_dropped_exact_hits:
        print("\n" + "=" * 80)
        print("                 DROPPED PROTECTED EXACT HITS")
        print("=" * 80)
        print(f"{'QID':<5} | {'Chunk ID':<36} | {'Exact Layer Rank':<18} | {'File Path'}")
        print("-" * 80)
        for val in all_dropped_exact_hits:
            print(f"{val['query_id']:<5} | {val['chunk_id']:<36} | {val['exact_layer_rank']:<18} | {val['relative_path']}")
        print("=" * 80)

    print(f"\nFull report written to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
