#!/usr/bin/env python3
"""Lexical Retrieval Layer Benchmark.

Measures:
  - Index build time and memory cost for the BM25 in-process index.
  - Per-query latency: dense-only vs dense+lexical.
  - hit@k and MRR for each retrieval mode across four query families:
      SYMBOL/EXACT  — exact symbol / env-key / config-key lookups
      DEPENDENCY    — "what calls X", "depends on Y"
      OVERVIEW      — broad "what is this project about" style
      SEMANTIC      — natural-language explanatory queries

Usage:
    PYTHONPATH=. .venv/bin/python scripts/lexical_layer_benchmark.py
    PYTHONPATH=. .venv/bin/python scripts/lexical_layer_benchmark.py \\
        --output backend/evals/reports/lexical_layer_benchmark_results.json

Decision gate printed at the end:
    ENABLE LEXICAL if mean latency delta < 150 ms AND hit@10 improves >= 0.02
    on any query family, otherwise KEEP DISABLED.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Eval fixtures — inline so the script is self-contained.
# Each case: query, expected_files (list), expected_symbols (list), family.
# ---------------------------------------------------------------------------
EVAL_CASES: list[dict[str, Any]] = [
    # --- SYMBOL / EXACT ---
    {
        "id": "lex-sym-001",
        "query": "Where is CODESEEK_DATABASE_URL documented or used?",
        "expected_files": ["retrieval/config.py", ".env.example"],
        "expected_symbols": ["CODESEEK_DATABASE_URL"],
        "family": "SYMBOL",
    },
    {
        "id": "lex-sym-002",
        "query": "Where is RETRIEVAL_ENABLE_LEXICAL configured?",
        "expected_files": ["retrieval/config.py"],
        "expected_symbols": ["RETRIEVAL_ENABLE_LEXICAL", "ENABLE_LEXICAL_RETRIEVAL"],
        "family": "SYMBOL",
    },
    {
        "id": "lex-sym-003",
        "query": "Where is BAAI/bge-small-en-v1.5 configured?",
        "expected_files": ["retrieval/config.py"],
        "expected_symbols": ["EMBEDDING_MODEL"],
        "family": "SYMBOL",
    },
    {
        "id": "lex-sym-004",
        "query": "Where is process_query defined?",
        "expected_files": ["retrieval/query/query_processor.py"],
        "expected_symbols": ["process_query"],
        "family": "SYMBOL",
    },
    {
        "id": "lex-sym-005",
        "query": "Where is the submission-key endpoint implemented?",
        "expected_files": ["retrieval/api_service.py"],
        "expected_symbols": [],
        "family": "SYMBOL",
    },
    # --- DEPENDENCY ---
    {
        "id": "lex-dep-001",
        "query": "Where is the qdrant-client dependency declared?",
        "expected_files": ["pyproject.toml"],
        "expected_symbols": ["qdrant-client"],
        "family": "DEPENDENCY",
    },
    {
        "id": "lex-dep-002",
        "query": "Where is FastAPI declared as a dependency?",
        "expected_files": ["pyproject.toml"],
        "expected_symbols": ["fastapi"],
        "family": "DEPENDENCY",
    },
    {
        "id": "lex-dep-003",
        "query": "Which code invalidates the lexical index after ingestion?",
        "expected_files": ["retrieval/search/searcher.py", "rag_ingestion/main.py"],
        "expected_symbols": [],
        "family": "DEPENDENCY",
    },
    {
        "id": "lex-dep-004",
        "query": "what calls run_query?",
        "expected_files": ["retrieval/main.py"],
        "expected_symbols": ["run_query"],
        "family": "DEPENDENCY",
    },
    # --- OVERVIEW ---
    {
        "id": "lex-ov-001",
        "query": "what is this project about",
        "expected_files": ["__repo_summary__.md", "README.md"],
        "expected_symbols": [],
        "family": "OVERVIEW",
    },
    {
        "id": "lex-ov-002",
        "query": "what tech stack is used",
        "expected_files": ["__repo_summary__.md"],
        "expected_symbols": [],
        "family": "OVERVIEW",
    },
    {
        "id": "lex-ov-003",
        "query": "architecture overview",
        "expected_files": ["__repo_summary__.md", "README.md"],
        "expected_symbols": [],
        "family": "OVERVIEW",
    },
    {
        "id": "lex-ov-004",
        "query": "what framework does this use",
        "expected_files": ["__repo_summary__.md"],
        "expected_symbols": [],
        "family": "OVERVIEW",
    },
    # --- SEMANTIC ---
    {
        "id": "lex-sem-001",
        "query": "how does the auth session flow work",
        "expected_files": ["retrieval/main.py", "retrieval/api_service.py"],
        "expected_symbols": ["auth_github", "create_auth_session"],
        "family": "SEMANTIC",
    },
    {
        "id": "lex-sem-002",
        "query": "explain how the retrieval pipeline assembles context",
        "expected_files": ["retrieval/main.py", "retrieval/generation/code_answers.py"],
        "expected_symbols": [],
        "family": "SEMANTIC",
    },
    {
        "id": "lex-sem-003",
        "query": "how does the ingestion pipeline work",
        "expected_files": ["rag_ingestion/main.py"],
        "expected_symbols": ["run_pipeline"],
        "family": "SEMANTIC",
    },
]

K = 10  # hit@k threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    family: str
    query: str
    hit_at_k_dense: int
    mrr_dense: float
    hit_at_k_lexical: int
    mrr_lexical: float
    latency_dense_ms: float
    latency_lexical_ms: float
    top_files_dense: list[str]
    top_files_lexical: list[str]


@dataclass
class FamilyMetrics:
    family: str
    n: int
    hit_at_k_dense: float
    mrr_dense: float
    hit_at_k_lexical: float
    mrr_lexical: float
    latency_dense_ms: float
    latency_lexical_ms: float
    latency_delta_ms: float
    hit_at_k_delta: float


def _hit_at_k(results: list[dict], expected_files: list[str], k: int = K) -> int:
    """Return 1 if any expected file appears in the top-k results."""
    if not expected_files:
        return 1  # no expected files → always a hit
    top_files = {str(r.get("relative_path", "")).lower() for r in results[:k]}
    for ef in expected_files:
        if any(ef.lower() in f for f in top_files):
            return 1
    return 0


def _mrr(results: list[dict], expected_files: list[str], k: int = K) -> float:
    if not expected_files:
        return 1.0
    for i, r in enumerate(results[:k]):
        rp = str(r.get("relative_path", "")).lower()
        for ef in expected_files:
            if ef.lower() in rp:
                return 1.0 / (i + 1)
    return 0.0


def _top_files(results: list[dict], n: int = 5) -> list[str]:
    return [str(r.get("relative_path", "?")) for r in results[:n]]


def _run_search(query_info: dict, enable_lexical: bool) -> tuple[list[dict], float]:
    """Run search with or without lexical layer, return (results, latency_ms)."""
    import retrieval.search.searcher as searcher_mod

    original = searcher_mod.ENABLE_LEXICAL_RETRIEVAL
    searcher_mod.ENABLE_LEXICAL_RETRIEVAL = enable_lexical

    # Warm up (first call builds index if lexical enabled)
    try:
        t0 = time.perf_counter()
        results = searcher_mod.search(query_info)
        latency_ms = (time.perf_counter() - t0) * 1000
    finally:
        searcher_mod.ENABLE_LEXICAL_RETRIEVAL = original

    return results, latency_ms


def _measure_index_build(collection: str) -> tuple[float, float]:
    """Return (build_time_ms, memory_delta_mb) for building the BM25 index."""
    import retrieval.search.searcher as searcher_mod

    # Clear cached index first
    searcher_mod._lexical_indexes.pop(collection, None)

    tracemalloc.start()
    t0 = time.perf_counter()
    searcher_mod._get_lexical_index(collection)
    build_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_mb = peak / (1024 * 1024)
    return build_ms, mem_mb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_path: str | None = None) -> None:
    from retrieval.query.query_processor import process_query
    from retrieval.config import get_collection_name
    import retrieval.search.searcher as searcher_mod

    collection = get_collection_name()

    print("Lexical Retrieval Layer Benchmark")
    print("==================================")
    print(f"Collection : {collection}")
    print(f"Cases      : {len(EVAL_CASES)} across 4 query families")
    print(f"hit@k      : k={K}\n")

    # --- 1. Index build cost
    print("Building BM25 in-process index (cold start)…")
    build_ms, build_mem_mb = _measure_index_build(collection)
    index = searcher_mod._get_lexical_index(collection)
    doc_count = len(index.documents)
    print(f"  Documents indexed : {doc_count}")
    print(f"  Build time        : {build_ms:.1f} ms")
    print(f"  Peak memory       : {build_mem_mb:.1f} MB\n")

    # Warm up lexical index so per-query latency doesn't include build time
    searcher_mod._get_lexical_index(collection)  # ensure cached

    # --- 2. Per-query benchmark
    print("Running per-query benchmark…")
    case_results: list[CaseResult] = []

    for case in EVAL_CASES:
        qi = process_query(case["query"])
        expected = case["expected_files"]

        # Dense only — run 3 times, take median
        dense_latencies = []
        dense_results = []
        for _ in range(3):
            r, lat = _run_search(qi, enable_lexical=False)
            dense_latencies.append(lat)
            dense_results = r
        dense_lat = sorted(dense_latencies)[1]  # median

        # Dense + lexical — run 3 times, take median
        lex_latencies = []
        lex_results = []
        for _ in range(3):
            r, lat = _run_search(qi, enable_lexical=True)
            lex_latencies.append(lat)
            lex_results = r
        lex_lat = sorted(lex_latencies)[1]

        cr = CaseResult(
            case_id=case["id"],
            family=case["family"],
            query=case["query"],
            hit_at_k_dense=_hit_at_k(dense_results, expected),
            mrr_dense=_mrr(dense_results, expected),
            hit_at_k_lexical=_hit_at_k(lex_results, expected),
            mrr_lexical=_mrr(lex_results, expected),
            latency_dense_ms=round(dense_lat, 1),
            latency_lexical_ms=round(lex_lat, 1),
            top_files_dense=_top_files(dense_results),
            top_files_lexical=_top_files(lex_results),
        )
        case_results.append(cr)

        delta_hit = cr.hit_at_k_lexical - cr.hit_at_k_dense
        delta_lat = cr.latency_lexical_ms - cr.latency_dense_ms
        mark = "+" if delta_hit > 0 else (" " if delta_hit == 0 else "-")
        print(
            f"  [{cr.family:<10}] hit={mark}{delta_hit:+d}  "
            f"lat_dense={cr.latency_dense_ms:>6.1f}ms  "
            f"lat_lex={cr.latency_lexical_ms:>6.1f}ms (+{delta_lat:.1f}ms)  "
            f"| {cr.query[:55]}"
        )

    # --- 3. Family aggregates
    families = ["SYMBOL", "DEPENDENCY", "OVERVIEW", "SEMANTIC"]
    family_metrics: list[FamilyMetrics] = []

    print("\nPer-Family Aggregates")
    print("---------------------")
    for fam in families:
        subset = [cr for cr in case_results if cr.family == fam]
        if not subset:
            continue
        n = len(subset)
        fm = FamilyMetrics(
            family=fam,
            n=n,
            hit_at_k_dense=sum(c.hit_at_k_dense for c in subset) / n,
            mrr_dense=sum(c.mrr_dense for c in subset) / n,
            hit_at_k_lexical=sum(c.hit_at_k_lexical for c in subset) / n,
            mrr_lexical=sum(c.mrr_lexical for c in subset) / n,
            latency_dense_ms=sum(c.latency_dense_ms for c in subset) / n,
            latency_lexical_ms=sum(c.latency_lexical_ms for c in subset) / n,
            latency_delta_ms=round(
                sum(c.latency_lexical_ms - c.latency_dense_ms for c in subset) / n, 1
            ),
            hit_at_k_delta=round(
                sum(c.hit_at_k_lexical - c.hit_at_k_dense for c in subset) / n, 3
            ),
        )
        family_metrics.append(fm)

        delta_sign = "▲" if fm.hit_at_k_delta > 0 else ("▼" if fm.hit_at_k_delta < 0 else "=")
        print(
            f"  {fam:<12}  n={n}  "
            f"hit@{K}_dense={fm.hit_at_k_dense:.3f}  hit@{K}_lex={fm.hit_at_k_lexical:.3f}  "
            f"Δhit={delta_sign}{abs(fm.hit_at_k_delta):.3f}  "
            f"lat_delta=+{fm.latency_delta_ms:.1f}ms"
        )

    # --- 4. Overall
    all_dense_hit = sum(c.hit_at_k_dense for c in case_results) / len(case_results)
    all_lex_hit = sum(c.hit_at_k_lexical for c in case_results) / len(case_results)
    all_dense_mrr = sum(c.mrr_dense for c in case_results) / len(case_results)
    all_lex_mrr = sum(c.mrr_lexical for c in case_results) / len(case_results)
    mean_latency_delta = sum(c.latency_lexical_ms - c.latency_dense_ms for c in case_results) / len(case_results)

    print(f"\n  Overall            n={len(case_results)}  "
          f"hit@{K}_dense={all_dense_hit:.3f}  hit@{K}_lex={all_lex_hit:.3f}  "
          f"Δhit={all_lex_hit - all_dense_hit:+.3f}  "
          f"mean_lat_delta=+{mean_latency_delta:.1f}ms")

    # --- 5. Tuning plan + decision gate
    LATENCY_THRESHOLD_MS = 150.0
    RECALL_IMPROVEMENT_THRESHOLD = 0.02

    improving_families = [
        fm.family for fm in family_metrics if fm.hit_at_k_delta >= RECALL_IMPROVEMENT_THRESHOLD
    ]
    latency_acceptable = mean_latency_delta < LATENCY_THRESHOLD_MS
    recall_improved = bool(improving_families)
    should_enable = latency_acceptable and recall_improved

    verdict = "ENABLE LEXICAL" if should_enable else "KEEP DISABLED"
    print(f"\nDecision Gate (latency < {LATENCY_THRESHOLD_MS}ms AND Δhit >= {RECALL_IMPROVEMENT_THRESHOLD})")
    print(f"  Latency delta     : {mean_latency_delta:.1f} ms  → {'✓ acceptable' if latency_acceptable else '✗ too slow'}")
    print(f"  Families improved : {improving_families or 'none'}  → {'✓' if recall_improved else '✗ no gain >= 0.02'}")
    print(f"  Verdict           : {verdict}")

    # --- 6. Tuning plan
    print("\nTuning Plan (lexical vs dense weighting by query family)")
    print("--------------------------------------------------------")
    tuning_plan = _build_tuning_plan(family_metrics, build_ms, build_mem_mb)
    for line in tuning_plan["notes"]:
        print(f"  {line}")

    # --- 7. Write JSON output
    results_payload = {
        "index": {
            "collection": collection,
            "doc_count": doc_count,
            "build_time_ms": round(build_ms, 1),
            "peak_memory_mb": round(build_mem_mb, 1),
        },
        "overall": {
            "n": len(case_results),
            "hit_at_k_dense": round(all_dense_hit, 4),
            "hit_at_k_lexical": round(all_lex_hit, 4),
            "hit_at_k_delta": round(all_lex_hit - all_dense_hit, 4),
            "mrr_dense": round(all_dense_mrr, 4),
            "mrr_lexical": round(all_lex_mrr, 4),
            "mean_latency_delta_ms": round(mean_latency_delta, 1),
        },
        "by_family": [asdict(fm) for fm in family_metrics],
        "cases": [asdict(cr) for cr in case_results],
        "decision": {
            "verdict": verdict,
            "latency_delta_ms": round(mean_latency_delta, 1),
            "latency_threshold_ms": LATENCY_THRESHOLD_MS,
            "latency_acceptable": latency_acceptable,
            "improving_families": improving_families,
            "recall_improved": recall_improved,
        },
        "tuning_plan": tuning_plan,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results_payload, fh, indent=2)
        print(f"\nResults written to: {output_path}")

    return results_payload


def _build_tuning_plan(
    family_metrics: list[FamilyMetrics],
    build_ms: float,
    build_mem_mb: float,
) -> dict:
    """Generate a query-family tuning plan based on measured deltas."""
    notes = []
    recommendations: dict[str, str] = {}

    family_map = {fm.family: fm for fm in family_metrics}

    for fam, fm in family_map.items():
        if fm.hit_at_k_delta > 0.05:
            rec = "WEIGHT_LEXICAL_HIGH — lexical clearly helps; increase RRF weight for this family"
        elif fm.hit_at_k_delta > 0.0:
            rec = "WEIGHT_LEXICAL_MEDIUM — marginal gain; keep lexical active, monitor MRR"
        elif fm.hit_at_k_delta == 0.0:
            rec = "WEIGHT_EQUAL — no measurable difference; equal weighting safe"
        else:
            rec = "WEIGHT_DENSE_ONLY — lexical hurts recall for this family; consider disabling per-intent"
        recommendations[fam] = rec

    notes.append(f"Index build cost : {build_ms:.1f} ms, {build_mem_mb:.1f} MB peak — paid once per worker startup.")
    notes.append(f"Index is cached  : subsequent queries pay only BM25 scoring cost (in-process, no network).")
    notes.append("")
    for fam, rec in recommendations.items():
        fm = family_map[fam]
        notes.append(
            f"{fam:<12}: Δhit={fm.hit_at_k_delta:+.3f}  Δlat=+{fm.latency_delta_ms:.1f}ms  → {rec}"
        )
    notes.append("")
    notes.append("Defer per-family weight tuning until weighted-fusion eval baselines exist.")
    notes.append("Gate: only tune weights after running retrieval_eval.py with lexical enabled")
    notes.append("      across all eval fixtures and recording baseline hit@k per family.")

    return {"notes": notes, "recommendations": recommendations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lexical retrieval layer benchmark")
    parser.add_argument(
        "--output",
        default="backend/evals/reports/lexical_layer_benchmark_results.json",
        help="Path to write JSON results (default: backend/evals/reports/lexical_layer_benchmark_results.json)",
    )
    args = parser.parse_args()
    main(output_path=args.output)
