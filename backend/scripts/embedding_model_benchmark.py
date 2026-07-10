"""Embedding model benchmark for CodeSeek retrieval.

Compares the current model (BAAI/bge-small-en-v1.5, 384-dim) against a
stronger alternative (BAAI/bge-base-en-v1.5, 768-dim) without requiring
a full re-ingestion.

Strategy
--------
Current model  — uses the live Qdrant collection (real end-to-end search).
Alternative    — encodes queries with the candidate model, pulls stored
                 chunk vectors from Qdrant, re-scores by cosine similarity,
                 and evaluates hit@k / MRR on the resulting ranking.
                 This is a retrieval-signal approximation: it isolates the
                 embedding-quality dimension from infrastructure differences.

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/embedding_model_benchmark.py

Optional flags:
    --eval-file <path>      single fixture (default: runs all suite fixtures)
    --k <int>               hit@k cutoff (default: 10)
    --alternative <model>   HuggingFace model id (default: BAAI/bge-base-en-v1.5)
    --output <path>         write JSON results to path
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import tracemalloc
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal bootstrap so the script can be run as a standalone with PYTHONPATH=.
# ---------------------------------------------------------------------------
try:
    from retrieval.config import (
        EMBEDDING_MODEL as CURRENT_MODEL,
        QUERY_PREFIX,
        get_collection_name,
    )
    from retrieval.search.searcher import search
    from retrieval.query.query_processor import process_query
except ImportError as exc:
    sys.exit(f"Import error – run with PYTHONPATH=.: {exc}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ALTERNATIVE = "BAAI/bge-base-en-v1.5"
ALTERNATIVE_DIM = 768
CURRENT_DIM = 384
SUITE_CONFIG_PATH = Path("evals/datasets/eval_suite_multi_repo.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _matches_file_or_symbol(item: dict, files: list[str], symbols: list[str]) -> bool:
    rp = _norm(item.get("relative_path", ""))
    sn = _norm(item.get("symbol_name", ""))
    return any(rp == _norm(f) for f in files) or any(sn == _norm(s) for s in symbols)


def _hit_at_k(candidates: list[dict], files: list[str], symbols: list[str], k: int) -> int:
    if not files and not symbols:
        return 1
    for item in candidates[:k]:
        if _matches_file_or_symbol(item, files, symbols):
            return 1
    return 0


def _mrr_at_k(candidates: list[dict], files: list[str], symbols: list[str], k: int) -> float:
    if not files and not symbols:
        return 1.0
    for i, item in enumerate(candidates[:k], start=1):
        if _matches_file_or_symbol(item, files, symbols):
            return 1.0 / i
    return 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def _get_qdrant_client():
    from retrieval.support.qdrant_config import create_qdrant_client
    return create_qdrant_client(timeout=10.0, check_compatibility=False)


def _pull_chunk_vectors(collection: str, max_points: int = 5000) -> list[dict]:
    """Scroll the collection and return payloads + stored vectors."""
    client = _get_qdrant_client()
    records: list[dict] = []
    offset = None
    while len(records) < max_points:
        try:
            response = client.scroll(
                collection_name=collection,
                limit=min(256, max_points - len(records)),
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            hits, offset = response
        except Exception as exc:
            print(f"[warn] Qdrant scroll error: {exc}", file=sys.stderr)
            break
        for hit in hits:
            payload = dict(hit.payload or {})
            vec = hit.vector
            if vec is None:
                continue
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            elif not isinstance(vec, list):
                vec = list(vec)
            payload["_vector"] = vec
            records.append(payload)
        if not offset:
            break
    return records


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _load_model(model_id: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id)


def _encode_queries(model, queries: list[str], prefix: str = "") -> tuple[list[list[float]], float]:
    """Encode queries and return (embeddings, mean_latency_ms)."""
    times: list[float] = []
    embeddings: list[list[float]] = []
    for q in queries:
        t0 = time.perf_counter()
        emb = model.encode(prefix + q).tolist()
        times.append((time.perf_counter() - t0) * 1000)
        embeddings.append(emb)
    mean_ms = sum(times) / len(times) if times else 0.0
    return embeddings, mean_ms


def _measure_model_memory_mb(model_id: str) -> float:
    """Peak memory delta (MB) when loading the model."""
    tracemalloc.start()
    _load_model(model_id)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

def _evaluate_alternative(
    query: str,
    query_emb: list[float],
    chunk_records: list[dict],
    expected_files: list[str],
    expected_symbols: list[str],
    k: int,
) -> tuple[int, float]:
    """Rank chunks by cosine similarity to query_emb, return hit@k and MRR@k."""
    scored = [
        (rec, _cosine(query_emb, rec["_vector"]))
        for rec in chunk_records
        if "_vector" in rec
    ]
    scored.sort(key=lambda x: -x[1])
    top_k = [item for item, _ in scored[:k]]
    return (
        _hit_at_k(top_k, expected_files, expected_symbols, k),
        _mrr_at_k(top_k, expected_files, expected_symbols, k),
    )


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _load_fixture_paths() -> list[Path]:
    """Return eval files that run against the live Qdrant collection (ingest_before_eval=false).

    These fixtures query the currently ingested collection without rebuilding it,
    making them safe to run during a benchmark that must not alter the collection.
    We also always include the exact-wording and flow-phase1 files since they
    are the richest signal sets for the CodeSeek collection.
    """
    if not SUITE_CONFIG_PATH.exists():
        return []
    cfg = json.loads(SUITE_CONFIG_PATH.read_text(encoding="utf-8"))
    datasets = cfg.get("datasets", [])
    paths: list[Path] = []
    for d in datasets:
        # Only use datasets that do NOT require re-ingestion — benchmark must
        # run against the already-indexed collection.
        if d.get("ingest_before_eval", True):
            continue
        p = Path(d.get("eval_file", ""))
        if p.exists():
            paths.append(p)
    # Fallback: if nothing is marked ingest_before_eval=false, include all.
    if not paths:
        for d in datasets:
            p = Path(d.get("eval_file", ""))
            if p.exists():
                paths.append(p)
    return paths


def _load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cases", [])


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    eval_paths: list[Path],
    alternative_model_id: str,
    k: int,
    output_path: Path | None,
) -> None:
    collection = get_collection_name()
    print(f"\nEmbedding Model Benchmark")
    print(f"=========================")
    print(f"Current model   : {CURRENT_MODEL}  (dim={CURRENT_DIM})")
    print(f"Alternative     : {alternative_model_id}  (dim={ALTERNATIVE_DIM})")
    print(f"Collection      : {collection}")
    print(f"Eval fixtures   : {[str(p) for p in eval_paths]}")
    print()

    # --- Step 1: pull chunk vectors from Qdrant (needed for alternative scoring)
    print("Pulling chunk vectors from Qdrant (for alternative model scoring)…")
    t_pull = time.perf_counter()
    chunk_records = _pull_chunk_vectors(collection)
    pull_ms = (time.perf_counter() - t_pull) * 1000
    print(f"  Fetched {len(chunk_records)} chunks in {pull_ms:.0f} ms")

    # Filter to records that have a vector matching CURRENT_DIM
    valid_chunks = [r for r in chunk_records if len(r.get("_vector", [])) == CURRENT_DIM]
    print(f"  {len(valid_chunks)} chunks have {CURRENT_DIM}-dim vectors (current model)\n")

    # --- Step 2: load alternative model + measure memory
    print(f"Loading alternative model: {alternative_model_id} …")
    tracemalloc.start()
    alt_model = _load_model(alternative_model_id)
    _, alt_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    alt_memory_mb = alt_peak / (1024 * 1024)
    print(f"  Peak memory delta: {alt_memory_mb:.1f} MB\n")

    print(f"Loading current model: {CURRENT_MODEL} …")
    tracemalloc.start()
    cur_model = _load_model(CURRENT_MODEL)
    _, cur_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cur_memory_mb = cur_peak / (1024 * 1024)
    print(f"  Peak memory delta: {cur_memory_mb:.1f} MB\n")

    # --- Step 3: collect all queries across fixtures
    all_cases: list[tuple[Path, dict]] = []
    for path in eval_paths:
        for case in _load_cases(path):
            all_cases.append((path, case))

    if not all_cases:
        print("No eval cases found. Exiting.")
        return

    queries = [case["query"] for _, case in all_cases]

    # --- Step 4: detect dimension mismatch
    alt_dim_actual = len(alt_model.encode("test").tolist())
    dim_mismatch = alt_dim_actual != CURRENT_DIM
    if dim_mismatch:
        print(
            f"[!] Dimension mismatch detected:\n"
            f"    Stored collection vectors: {CURRENT_DIM}-dim (current model)\n"
            f"    Alternative model output : {alt_dim_actual}-dim\n"
            f"    Cosine similarity ranking is NOT valid across different dimensions.\n"
            f"    To evaluate the alternative model's recall, the collection must be\n"
            f"    fully re-ingested with the new model. This benchmark can only measure\n"
            f"    encoding latency and memory cost for the alternative model.\n"
        )

    # --- Step 5: encode with both models, measure latency
    print(f"Encoding {len(queries)} queries with current model…")
    cur_embs, cur_encode_ms = _encode_queries(cur_model, queries, prefix=QUERY_PREFIX)
    print(f"  Mean encode latency: {cur_encode_ms:.1f} ms/query")

    print(f"Encoding {len(queries)} queries with alternative model…")
    alt_embs, alt_encode_ms = _encode_queries(alt_model, queries, prefix=QUERY_PREFIX)
    print(f"  Mean encode latency: {alt_encode_ms:.1f} ms/query\n")

    # --- Step 5: evaluate per case
    fixture_results: dict[str, dict] = {}
    per_case_results: list[dict] = []

    for idx, (path, case) in enumerate(all_cases):
        files = case.get("expected_files", [])
        symbols = case.get("expected_symbols", [])

        # Current model — use real Qdrant search
        query_info = process_query(case["query"])
        t0 = time.perf_counter()
        candidates = search(query_info)
        search_ms = (time.perf_counter() - t0) * 1000
        cur_hit = _hit_at_k(candidates, files, symbols, k)
        cur_mrr = _mrr_at_k(candidates, files, symbols, k)

        # Alternative model — only valid if dimensions match the stored collection
        if dim_mismatch:
            alt_hit, alt_mrr = None, None
        else:
            alt_hit, alt_mrr = _evaluate_alternative(
                case["query"], alt_embs[idx], valid_chunks, files, symbols, k
            )

        per_case_results.append({
            "fixture": str(path),
            "id": case.get("id", ""),
            "query": case["query"],
            "current_hit": cur_hit,
            "current_mrr": cur_mrr,
            "current_search_ms": round(search_ms, 1),
            "alternative_hit": alt_hit,
            "alternative_mrr": alt_mrr,
            "dim_mismatch": dim_mismatch,
        })

        fname = path.stem
        if fname not in fixture_results:
            fixture_results[fname] = {
                "cur_hits": [], "cur_mrrs": [],
                "alt_hits": [], "alt_mrrs": [],
            }
        fixture_results[fname]["cur_hits"].append(cur_hit)
        fixture_results[fname]["cur_mrrs"].append(cur_mrr)
        if not dim_mismatch:
            fixture_results[fname]["alt_hits"].append(alt_hit)
            fixture_results[fname]["alt_mrrs"].append(alt_mrr)

    # --- Step 6: print results
    print("Per-fixture Results (Current Model — real Qdrant search)")
    print("----------------------------------------------------------")
    if dim_mismatch:
        print(f"  NOTE: Alternative model recall comparison SKIPPED (dim mismatch: {CURRENT_DIM} vs {alt_dim_actual}).")
        print(f"  To compare recall, re-ingest the collection with {alternative_model_id}.\n")
    header = f"{'Fixture':<35} {'cur hit@k':>10} {'cur mrr':>9}"
    if not dim_mismatch:
        header += f" {'alt hit@k':>10} {'alt mrr':>9} {'delta hit':>10}"
    print(header)
    print("-" * len(header))

    all_cur_hits: list[int] = []
    all_cur_mrrs: list[float] = []
    all_alt_hits: list[int] = []
    all_alt_mrrs: list[float] = []

    for fname, res in sorted(fixture_results.items()):
        ch = sum(res["cur_hits"]) / len(res["cur_hits"])
        cm = sum(res["cur_mrrs"]) / len(res["cur_mrrs"])
        row = f"{fname:<35} {ch:>10.3f} {cm:>9.3f}"
        if not dim_mismatch and res["alt_hits"]:
            ah = sum(res["alt_hits"]) / len(res["alt_hits"])
            am = sum(res["alt_mrrs"]) / len(res["alt_mrrs"])
            delta = ah - ch
            row += f" {ah:>10.3f} {am:>9.3f} {delta:>+10.3f}"
        print(row)
        all_cur_hits.extend(res["cur_hits"])
        all_cur_mrrs.extend(res["cur_mrrs"])
        if not dim_mismatch:
            all_alt_hits.extend(res.get("alt_hits", []))
            all_alt_mrrs.extend(res.get("alt_mrrs", []))

    print()
    total = len(all_cur_hits)
    overall_cur_hit = sum(all_cur_hits) / total
    overall_cur_mrr = sum(all_cur_mrrs) / total
    row = f"{'Overall':<35} {overall_cur_hit:>10.3f} {overall_cur_mrr:>9.3f}"
    if not dim_mismatch and all_alt_hits:
        overall_alt_hit = sum(all_alt_hits) / len(all_alt_hits)
        overall_alt_mrr = sum(all_alt_mrrs) / len(all_alt_mrrs)
        row += f" {overall_alt_hit:>10.3f} {overall_alt_mrr:>9.3f} {overall_alt_hit - overall_cur_hit:>+10.3f}"
    print(row)
    print()

    print("Model Characteristics")
    print("---------------------")
    print(f"  Current  ({CURRENT_MODEL})")
    print(f"    Dimensions   : {CURRENT_DIM}")
    print(f"    Encode ms/q  : {cur_encode_ms:.1f}")
    print(f"    Memory peak  : {cur_memory_mb:.1f} MB")
    print()
    print(f"  Alternative ({alternative_model_id})")
    print(f"    Dimensions   : {alt_dim_actual}")
    print(f"    Encode ms/q  : {alt_encode_ms:.1f}")
    print(f"    Memory peak  : {alt_memory_mb:.1f} MB")
    print(f"    Memory delta : {alt_memory_mb - cur_memory_mb:+.1f} MB")
    if dim_mismatch:
        print(f"    Re-ingestion : REQUIRED (dim change {CURRENT_DIM} → {alt_dim_actual})")
    print()

    if dim_mismatch:
        verdict = "KEEP CURRENT — re-ingestion required to compare recall"
        print(f"Decision gate: {verdict}")
        print(f"  Alternative requires full collection re-ingestion ({CURRENT_DIM}→{alt_dim_actual} dims).")
        print(f"  Operational cost: re-embed all {len(valid_chunks)} chunks + recreate Qdrant collection.")
        print(f"  Encoding cost delta: {alt_encode_ms - cur_encode_ms:+.1f} ms/query encode time.")
        print(f"  Memory cost delta  : {alt_memory_mb - cur_memory_mb:+.1f} MB peak load.")
        print(f"  MTEB score delta   : +1.38 (bge-base 63.55 vs bge-small 62.17 on MTEB benchmark).")
        print(f"  Recommendation     : Not justified unless hit@k improves >0.02 after re-ingestion eval.")
    elif all_alt_hits:
        overall_alt_hit = sum(all_alt_hits) / len(all_alt_hits)
        if overall_alt_hit <= overall_cur_hit + 0.02:
            verdict = "KEEP CURRENT"
            print(f"Decision gate: {verdict}")
            print(f"  Gain is within noise margin (≤0.02) — switching is not justified by recall alone.")
        else:
            verdict = "CONSIDER ALTERNATIVE"
            print(f"Decision gate: {verdict}")
            print(f"  Alternative shows +{(overall_alt_hit - overall_cur_hit):.3f} hit@{k} gain — evaluate operational cost before switching.")
    else:
        verdict = "KEEP CURRENT"
        print(f"Decision gate: {verdict}")
    print()

    # --- Step 7: write JSON output
    summary = {
        "current_model": CURRENT_MODEL,
        "current_dim": CURRENT_DIM,
        "alternative_model": alternative_model_id,
        "alternative_dim": alt_dim_actual,
        "dim_mismatch": dim_mismatch,
        "dim_mismatch_note": (
            f"Alternative requires full re-ingestion ({CURRENT_DIM}→{alt_dim_actual} dims). "
            "Recall comparison not valid without re-ingesting the collection."
        ) if dim_mismatch else None,
        "k": k,
        "total_cases": total,
        "current_baseline": {
            "hit_at_k": round(overall_cur_hit, 4),
            "mrr_at_k": round(overall_cur_mrr, 4),
        },
        "alternative_recall": None if dim_mismatch else {
            "hit_at_k": round(sum(all_alt_hits) / len(all_alt_hits), 4),
            "mrr_at_k": round(sum(all_alt_mrrs) / len(all_alt_mrrs), 4),
            "delta_hit_at_k": round(sum(all_alt_hits) / len(all_alt_hits) - overall_cur_hit, 4),
        },
        "mteb_scores": {
            "current": 62.17,
            "alternative": 63.55,
            "delta": 1.38,
        },
        "model_characteristics": {
            "current": {
                "encode_ms_per_query": round(cur_encode_ms, 2),
                "memory_peak_mb": round(cur_memory_mb, 1),
            },
            "alternative": {
                "encode_ms_per_query": round(alt_encode_ms, 2),
                "memory_peak_mb": round(alt_memory_mb, 1),
                "re_ingestion_required": dim_mismatch,
                "chunks_to_re_embed": len(valid_chunks) if dim_mismatch else 0,
            },
        },
        "verdict": verdict,
        "per_fixture": {
            fname: {
                "current_hit_at_k": round(sum(res["cur_hits"]) / len(res["cur_hits"]), 4),
                "current_mrr_at_k": round(sum(res["cur_mrrs"]) / len(res["cur_mrrs"]), 4),
                "alternative_hit_at_k": round(sum(res["alt_hits"]) / len(res["alt_hits"]), 4) if res["alt_hits"] else None,
                "alternative_mrr_at_k": round(sum(res["alt_mrrs"]) / len(res["alt_mrrs"]), 4) if res["alt_mrrs"] else None,
            }
            for fname, res in fixture_results.items()
        },
        "per_case": per_case_results,
    }

    if output_path:
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Results written to: {output_path}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding models for CodeSeek retrieval.")
    parser.add_argument("--eval-file", help="Single eval fixture JSON (default: all suite fixtures)")
    parser.add_argument("--k", type=int, default=10, help="hit@k cutoff (default: 10)")
    parser.add_argument("--alternative", default=DEFAULT_ALTERNATIVE, help="Alternative HuggingFace model ID")
    parser.add_argument("--output", help="Write JSON results to this path")
    args = parser.parse_args()

    if args.eval_file:
        eval_paths = [Path(args.eval_file)]
    else:
        eval_paths = _load_fixture_paths()
        if not eval_paths:
            sys.exit("No eval fixtures found. Pass --eval-file or ensure eval_suite_multi_repo.json exists.")

    output_path = Path(args.output) if args.output else None
    run_benchmark(eval_paths, args.alternative, args.k, output_path)


if __name__ == "__main__":
    main()
