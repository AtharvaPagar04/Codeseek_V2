"""Run retrieval eval across multiple datasets/repos and aggregate metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from retrieval.support.isolation import expected_collection_name

HIT_RE = re.compile(r"^hit@\d+:\s*([0-9.]+)\s*$")
MRR_RE = re.compile(r"^mrr@\d+:\s*([0-9.]+)\s*$")
COV_RE = re.compile(r"^citation_coverage:\s*([0-9.]+)\s*$")
EXPECTED_FILE_RE = re.compile(r"^expected_file_score:\s*([0-9.]+)\s*$")
EXPECTED_SYMBOL_RE = re.compile(r"^expected_symbol_score:\s*([0-9.]+)\s*$")
EXPECTED_FRAMEWORK_RE = re.compile(r"^expected_framework_score:\s*([0-9.]+)\s*$")
EXPECTED_DEPENDENCY_RE = re.compile(r"^expected_dependency_score:\s*([0-9.]+)\s*$")
EXPECTED_NO_ANSWER_RE = re.compile(r"^expected_no_answer_score:\s*([0-9.]+)\s*$")
EXPECTED_RESPONSE_MODE_RE = re.compile(r"^expected_response_mode_score:\s*([0-9.]+)\s*$")
EXPECTED_ANSWER_TERM_RE = re.compile(r"^expected_answer_term_score:\s*([0-9.]+)\s*$")
TOPIC_SHIFT_RE = re.compile(r"^topic_shift_accuracy:\s*([0-9.]+)\s*$")
FOLLOWUP_PRECISION_RE = re.compile(r"^followup_precision:\s*([0-9.]+)\s*$")
FOLLOWUP_RECALL_RE = re.compile(r"^followup_recall:\s*([0-9.]+)\s*$")
FOLLOWUP_DECISION_RE = re.compile(r"^followup_decision_score:\s*([0-9.]+)\s*$")
HISTORY_INJECTION_SCORE_RE = re.compile(r"^history_injection_score:\s*([0-9.]+)\s*$")
PREVIOUS_CANDIDATE_SCORE_RE = re.compile(r"^previous_candidate_injection_score:\s*([0-9.]+)\s*$")
QUERY_REWRITE_SCORE_RE = re.compile(r"^query_rewrite_score:\s*([0-9.]+)\s*$")
LOW_CONFIDENCE_SCORE_RE = re.compile(r"^low_confidence_refusal_score:\s*([0-9.]+)\s*$")
ANSWER_RELEVANCE_RE = re.compile(r"^answer_relevance_score:\s*([0-9.]+)\s*$")
SOURCE_FAITHFULNESS_RE = re.compile(r"^source_faithfulness_score:\s*([0-9.]+)\s*$")
WRONG_TOPIC_RE = re.compile(r"^wrong_topic_answer_score:\s*([0-9.]+)\s*$")
RETRIEVAL_CONFIDENCE_RE = re.compile(r"^retrieval_confidence_score:\s*([0-9.]+)\s*$")
HISTORY_INJECTION_RATE_RE = re.compile(r"^history_injection_rate:\s*([0-9.]+)\s*$")
PREVIOUS_CANDIDATE_RATE_RE = re.compile(r"^previous_candidate_injection_rate:\s*([0-9.]+)\s*$")
QUERY_REWRITE_RATE_RE = re.compile(r"^query_rewrite_rate:\s*([0-9.]+)\s*$")
LOW_CONFIDENCE_RATE_RE = re.compile(r"^low_confidence_refusal_rate:\s*([0-9.]+)\s*$")
LATENCY_P50_RE = re.compile(r"^latency_p50_ms:\s*(\d+)\s*$")
LATENCY_P95_RE = re.compile(r"^latency_p95_ms:\s*(\d+)\s*$")
RETRIEVAL_ONLY_P50_RE = re.compile(r"^retrieval_only_latency_p50_ms:\s*(\d+)\s*$")
RETRIEVAL_ONLY_P95_RE = re.compile(r"^retrieval_only_latency_p95_ms:\s*(\d+)\s*$")
DETERMINISTIC_P50_RE = re.compile(r"^deterministic_latency_p50_ms:\s*(\d+)\s*$")
DETERMINISTIC_P95_RE = re.compile(r"^deterministic_latency_p95_ms:\s*(\d+)\s*$")
LLM_BACKEND_P50_RE = re.compile(r"^llm_backend_latency_p50_ms:\s*(\d+)\s*$")
LLM_BACKEND_P95_RE = re.compile(r"^llm_backend_latency_p95_ms:\s*(\d+)\s*$")
LLM_PROVIDER_P50_RE = re.compile(r"^llm_provider_latency_p50_ms:\s*(\d+)\s*$")
LLM_PROVIDER_P95_RE = re.compile(r"^llm_provider_latency_p95_ms:\s*(\d+)\s*$")
LLM_TOTAL_P50_RE = re.compile(r"^llm_total_latency_p50_ms:\s*(\d+)\s*$")
LLM_TOTAL_P95_RE = re.compile(r"^llm_total_latency_p95_ms:\s*(\d+)\s*$")
CASES_RE = re.compile(r"^Cases:\s*(\d+)\s*$")


def _parse_eval_output(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if match := HIT_RE.match(line):
            metrics["hit"] = float(match.group(1))
        elif match := MRR_RE.match(line):
            metrics["mrr"] = float(match.group(1))
        elif match := COV_RE.match(line):
            metrics["cov"] = float(match.group(1))
        elif match := EXPECTED_FILE_RE.match(line):
            metrics["expected_file"] = float(match.group(1))
        elif match := EXPECTED_SYMBOL_RE.match(line):
            metrics["expected_symbol"] = float(match.group(1))
        elif match := EXPECTED_FRAMEWORK_RE.match(line):
            metrics["expected_framework"] = float(match.group(1))
        elif match := EXPECTED_DEPENDENCY_RE.match(line):
            metrics["expected_dependency"] = float(match.group(1))
        elif match := EXPECTED_NO_ANSWER_RE.match(line):
            metrics["expected_no_answer"] = float(match.group(1))
        elif match := EXPECTED_RESPONSE_MODE_RE.match(line):
            metrics["expected_response_mode"] = float(match.group(1))
        elif match := EXPECTED_ANSWER_TERM_RE.match(line):
            metrics["expected_answer_term"] = float(match.group(1))
        elif match := TOPIC_SHIFT_RE.match(line):
            metrics["topic_shift_accuracy"] = float(match.group(1))
        elif match := FOLLOWUP_PRECISION_RE.match(line):
            metrics["followup_precision"] = float(match.group(1))
        elif match := FOLLOWUP_RECALL_RE.match(line):
            metrics["followup_recall"] = float(match.group(1))
        elif match := FOLLOWUP_DECISION_RE.match(line):
            metrics["followup_decision"] = float(match.group(1))
        elif match := HISTORY_INJECTION_SCORE_RE.match(line):
            metrics["history_injection_score"] = float(match.group(1))
        elif match := PREVIOUS_CANDIDATE_SCORE_RE.match(line):
            metrics["previous_candidate_injection_score"] = float(match.group(1))
        elif match := QUERY_REWRITE_SCORE_RE.match(line):
            metrics["query_rewrite_score"] = float(match.group(1))
        elif match := LOW_CONFIDENCE_SCORE_RE.match(line):
            metrics["low_confidence_refusal_score"] = float(match.group(1))
        elif match := ANSWER_RELEVANCE_RE.match(line):
            metrics["answer_relevance_score"] = float(match.group(1))
        elif match := SOURCE_FAITHFULNESS_RE.match(line):
            metrics["source_faithfulness_score"] = float(match.group(1))
        elif match := WRONG_TOPIC_RE.match(line):
            metrics["wrong_topic_answer_score"] = float(match.group(1))
        elif match := RETRIEVAL_CONFIDENCE_RE.match(line):
            metrics["retrieval_confidence_score"] = float(match.group(1))
        elif match := HISTORY_INJECTION_RATE_RE.match(line):
            metrics["history_injection_rate"] = float(match.group(1))
        elif match := PREVIOUS_CANDIDATE_RATE_RE.match(line):
            metrics["previous_candidate_injection_rate"] = float(match.group(1))
        elif match := QUERY_REWRITE_RATE_RE.match(line):
            metrics["query_rewrite_rate"] = float(match.group(1))
        elif match := LOW_CONFIDENCE_RATE_RE.match(line):
            metrics["low_confidence_refusal_rate"] = float(match.group(1))
        elif match := LATENCY_P50_RE.match(line):
            metrics["latency_p50_ms"] = float(match.group(1))
        elif match := LATENCY_P95_RE.match(line):
            metrics["latency_p95_ms"] = float(match.group(1))
        elif match := RETRIEVAL_ONLY_P50_RE.match(line):
            metrics["retrieval_only_latency_p50_ms"] = float(match.group(1))
        elif match := RETRIEVAL_ONLY_P95_RE.match(line):
            metrics["retrieval_only_latency_p95_ms"] = float(match.group(1))
        elif match := DETERMINISTIC_P50_RE.match(line):
            metrics["deterministic_latency_p50_ms"] = float(match.group(1))
        elif match := DETERMINISTIC_P95_RE.match(line):
            metrics["deterministic_latency_p95_ms"] = float(match.group(1))
        elif match := LLM_BACKEND_P50_RE.match(line):
            metrics["llm_backend_latency_p50_ms"] = float(match.group(1))
        elif match := LLM_BACKEND_P95_RE.match(line):
            metrics["llm_backend_latency_p95_ms"] = float(match.group(1))
        elif match := LLM_PROVIDER_P50_RE.match(line):
            metrics["llm_provider_latency_p50_ms"] = float(match.group(1))
        elif match := LLM_PROVIDER_P95_RE.match(line):
            metrics["llm_provider_latency_p95_ms"] = float(match.group(1))
        elif match := LLM_TOTAL_P50_RE.match(line):
            metrics["llm_total_latency_p50_ms"] = float(match.group(1))
        elif match := LLM_TOTAL_P95_RE.match(line):
            metrics["llm_total_latency_p95_ms"] = float(match.group(1))
        elif match := CASES_RE.match(line):
            metrics["cases"] = float(match.group(1))
    required = {
        "hit",
        "mrr",
        "cov",
        "expected_file",
        "expected_symbol",
        "expected_framework",
        "expected_dependency",
        "expected_no_answer",
        "expected_response_mode",
        "expected_answer_term",
        "topic_shift_accuracy",
        "followup_precision",
        "followup_recall",
        "followup_decision",
        "history_injection_score",
        "previous_candidate_injection_score",
        "query_rewrite_score",
        "low_confidence_refusal_score",
        "answer_relevance_score",
        "source_faithfulness_score",
        "wrong_topic_answer_score",
        "retrieval_confidence_score",
        "history_injection_rate",
        "previous_candidate_injection_rate",
        "query_rewrite_rate",
        "low_confidence_refusal_rate",
        "latency_p50_ms",
        "latency_p95_ms",
        "retrieval_only_latency_p50_ms",
        "retrieval_only_latency_p95_ms",
        "deterministic_latency_p50_ms",
        "deterministic_latency_p95_ms",
        "llm_backend_latency_p50_ms",
        "llm_backend_latency_p95_ms",
        "llm_provider_latency_p50_ms",
        "llm_provider_latency_p95_ms",
        "llm_total_latency_p50_ms",
        "llm_total_latency_p95_ms",
        "cases",
    }
    missing = required - set(metrics)
    if missing:
        raise RuntimeError(f"Missing metrics in eval output: {sorted(missing)}")
    return metrics


def _resolve_dataset_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def _run_dataset(project_root: Path, dataset: dict) -> dict:
    eval_file = str(_resolve_dataset_path(project_root, dataset["eval_file"]).resolve())
    repo_root = str(_resolve_dataset_path(project_root, dataset["repo_root"]).resolve())
    k = int(dataset.get("k", 10))

    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root)
    env["QDRANT_COLLECTION_NAME"] = str(
        dataset.get("collection_name", expected_collection_name(repo_root))
    )
    env["RETRIEVAL_REPO_ROOT"] = repo_root

    if dataset.get("ingest_before_eval", False):
        if dataset.get("recreate_collection", False):
            env["QDRANT_RECREATE_COLLECTION"] = "1"
            env["INGESTION_ENABLE_INCREMENTAL_FILE_SKIP"] = "0"
        ingest_cmd = [
            str(project_root / ".venv" / "bin" / "python"),
            "-m",
            "rag_ingestion.main",
            repo_root,
        ]
        ingest_proc = subprocess.run(
            ingest_cmd,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if ingest_proc.returncode != 0:
            raise RuntimeError(
                f"Ingestion failed for dataset {dataset['id']}\n"
                f"stdout:\n{ingest_proc.stdout}\n\nstderr:\n{ingest_proc.stderr}"
            )

    cmd = [
        str(project_root / ".venv" / "bin" / "python"),
        str(project_root / "scripts" / "retrieval_eval.py"),
        "--eval-file",
        eval_file,
        "--k",
        str(k),
    ]
    provider = str(dataset.get("provider", "")).strip()
    api_key_env = str(dataset.get("api_key_env", "")).strip()
    model = str(dataset.get("model", "")).strip()
    if provider and api_key_env:
        cmd.extend(["--provider", provider, "--api-key-env", api_key_env])
    if model:
        cmd.extend(["--model", model])
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Eval failed for dataset {dataset['id']}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    metrics = _parse_eval_output(proc.stdout)
    metrics["id"] = dataset["id"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-repo retrieval eval suite.")
    parser.add_argument(
        "--suite-file",
        default="evals/datasets/eval_suite_multi_repo.json",
        help="Suite JSON with datasets list",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write suite metrics as JSON.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    suite = json.loads((project_root / args.suite_file).read_text(encoding="utf-8"))
    datasets = suite.get("datasets", [])
    if not datasets:
        raise SystemExit("No datasets found in suite file.")

    results = []
    for dataset in datasets:
        metrics = _run_dataset(project_root, dataset)
        results.append(metrics)
        print(
            f"{dataset['id']}: cases={int(metrics['cases'])} "
            f"hit@k={metrics['hit']:.3f} mrr@k={metrics['mrr']:.3f} "
            f"citation_coverage={metrics['cov']:.3f} "
            f"expected_file={metrics['expected_file']:.3f} "
            f"expected_symbol={metrics['expected_symbol']:.3f} "
            f"expected_framework={metrics['expected_framework']:.3f} "
            f"expected_dependency={metrics['expected_dependency']:.3f} "
            f"expected_no_answer={metrics['expected_no_answer']:.3f} "
            f"expected_response_mode={metrics['expected_response_mode']:.3f} "
            f"expected_answer_term={metrics['expected_answer_term']:.3f} "
            f"latency_p50_ms={int(metrics['latency_p50_ms'])} "
            f"latency_p95_ms={int(metrics['latency_p95_ms'])} "
            f"retrieval_only_p50_ms={int(metrics['retrieval_only_latency_p50_ms'])} "
            f"deterministic_p50_ms={int(metrics['deterministic_latency_p50_ms'])} "
            f"llm_total_p50_ms={int(metrics['llm_total_latency_p50_ms'])}"
        )

    total_cases = sum(r["cases"] for r in results)
    agg_hit = sum(r["hit"] * r["cases"] for r in results) / total_cases
    agg_mrr = sum(r["mrr"] * r["cases"] for r in results) / total_cases
    agg_cov = sum(r["cov"] * r["cases"] for r in results) / total_cases
    agg_file = sum(r["expected_file"] * r["cases"] for r in results) / total_cases
    agg_symbol = sum(r["expected_symbol"] * r["cases"] for r in results) / total_cases
    agg_framework = sum(r["expected_framework"] * r["cases"] for r in results) / total_cases
    agg_dependency = sum(r["expected_dependency"] * r["cases"] for r in results) / total_cases
    agg_no_answer = sum(r["expected_no_answer"] * r["cases"] for r in results) / total_cases
    agg_response_mode = sum(r["expected_response_mode"] * r["cases"] for r in results) / total_cases
    agg_answer_term = sum(r["expected_answer_term"] * r["cases"] for r in results) / total_cases
    agg_topic_shift = sum(r["topic_shift_accuracy"] * r["cases"] for r in results) / total_cases
    agg_followup_precision = sum(r["followup_precision"] * r["cases"] for r in results) / total_cases
    agg_followup_recall = sum(r["followup_recall"] * r["cases"] for r in results) / total_cases
    agg_followup_decision = sum(r["followup_decision"] * r["cases"] for r in results) / total_cases
    agg_history_injection_score = sum(r["history_injection_score"] * r["cases"] for r in results) / total_cases
    agg_previous_candidate_score = sum(r["previous_candidate_injection_score"] * r["cases"] for r in results) / total_cases
    agg_query_rewrite_score = sum(r["query_rewrite_score"] * r["cases"] for r in results) / total_cases
    agg_low_confidence_score = sum(r["low_confidence_refusal_score"] * r["cases"] for r in results) / total_cases
    agg_answer_relevance = sum(r["answer_relevance_score"] * r["cases"] for r in results) / total_cases
    agg_source_faithfulness = sum(r["source_faithfulness_score"] * r["cases"] for r in results) / total_cases
    agg_wrong_topic = sum(r["wrong_topic_answer_score"] * r["cases"] for r in results) / total_cases
    agg_retrieval_confidence = sum(r["retrieval_confidence_score"] * r["cases"] for r in results) / total_cases
    agg_history_injection_rate = sum(r["history_injection_rate"] * r["cases"] for r in results) / total_cases
    agg_previous_candidate_rate = sum(r["previous_candidate_injection_rate"] * r["cases"] for r in results) / total_cases
    agg_query_rewrite_rate = sum(r["query_rewrite_rate"] * r["cases"] for r in results) / total_cases
    agg_low_confidence_rate = sum(r["low_confidence_refusal_rate"] * r["cases"] for r in results) / total_cases
    max_latency_p50 = max(r["latency_p50_ms"] for r in results)
    max_latency_p95 = max(r["latency_p95_ms"] for r in results)
    max_retrieval_only_p50 = max(r["retrieval_only_latency_p50_ms"] for r in results)
    max_retrieval_only_p95 = max(r["retrieval_only_latency_p95_ms"] for r in results)
    max_deterministic_p50 = max(r["deterministic_latency_p50_ms"] for r in results)
    max_deterministic_p95 = max(r["deterministic_latency_p95_ms"] for r in results)
    max_llm_backend_p50 = max(r["llm_backend_latency_p50_ms"] for r in results)
    max_llm_backend_p95 = max(r["llm_backend_latency_p95_ms"] for r in results)
    max_llm_provider_p50 = max(r["llm_provider_latency_p50_ms"] for r in results)
    max_llm_provider_p95 = max(r["llm_provider_latency_p95_ms"] for r in results)
    max_llm_total_p50 = max(r["llm_total_latency_p50_ms"] for r in results)
    max_llm_total_p95 = max(r["llm_total_latency_p95_ms"] for r in results)

    print("\nAggregate")
    print("=========")
    print(f"datasets: {len(results)}")
    print(f"cases: {int(total_cases)}")
    print(f"weighted_hit@k: {agg_hit:.3f}")
    print(f"weighted_mrr@k: {agg_mrr:.3f}")
    print(f"weighted_citation_coverage: {agg_cov:.3f}")
    print(f"weighted_expected_file_score: {agg_file:.3f}")
    print(f"weighted_expected_symbol_score: {agg_symbol:.3f}")
    print(f"weighted_expected_framework_score: {agg_framework:.3f}")
    print(f"weighted_expected_dependency_score: {agg_dependency:.3f}")
    print(f"weighted_expected_no_answer_score: {agg_no_answer:.3f}")
    print(f"weighted_expected_response_mode_score: {agg_response_mode:.3f}")
    print(f"weighted_expected_answer_term_score: {agg_answer_term:.3f}")
    print(f"weighted_topic_shift_accuracy: {agg_topic_shift:.3f}")
    print(f"weighted_followup_precision: {agg_followup_precision:.3f}")
    print(f"weighted_followup_recall: {agg_followup_recall:.3f}")
    print(f"weighted_followup_decision_score: {agg_followup_decision:.3f}")
    print(f"weighted_history_injection_score: {agg_history_injection_score:.3f}")
    print(f"weighted_previous_candidate_injection_score: {agg_previous_candidate_score:.3f}")
    print(f"weighted_query_rewrite_score: {agg_query_rewrite_score:.3f}")
    print(f"weighted_low_confidence_refusal_score: {agg_low_confidence_score:.3f}")
    print(f"weighted_answer_relevance_score: {agg_answer_relevance:.3f}")
    print(f"weighted_source_faithfulness_score: {agg_source_faithfulness:.3f}")
    print(f"weighted_wrong_topic_answer_score: {agg_wrong_topic:.3f}")
    print(f"weighted_retrieval_confidence_score: {agg_retrieval_confidence:.3f}")
    print(f"weighted_history_injection_rate: {agg_history_injection_rate:.3f}")
    print(f"weighted_previous_candidate_injection_rate: {agg_previous_candidate_rate:.3f}")
    print(f"weighted_query_rewrite_rate: {agg_query_rewrite_rate:.3f}")
    print(f"weighted_low_confidence_refusal_rate: {agg_low_confidence_rate:.3f}")
    print(f"max_latency_p50_ms: {int(max_latency_p50)}")
    print(f"max_latency_p95_ms: {int(max_latency_p95)}")
    print(f"max_retrieval_only_latency_p50_ms: {int(max_retrieval_only_p50)}")
    print(f"max_retrieval_only_latency_p95_ms: {int(max_retrieval_only_p95)}")
    print(f"max_deterministic_latency_p50_ms: {int(max_deterministic_p50)}")
    print(f"max_deterministic_latency_p95_ms: {int(max_deterministic_p95)}")
    print(f"max_llm_backend_latency_p50_ms: {int(max_llm_backend_p50)}")
    print(f"max_llm_backend_latency_p95_ms: {int(max_llm_backend_p95)}")
    print(f"max_llm_provider_latency_p50_ms: {int(max_llm_provider_p50)}")
    print(f"max_llm_provider_latency_p95_ms: {int(max_llm_provider_p95)}")
    print(f"max_llm_total_latency_p50_ms: {int(max_llm_total_p50)}")
    print(f"max_llm_total_latency_p95_ms: {int(max_llm_total_p95)}")

    if args.json_out:
        payload = {
            "datasets": results,
            "aggregate": {
                "datasets": len(results),
                "cases": int(total_cases),
                "weighted_hit@k": round(agg_hit, 6),
                "weighted_mrr@k": round(agg_mrr, 6),
                "weighted_citation_coverage": round(agg_cov, 6),
                "weighted_expected_file_score": round(agg_file, 6),
                "weighted_expected_symbol_score": round(agg_symbol, 6),
                "weighted_expected_framework_score": round(agg_framework, 6),
                "weighted_expected_dependency_score": round(agg_dependency, 6),
                "weighted_expected_no_answer_score": round(agg_no_answer, 6),
                "weighted_expected_response_mode_score": round(agg_response_mode, 6),
                "weighted_expected_answer_term_score": round(agg_answer_term, 6),
                "weighted_topic_shift_accuracy": round(agg_topic_shift, 6),
                "weighted_followup_precision": round(agg_followup_precision, 6),
                "weighted_followup_recall": round(agg_followup_recall, 6),
                "weighted_followup_decision_score": round(agg_followup_decision, 6),
                "weighted_history_injection_score": round(agg_history_injection_score, 6),
                "weighted_previous_candidate_injection_score": round(agg_previous_candidate_score, 6),
                "weighted_query_rewrite_score": round(agg_query_rewrite_score, 6),
                "weighted_low_confidence_refusal_score": round(agg_low_confidence_score, 6),
                "weighted_answer_relevance_score": round(agg_answer_relevance, 6),
                "weighted_source_faithfulness_score": round(agg_source_faithfulness, 6),
                "weighted_wrong_topic_answer_score": round(agg_wrong_topic, 6),
                "weighted_retrieval_confidence_score": round(agg_retrieval_confidence, 6),
                "weighted_history_injection_rate": round(agg_history_injection_rate, 6),
                "weighted_previous_candidate_injection_rate": round(agg_previous_candidate_rate, 6),
                "weighted_query_rewrite_rate": round(agg_query_rewrite_rate, 6),
                "weighted_low_confidence_refusal_rate": round(agg_low_confidence_rate, 6),
                "max_latency_p50_ms": int(max_latency_p50),
                "max_latency_p95_ms": int(max_latency_p95),
                "max_retrieval_only_latency_p50_ms": int(max_retrieval_only_p50),
                "max_retrieval_only_latency_p95_ms": int(max_retrieval_only_p95),
                "max_deterministic_latency_p50_ms": int(max_deterministic_p50),
                "max_deterministic_latency_p95_ms": int(max_deterministic_p95),
                "max_llm_backend_latency_p50_ms": int(max_llm_backend_p50),
                "max_llm_backend_latency_p95_ms": int(max_llm_backend_p95),
                "max_llm_provider_latency_p50_ms": int(max_llm_provider_p50),
                "max_llm_provider_latency_p95_ms": int(max_llm_provider_p95),
                "max_llm_total_latency_p50_ms": int(max_llm_total_p50),
                "max_llm_total_latency_p95_ms": int(max_llm_total_p95),
            },
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
