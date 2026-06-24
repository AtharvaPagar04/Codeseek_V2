"""Validate per-dataset and aggregate retrieval suite thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fmt(name: str, got: float, expected: float) -> str:
    return f"{name} {got:.3f} < {expected:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate retrieval suite thresholds.")
    parser.add_argument("--metrics-json", required=True, help="Path to suite JSON output")
    parser.add_argument("--thresholds-json", required=True, help="Path to thresholds JSON")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    thresholds = json.loads(Path(args.thresholds_json).read_text(encoding="utf-8"))

    by_id = {d["id"]: d for d in metrics.get("datasets", [])}
    failures: list[str] = []

    for dataset in thresholds.get("datasets", []):
        ds_id = dataset["id"]
        got = by_id.get(ds_id)
        if got is None:
            failures.append(f"missing dataset metrics: {ds_id}")
            continue
        min_hit = float(dataset.get("min_hit", 0.0))
        min_mrr = float(dataset.get("min_mrr", 0.0))
        min_cov = float(dataset.get("min_citation_coverage", 0.0))
        min_response_mode = float(dataset.get("min_expected_response_mode", 0.0))
        min_answer_term = float(dataset.get("min_expected_answer_term", 0.0))
        max_latency_p50 = dataset.get("max_latency_p50_ms")
        max_latency_p95 = dataset.get("max_latency_p95_ms")
        baseline_cov = dataset.get("baseline_citation_coverage")
        max_drop = float(dataset.get("max_citation_drop", 1.0))

        if got["hit"] < min_hit:
            failures.append(f"[{ds_id}] " + _fmt("hit@k", got["hit"], min_hit))
        if got["mrr"] < min_mrr:
            failures.append(f"[{ds_id}] " + _fmt("mrr@k", got["mrr"], min_mrr))
        if got["cov"] < min_cov:
            failures.append(
                f"[{ds_id}] " + _fmt("citation_coverage", got["cov"], min_cov)
            )
        if got.get("expected_response_mode", 1.0) < min_response_mode:
            failures.append(
                f"[{ds_id}] " + _fmt(
                    "expected_response_mode",
                    float(got.get("expected_response_mode", 0.0)),
                    min_response_mode,
                )
            )
        if got.get("expected_answer_term", 1.0) < min_answer_term:
            failures.append(
                f"[{ds_id}] " + _fmt(
                    "expected_answer_term",
                    float(got.get("expected_answer_term", 0.0)),
                    min_answer_term,
                )
            )
        if max_latency_p50 is not None and float(got.get("latency_p50_ms", 0.0)) > float(max_latency_p50):
            failures.append(
                f"[{ds_id}] latency_p50_ms {float(got.get('latency_p50_ms', 0.0)):.0f} > {float(max_latency_p50):.0f}"
            )
        if max_latency_p95 is not None and float(got.get("latency_p95_ms", 0.0)) > float(max_latency_p95):
            failures.append(
                f"[{ds_id}] latency_p95_ms {float(got.get('latency_p95_ms', 0.0)):.0f} > {float(max_latency_p95):.0f}"
            )
        if baseline_cov is not None:
            baseline_cov = float(baseline_cov)
            drop = baseline_cov - float(got["cov"])
            if drop > max_drop:
                failures.append(
                    f"[{ds_id}] citation_coverage regression: "
                    f"baseline={baseline_cov:.3f}, current={got['cov']:.3f}, drop={drop:.3f} > {max_drop:.3f}"
                )

    agg = metrics.get("aggregate", {})
    agg_thresholds = thresholds.get("aggregate", {})
    for key, threshold_key in (
        ("weighted_hit@k", "min_weighted_hit"),
        ("weighted_mrr@k", "min_weighted_mrr"),
        ("weighted_citation_coverage", "min_weighted_citation_coverage"),
    ):
        expected = agg_thresholds.get(threshold_key)
        if expected is None:
            continue
        got_value = float(agg.get(key, 0.0))
        expected_value = float(expected)
        if got_value < expected_value:
            failures.append(f"[aggregate] " + _fmt(key, got_value, expected_value))

    print("Suite metrics check summary:")
    for ds in metrics.get("datasets", []):
        print(
            f"- {ds['id']}: hit={ds['hit']:.3f} mrr={ds['mrr']:.3f} "
            f"citation={ds['cov']:.3f} response_mode={ds.get('expected_response_mode', 1.0):.3f} "
            f"answer_terms={ds.get('expected_answer_term', 1.0):.3f} "
            f"latency_p50_ms={int(ds.get('latency_p50_ms', 0))} "
            f"latency_p95_ms={int(ds.get('latency_p95_ms', 0))}"
        )
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
            print(f"::error::{failure}")
        sys.exit(1)

    print("All suite thresholds passed.")


if __name__ == "__main__":
    main()
