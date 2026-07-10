"""Fail CI when retrieval eval metrics drop below thresholds."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PATTERNS = {
    "hit@k": re.compile(r"^hit@\d+:\s*([0-9.]+)\s*$"),
    "mrr@k": re.compile(r"^mrr@\d+:\s*([0-9.]+)\s*$"),
    "citation_coverage": re.compile(r"^citation_coverage:\s*([0-9.]+)\s*$"),
}


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        for key, pattern in PATTERNS.items():
            match = pattern.match(line)
            if match:
                metrics[key] = float(match.group(1))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate retrieval eval metric thresholds.")
    parser.add_argument("--eval-output", required=True, help="Path to saved eval output text file")
    parser.add_argument("--min-hit", type=float, default=0.98)
    parser.add_argument("--min-mrr", type=float, default=0.85)
    parser.add_argument("--min-citation", type=float, default=0.88)
    args = parser.parse_args()

    text = Path(args.eval_output).read_text(encoding="utf-8")
    metrics = parse_metrics(text)

    missing = [k for k in ("hit@k", "mrr@k", "citation_coverage") if k not in metrics]
    if missing:
        print(f"Missing metrics in eval output: {', '.join(missing)}")
        sys.exit(1)

    failures = []
    if metrics["hit@k"] < args.min_hit:
        failures.append(f"hit@k {metrics['hit@k']:.3f} < {args.min_hit:.3f}")
    if metrics["mrr@k"] < args.min_mrr:
        failures.append(f"mrr@k {metrics['mrr@k']:.3f} < {args.min_mrr:.3f}")
    if metrics["citation_coverage"] < args.min_citation:
        failures.append(
            f"citation_coverage {metrics['citation_coverage']:.3f} < {args.min_citation:.3f}"
        )

    print(
        "Parsed metrics:",
        f"hit@k={metrics['hit@k']:.3f}",
        f"mrr@k={metrics['mrr@k']:.3f}",
        f"citation_coverage={metrics['citation_coverage']:.3f}",
    )
    if failures:
        print("Threshold check failed:")
        for failure in failures:
            print(f" - {failure}")
        sys.exit(1)

    print("Threshold check passed.")


if __name__ == "__main__":
    main()
