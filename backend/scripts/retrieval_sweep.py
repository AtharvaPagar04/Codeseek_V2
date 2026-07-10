"""Grid-search retrieval knobs against eval set metrics."""

from __future__ import annotations

import argparse
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path


HIT_RE = re.compile(r"^hit@(\d+):\s*([0-9.]+)\s*$")
COV_RE = re.compile(r"^citation_coverage:\s*([0-9.]+)\s*$")


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _run_eval(
    project_root: Path,
    eval_file: Path,
    k: int,
    top_k_dense: int,
    top_k_after_merge: int,
    call_expansion_limit: int,
    repo_root: str,
) -> tuple[float, float]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root)
    env["RETRIEVAL_TOP_K_DENSE"] = str(top_k_dense)
    env["RETRIEVAL_TOP_K_AFTER_MERGE"] = str(top_k_after_merge)
    env["RETRIEVAL_CALL_EXPANSION_LIMIT"] = str(call_expansion_limit)
    env["RETRIEVAL_REPO_ROOT"] = repo_root

    cmd = [
        str(project_root / ".venv" / "bin" / "python"),
        str(project_root / "scripts" / "retrieval_eval.py"),
        "--eval-file",
        str(eval_file),
        "--k",
        str(k),
    ]
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
            f"eval failed for config d={top_k_dense},m={top_k_after_merge},c={call_expansion_limit}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    hit = None
    cov = None
    for line in proc.stdout.splitlines():
        hit_match = HIT_RE.match(line.strip())
        if hit_match:
            hit = float(hit_match.group(2))
            continue
        cov_match = COV_RE.match(line.strip())
        if cov_match:
            cov = float(cov_match.group(1))
            continue
    if hit is None or cov is None:
        raise RuntimeError(f"Could not parse metrics from eval output:\n{proc.stdout}")
    return hit, cov


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep retrieval knobs and rank configs.")
    parser.add_argument("--eval-file", required=True, help="Path to eval JSON file")
    parser.add_argument("--repo-root", required=True, help="Repository root for file assembly")
    parser.add_argument("--k", type=int, default=10, help="K for hit@k")
    parser.add_argument(
        "--top-k-dense",
        default="10,15,20",
        help="Comma-separated values for RETRIEVAL_TOP_K_DENSE",
    )
    parser.add_argument(
        "--top-k-after-merge",
        default="8,10,12",
        help="Comma-separated values for RETRIEVAL_TOP_K_AFTER_MERGE",
    )
    parser.add_argument(
        "--call-expansion-limit",
        default="3,5,8",
        help="Comma-separated values for RETRIEVAL_CALL_EXPANSION_LIMIT",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    eval_file = Path(args.eval_file).resolve()
    repo_root = str(Path(args.repo_root).resolve())

    dense_values = _parse_csv_ints(args.top_k_dense)
    merge_values = _parse_csv_ints(args.top_k_after_merge)
    call_values = _parse_csv_ints(args.call_expansion_limit)

    rows: list[dict] = []
    for dense, merge, call_limit in itertools.product(
        dense_values, merge_values, call_values
    ):
        hit, cov = _run_eval(
            project_root=project_root,
            eval_file=eval_file,
            k=args.k,
            top_k_dense=dense,
            top_k_after_merge=merge,
            call_expansion_limit=call_limit,
            repo_root=repo_root,
        )
        rows.append(
            {
                "top_k_dense": dense,
                "top_k_after_merge": merge,
                "call_expansion_limit": call_limit,
                "hit_at_k": hit,
                "citation_coverage": cov,
            }
        )
        print(
            f"tested d={dense} m={merge} c={call_limit} -> "
            f"hit@{args.k}={hit:.3f} citation_coverage={cov:.3f}"
        )

    rows.sort(
        key=lambda r: (
            -r["hit_at_k"],
            -r["citation_coverage"],
            r["top_k_dense"],
            r["top_k_after_merge"],
            r["call_expansion_limit"],
        )
    )

    print("\nRanked Results")
    print("==============")
    for idx, row in enumerate(rows, start=1):
        print(
            f"{idx:02d}. d={row['top_k_dense']} m={row['top_k_after_merge']} "
            f"c={row['call_expansion_limit']} | hit@{args.k}={row['hit_at_k']:.3f} "
            f"citation_coverage={row['citation_coverage']:.3f}"
        )

    best = rows[0]
    print("\nBest Config")
    print("===========")
    print(
        f"RETRIEVAL_TOP_K_DENSE={best['top_k_dense']}\n"
        f"RETRIEVAL_TOP_K_AFTER_MERGE={best['top_k_after_merge']}\n"
        f"RETRIEVAL_CALL_EXPANSION_LIMIT={best['call_expansion_limit']}"
    )


if __name__ == "__main__":
    main()
