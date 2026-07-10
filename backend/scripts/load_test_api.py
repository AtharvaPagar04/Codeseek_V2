"""Concurrent load test for retrieval API."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import Any

import httpx


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, int(round((q / 100.0) * (len(values) - 1)))))
    return sorted(values)[idx]


async def _one(
    client: httpx.AsyncClient,
    url: str,
    token: str,
    query: str,
    request_id: str,
) -> tuple[int, float]:
    started = time.perf_counter()
    response = await client.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "X-Request-Id": request_id,
            "Content-Type": "application/json",
        },
        json={"query": query},
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return response.status_code, latency_ms


async def run(args: argparse.Namespace) -> dict[str, Any]:
    url = f"{args.base_url.rstrip('/')}/api/v1/query"
    sem = asyncio.Semaphore(args.concurrency)
    statuses: list[int] = []
    latencies: list[float] = []

    async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
        async def worker(i: int) -> None:
            async with sem:
                status, lat_ms = await _one(
                    client,
                    url=url,
                    token=args.api_key,
                    query=args.query,
                    request_id=f"load-{i}",
                )
                statuses.append(status)
                latencies.append(lat_ms)

        await asyncio.gather(*(worker(i) for i in range(args.requests)))

    ok = sum(1 for s in statuses if 200 <= s < 300)
    errors = len(statuses) - ok
    return {
        "total_requests": len(statuses),
        "ok_requests": ok,
        "error_requests": errors,
        "error_rate": (errors / len(statuses)) if statuses else 0.0,
        "latency_ms": {
            "avg": statistics.fmean(latencies) if latencies else 0.0,
            "p50": _pct(latencies, 50),
            "p95": _pct(latencies, 95),
            "p99": _pct(latencies, 99),
            "max": max(latencies) if latencies else 0.0,
        },
        "status_counts": {str(code): statuses.count(code) for code in sorted(set(statuses))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent load test for retrieval API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--query", default="Where is account_info implemented?")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-error-rate", type=float, default=-1.0)
    parser.add_argument("--max-p95-ms", type=float, default=-1.0)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    result = asyncio.run(run(args))
    print("Load Test Results")
    print("=================")
    print(f"total_requests: {result['total_requests']}")
    print(f"ok_requests: {result['ok_requests']}")
    print(f"error_requests: {result['error_requests']}")
    print(f"error_rate: {result['error_rate']:.3f}")
    print("latency_ms:")
    print(f"  avg: {result['latency_ms']['avg']:.2f}")
    print(f"  p50: {result['latency_ms']['p50']:.2f}")
    print(f"  p95: {result['latency_ms']['p95']:.2f}")
    print(f"  p99: {result['latency_ms']['p99']:.2f}")
    print(f"  max: {result['latency_ms']['max']:.2f}")
    print("status_counts:")
    for status, count in result["status_counts"].items():
        print(f"  {status}: {count}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    failures = []
    if args.max_error_rate >= 0 and result["error_rate"] > args.max_error_rate:
        failures.append(
            f"error_rate {result['error_rate']:.3f} > {args.max_error_rate:.3f}"
        )
    p95 = float(result["latency_ms"]["p95"])
    if args.max_p95_ms >= 0 and p95 > args.max_p95_ms:
        failures.append(f"p95 {p95:.2f}ms > {args.max_p95_ms:.2f}ms")
    if failures:
        print("Load gate failed:")
        for item in failures:
            print(f"- {item}")
        sys.exit(1)


if __name__ == "__main__":
    main()
