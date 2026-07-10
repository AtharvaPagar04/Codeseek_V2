"""Black-box API checks against a running retrieval service."""

from __future__ import annotations

import argparse
import sys

import requests


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Black-box checks for retrieval API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--query", default="Where is account_info implemented?")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    # Health endpoint
    health = requests.get(f"{base}/api/v1/health", timeout=20)
    health.raise_for_status()
    health_json = health.json()
    _require("status" in health_json, "health payload missing status")

    # Unauthorized request should fail
    unauth = requests.post(
        f"{base}/api/v1/query",
        json={"query": args.query},
        timeout=30,
    )
    _require(unauth.status_code == 401, f"expected 401, got {unauth.status_code}")

    # Authorized request should succeed
    req = requests.post(
        f"{base}/api/v1/query",
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "X-Request-Id": "blackbox-1",
            "Content-Type": "application/json",
        },
        json={"query": args.query},
        timeout=120,
    )
    req.raise_for_status()
    data = req.json()
    _require("request_id" in data and data["request_id"], "missing request_id")
    _require("answer" in data and isinstance(data["answer"], str), "missing answer")
    _require("sources" in data and isinstance(data["sources"], list), "missing sources list")
    _require("metrics" in data and isinstance(data["metrics"], dict), "missing metrics")

    # Metrics endpoint should expose Prometheus text
    metrics = requests.get(f"{base}/api/v1/metrics", timeout=20)
    metrics.raise_for_status()
    body = metrics.text
    _require("codeseek_api_requests_total" in body, "missing codeseek_api_requests_total metric")

    print("API black-box checks passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"API black-box checks failed: {exc}")
        sys.exit(1)
