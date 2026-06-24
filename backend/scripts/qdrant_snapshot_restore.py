"""Restore a Qdrant collection snapshot from a local file."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore Qdrant collection snapshot")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--snapshot-file", required=True)
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot_file)
    if not snapshot_path.exists():
        raise SystemExit(f"Snapshot file not found: {snapshot_path}")

    base = f"http://{args.host}:{args.port}"
    upload_url = f"{base}/collections/{args.collection}/snapshots/upload"
    with snapshot_path.open("rb") as handle:
        resp = requests.post(
            upload_url,
            files={"snapshot": (snapshot_path.name, handle, "application/octet-stream")},
            timeout=120,
        )
    resp.raise_for_status()
    print(f"Snapshot restore requested for collection: {args.collection}")


if __name__ == "__main__":
    main()
