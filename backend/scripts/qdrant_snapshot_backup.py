"""Create a Qdrant collection snapshot and download it locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup Qdrant collection snapshot")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--out-dir", default="backups/qdrant")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    create_url = f"{base}/collections/{args.collection}/snapshots"
    create_resp = requests.post(create_url, timeout=30)
    create_resp.raise_for_status()
    snap_name = create_resp.json()["result"]["name"]

    download_url = f"{base}/collections/{args.collection}/snapshots/{snap_name}"
    target = out_dir / f"{args.collection}__{snap_name}"
    with requests.get(download_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    print(f"Snapshot saved: {target}")


if __name__ == "__main__":
    main()
