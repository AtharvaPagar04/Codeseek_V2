"""Scheduled Qdrant snapshot backups with retention policy."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def _invoke_backup(host: str, port: int, collection: str, out_dir: str) -> None:
    script = Path(__file__).resolve().parent / "qdrant_snapshot_backup.py"
    cmd = [
        "python",
        str(script),
        "--host",
        host,
        "--port",
        str(port),
        "--collection",
        collection,
        "--out-dir",
        out_dir,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Backup failed for collection {collection}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )


def _collection_files(out_dir: Path, collection: str) -> list[Path]:
    prefix = f"{collection}__"
    return sorted([p for p in out_dir.glob(f"{prefix}*") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)


def _prune_retention(
    out_dir: Path,
    collection: str,
    keep_last: int,
    max_age_days: int,
) -> tuple[int, int]:
    files = _collection_files(out_dir, collection)
    now = time.time()
    cutoff = now - (max_age_days * 86400)
    deleted_count = 0
    kept_count = 0

    for idx, file in enumerate(files):
        mtime = file.stat().st_mtime
        too_old = mtime < cutoff
        over_count = idx >= keep_last
        if too_old or over_count:
            file.unlink(missing_ok=True)
            deleted_count += 1
        else:
            kept_count += 1
    return kept_count, deleted_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scheduled Qdrant backup + retention cleanup."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument(
        "--collections",
        required=True,
        help="Comma-separated collection names.",
    )
    parser.add_argument("--out-dir", default="backups/qdrant")
    parser.add_argument("--keep-last", type=int, default=14)
    parser.add_argument("--max-age-days", type=int, default=30)
    args = parser.parse_args()

    if args.keep_last <= 0:
        raise SystemExit("--keep-last must be > 0")
    if args.max_age_days <= 0:
        raise SystemExit("--max-age-days must be > 0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    if not collections:
        raise SystemExit("No collections provided.")

    print(f"Running scheduled backup for {len(collections)} collection(s)")
    for collection in collections:
        _invoke_backup(args.host, args.port, collection, str(out_dir))
        kept, deleted = _prune_retention(
            out_dir=out_dir,
            collection=collection,
            keep_last=args.keep_last,
            max_age_days=args.max_age_days,
        )
        print(
            f"[{collection}] retention applied: kept={kept} deleted={deleted} "
            f"(keep_last={args.keep_last}, max_age_days={args.max_age_days})"
        )


if __name__ == "__main__":
    main()
