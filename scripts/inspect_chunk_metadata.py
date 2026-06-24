#!/usr/bin/env python3
"""
inspect_chunk_metadata.py
=========================
Read-only Postgres-aware chunk metadata inspector for CodeSeek.

Inspects chunk descriptions, labels, file paths, symbols, line ranges,
chunk types, Qdrant payload keys, and Postgres session/file/chunk mappings.

Usage:
    python scripts/inspect_chunk_metadata.py --session-id <id> --limit 20
    python scripts/inspect_chunk_metadata.py --collection <name> --limit 20
    python scripts/inspect_chunk_metadata.py --collection <name> --keys
    python scripts/inspect_chunk_metadata.py --collection <name> --missing-descriptions
    python scripts/inspect_chunk_metadata.py --collection <name> --missing-labels
    python scripts/inspect_chunk_metadata.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# Payload field name candidates (checked in order)
# ---------------------------------------------------------------------------

DESCRIPTION_FIELDS = [
    "description",
    "generated_description",
    "chunk_description",
    "llm_description",
    "summary",
    "docstring",
    "semantic_summary",
]

LABEL_FIELDS = [
    "labels",
    "label",
    "semantic_label",
    "chunk_labels",
    "tags",
    "categories",
    "intent_labels",
]

PATH_FIELDS = [
    "relative_path",
    "path",
    "file_path",
    "repo_path",
    "source_path",
]

SYMBOL_FIELDS = [
    "symbol_name",
    "symbol",
    "function",
    "class_name",
    "name",
]

TYPE_FIELDS = [
    "chunk_type",
    "type",
    "kind",
]

START_LINE_FIELDS = [
    "start_line",
    "line_start",
]

END_LINE_FIELDS = [
    "end_line",
    "line_end",
]

CHUNK_ID_FIELDS = [
    "chunk_id",
    "id",
    "vector_id",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first(payload: dict, candidates: list[str], default=None):
    """Return the value of the first matching key found in payload."""
    for key in candidates:
        if key in payload:
            return payload[key]
    return default


def _normalize_labels(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    if isinstance(raw, str):
        return [raw] if raw else []
    return [str(raw)]


def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _safe_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Normalized chunk record
# ---------------------------------------------------------------------------

def normalize_point(point_id, payload: dict) -> dict:
    """Extract normalized fields from a raw Qdrant point payload."""
    description_raw = _first(payload, DESCRIPTION_FIELDS)
    desc_field_used = next((f for f in DESCRIPTION_FIELDS if f in payload), None)

    label_raw = _first(payload, LABEL_FIELDS)
    labels = _normalize_labels(label_raw)
    label_field_used = next((f for f in LABEL_FIELDS if f in payload), None)

    path = _first(payload, PATH_FIELDS, "")
    symbol = _first(payload, SYMBOL_FIELDS, "")
    chunk_type = _first(payload, TYPE_FIELDS, "")
    start_line = _safe_int(_first(payload, START_LINE_FIELDS))
    end_line = _safe_int(_first(payload, END_LINE_FIELDS))
    chunk_id = _first(payload, CHUNK_ID_FIELDS, "")

    description = str(description_raw) if description_raw is not None else ""

    return {
        "point_id": str(point_id),
        "chunk_id": chunk_id or "",
        "path": path or "",
        "symbol": symbol or "",
        "chunk_type": chunk_type or "",
        "start_line": start_line,
        "end_line": end_line,
        "labels": labels,
        "description": description,
        "payload_keys": sorted(payload.keys()),
        # internal meta for --keys and filters
        "_has_description": bool(description.strip()),
        "_has_labels": bool(labels),
        "_desc_field": desc_field_used,
        "_label_field": label_field_used,
        "_raw_payload": payload,
    }


# ---------------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------------

def _pg_connect(database_url: str, debug: bool = False):
    try:
        import psycopg
        from psycopg.rows import dict_row
        conn = psycopg.connect(database_url, row_factory=dict_row)
        return conn
    except ImportError:
        _err("psycopg is not installed. Install it with: pip install psycopg[binary]")
        sys.exit(1)
    except Exception as exc:
        if debug:
            traceback.print_exc()
        _err(
            f"Could not connect to Postgres at {_redact_url(database_url)}.\n"
            "  Provide --collection to inspect Qdrant directly."
        )
        return None


def _redact_url(url: str) -> str:
    """Redact password from a DSN URL for safe printing."""
    try:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(url)
        redacted = p._replace(netloc=f"{p.username or ''}:***@{p.hostname}" +
                               (f":{p.port}" if p.port else ""))
        return urlunparse(redacted)
    except Exception:
        return "<database-url>"


def _sqlite_connect(db_path: str, debug: bool = False):
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:
        if debug:
            traceback.print_exc()
        _err(f"Could not open SQLite database at {db_path}.")
        return None


def fetch_session(conn, session_id: str, backend: str, debug: bool = False) -> dict | None:
    """Fetch repo_sessions row for the given session_id."""
    sql = """
        SELECT
            id,
            repo_full_name,
            collection,
            status,
            repo_root,
            last_indexed_commit,
            indexed_branch,
            current_branch,
            updated_at
        FROM repo_sessions
        WHERE id = %s
    """ if backend == "postgres" else """
        SELECT
            id,
            repo_full_name,
            collection,
            status,
            repo_root,
            last_indexed_commit,
            indexed_branch,
            current_branch,
            updated_at
        FROM repo_sessions
        WHERE id = ?
    """
    try:
        cur = conn.cursor()
        cur.execute(sql, (session_id,))
        row = cur.fetchone()
        if row is None:
            return None
        if isinstance(row, dict):
            return dict(row)
        # sqlite3.Row
        return {k: row[k] for k in row.keys()}
    except Exception as exc:
        if debug:
            traceback.print_exc()
        _warn(f"Session lookup failed: {exc}")
        return None


def fetch_file_chunks(conn, session_id: str, backend: str, limit: int, debug: bool = False) -> list[dict]:
    """
    Fetch session_file_chunks joined with session_files for a session.
    Returns empty list if tables are missing or query fails.
    """
    sql = """
        SELECT
            sf.repo_path,
            sf.status AS file_status,
            sf.deleted_at,
            sfc.chunk_id,
            sfc.vector_id,
            sfc.symbol,
            sfc.start_line,
            sfc.end_line
        FROM session_file_chunks sfc
        JOIN session_files sf ON sf.id = sfc.session_file_id
        WHERE sf.session_id = {ph}
        LIMIT {ph}
    """.format(ph="%s" if backend == "postgres" else "?")

    try:
        cur = conn.cursor()
        cur.execute(sql, (session_id, limit))
        rows = cur.fetchall()
        if not rows:
            return []
        results = []
        for row in rows:
            if isinstance(row, dict):
                results.append(dict(row))
            else:
                results.append({k: row[k] for k in row.keys()})
        return results
    except Exception as exc:
        if debug:
            traceback.print_exc()
        _warn(f"session_file_chunks lookup failed (may not exist in this installation): {exc}")
        return []


# ---------------------------------------------------------------------------
# Qdrant scroll
# ---------------------------------------------------------------------------

def qdrant_scroll(
    qdrant_url: str,
    collection: str,
    limit: int,
    max_scan: int,
    filters: dict,
    debug: bool = False,
) -> list[dict]:
    """
    Scroll Qdrant collection, applying filters client-side.

    filters keys:
      path_substr, label_substr, symbol_substr,
      missing_descriptions, missing_labels
    """
    import urllib.request
    import urllib.error

    base = qdrant_url.rstrip("/")
    scroll_url = f"{base}/collections/{collection}/points/scroll"

    # Verify collection exists
    try:
        req = urllib.request.Request(f"{base}/collections/{collection}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            _err(f"Qdrant collection '{collection}' was not found.")
            sys.exit(1)
        if debug:
            traceback.print_exc()
        _err(f"Could not connect to Qdrant at {qdrant_url}: HTTP {exc.code}")
        sys.exit(1)
    except Exception as exc:
        if debug:
            traceback.print_exc()
        _err(f"Could not connect to Qdrant at {qdrant_url}: {exc}")
        sys.exit(1)

    collected: list[dict] = []
    next_offset = None
    scanned = 0

    path_sub = (filters.get("path_substr") or "").lower()
    label_sub = (filters.get("label_substr") or "").lower()
    symbol_sub = (filters.get("symbol_substr") or "").lower()
    want_missing_desc = filters.get("missing_descriptions", False)
    want_missing_labels = filters.get("missing_labels", False)

    # Batch size: fetch more per page when we have filters to satisfy
    batch = min(max(limit * 3, 50), 250)

    while len(collected) < limit and scanned < max_scan:
        payload_body: dict = {
            "limit": batch,
            "with_payload": True,
            "with_vector": False,
        }
        if next_offset is not None:
            payload_body["offset"] = next_offset

        body_bytes = json.dumps(payload_body).encode("utf-8")
        req = urllib.request.Request(
            scroll_url,
            data=body_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            if debug:
                traceback.print_exc()
            _err(f"Qdrant scroll request failed: {exc}")
            sys.exit(1)

        result = data.get("result", {})
        points = result.get("points", [])
        next_offset = result.get("next_page_offset")

        if not points:
            break

        for pt in points:
            scanned += 1
            if scanned > max_scan:
                break
            pt_id = pt.get("id", "")
            raw_payload = pt.get("payload") or {}
            norm = normalize_point(pt_id, raw_payload)

            # Apply filters
            if path_sub and path_sub not in norm["path"].lower():
                continue
            if symbol_sub and symbol_sub not in norm["symbol"].lower():
                continue
            if label_sub:
                labels_lower = " ".join(norm["labels"]).lower()
                if label_sub not in labels_lower:
                    continue
            if want_missing_desc and norm["_has_description"]:
                continue
            if want_missing_labels and norm["_has_labels"]:
                continue

            collected.append(norm)
            if len(collected) >= limit:
                break

        if not next_offset:
            break  # No more pages

    return collected, scanned


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def print_session_summary(session: dict) -> None:
    print()
    print("=" * 64)
    print("  SESSION SUMMARY")
    print("=" * 64)
    print(f"  Session ID     : {session.get('id', '')}")
    print(f"  Repo           : {session.get('repo_full_name', '')}")
    print(f"  Collection     : {session.get('collection', '')}")
    print(f"  Status         : {session.get('status', '')}")
    print(f"  Repo Root      : {session.get('repo_root', '')}")
    print(f"  Indexed Commit : {session.get('last_indexed_commit', '')}")
    print(f"  Indexed Branch : {session.get('indexed_branch', '')}")
    print(f"  Current Branch : {session.get('current_branch', '')}")
    print(f"  Updated At     : {session.get('updated_at', '')}")
    print("=" * 64)
    print()


def print_chunk(idx: int, norm: dict, full_description: bool = False, raw: bool = False) -> None:
    desc_limit = 500 if not full_description else 0
    desc = norm["description"]
    if desc_limit and len(desc) > desc_limit:
        desc = _truncate(desc, desc_limit)

    lines = ""
    if norm["start_line"] is not None and norm["end_line"] is not None:
        lines = f"{norm['start_line']}-{norm['end_line']}"
    elif norm["start_line"] is not None:
        lines = str(norm["start_line"])

    labels_str = ", ".join(norm["labels"]) if norm["labels"] else "(none)"
    keys_str = ", ".join(norm["payload_keys"])

    print(f"\nChunk {idx}")
    print(f"  Point ID   : {norm['point_id']}")
    print(f"  Chunk ID   : {norm['chunk_id']}")
    print(f"  Path       : {norm['path']}")
    print(f"  Symbol     : {norm['symbol']}")
    print(f"  Type       : {norm['chunk_type']}")
    print(f"  Lines      : {lines}")
    print(f"  Labels     : {labels_str}")
    if desc:
        print(f"  Description:")
        for line in desc.splitlines():
            print(f"    {line}")
    else:
        print(f"  Description: (none)")
    print(f"  Keys       : {keys_str}")

    if raw:
        print(f"  Raw payload:")
        print("    " + json.dumps(norm["_raw_payload"], default=str, indent=2).replace("\n", "\n    "))


def print_keys_summary(chunks: list[dict], scanned: int) -> None:
    key_counter: Counter = Counter()
    with_desc = 0
    without_desc = 0
    with_labels = 0
    without_labels = 0

    for norm in chunks:
        for k in norm["payload_keys"]:
            key_counter[k] += 1
        if norm["_has_description"]:
            with_desc += 1
        else:
            without_desc += 1
        if norm["_has_labels"]:
            with_labels += 1
        else:
            without_labels += 1

    print()
    print("=" * 64)
    print("  PAYLOAD KEY FREQUENCY")
    print("=" * 64)
    for key, count in key_counter.most_common():
        print(f"  {key:<40}: {count}")
    print()
    print(f"  Total scanned            : {scanned}")
    print(f"  Matching chunks shown    : {len(chunks)}")
    print(f"  With description         : {with_desc}")
    print(f"  Missing description      : {without_desc}")
    print(f"  With labels              : {with_labels}")
    print(f"  Missing labels           : {without_labels}")
    print("=" * 64)


def output_json(session: dict | None, chunks: list[dict], scanned: int) -> None:
    out = {
        "session": session,
        "scanned": scanned,
        "count": len(chunks),
        "chunks": [
            {
                "point_id": c["point_id"],
                "chunk_id": c["chunk_id"],
                "path": c["path"],
                "symbol": c["symbol"],
                "chunk_type": c["chunk_type"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "labels": c["labels"],
                "description": c["description"],
                "payload_keys": c["payload_keys"],
            }
            for c in chunks
        ],
    }
    print(json.dumps(out, default=str, indent=2))


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _err(msg: str) -> None:
    print(f"\n[ERROR] {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inspect_chunk_metadata.py",
        description=(
            "Read-only Postgres-aware chunk metadata inspector for CodeSeek.\n"
            "Inspects chunk descriptions, labels, file paths, symbols,\n"
            "line ranges, chunk types, and Qdrant payload keys."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/inspect_chunk_metadata.py --session-id abc123 --limit 10
  python scripts/inspect_chunk_metadata.py --collection codeseek --limit 20
  python scripts/inspect_chunk_metadata.py --collection codeseek --keys
  python scripts/inspect_chunk_metadata.py --collection codeseek --missing-descriptions
  python scripts/inspect_chunk_metadata.py --collection codeseek --path retrieval/search/searcher.py
  python scripts/inspect_chunk_metadata.py --collection codeseek --label domain:auth --raw
  python scripts/inspect_chunk_metadata.py --collection codeseek --json --limit 50
        """,
    )

    # Connection
    conn_g = p.add_argument_group("connection")
    conn_g.add_argument(
        "--session-id",
        metavar="ID",
        help="repo_sessions.id — resolves collection automatically.",
    )
    conn_g.add_argument(
        "--collection",
        metavar="NAME",
        help="Qdrant collection name. Skips session lookup when provided.",
    )
    conn_g.add_argument(
        "--database-url",
        metavar="URL",
        help="Postgres DSN. Falls back to CODESEEK_DATABASE_URL env var.",
    )
    conn_g.add_argument(
        "--sqlite-path",
        metavar="PATH",
        help="Path to SQLite database file. Falls back to CODESEEK_SQLITE_PATH / CODESEEK_DB_PATH / ./data/codeseek.db.",
    )
    conn_g.add_argument(
        "--qdrant-url",
        metavar="URL",
        default=None,
        help="Qdrant base URL. Falls back to QDRANT_URL env or http://localhost:6333.",
    )

    # Paging
    page_g = p.add_argument_group("paging")
    page_g.add_argument("--limit", type=int, default=20, metavar="N", help="Max chunks to display (default: 20).")
    page_g.add_argument("--max-scan", type=int, default=1000, metavar="N", help="Max points to scan in Qdrant scroll (default: 1000).")

    # Filters
    flt_g = p.add_argument_group("filters")
    flt_g.add_argument("--path", metavar="SUBSTR", help="Filter chunks by path substring (case-insensitive).")
    flt_g.add_argument("--label", metavar="SUBSTR", help="Filter chunks by label substring (case-insensitive).")
    flt_g.add_argument("--symbol", metavar="SUBSTR", help="Filter chunks by symbol substring (case-insensitive).")
    flt_g.add_argument("--missing-descriptions", action="store_true", help="Only show chunks with missing/empty description.")
    flt_g.add_argument("--missing-labels", action="store_true", help="Only show chunks with missing/empty labels.")

    # Output
    out_g = p.add_argument_group("output")
    out_g.add_argument("--raw", action="store_true", help="Print full raw Qdrant payload JSON for each chunk.")
    out_g.add_argument("--json", action="store_true", help="Output structured JSON instead of readable table.")
    out_g.add_argument("--keys", action="store_true", help="Print payload key frequency summary.")
    out_g.add_argument("--full-description", action="store_true", help="Show full description text (not truncated).")

    # Debug
    p.add_argument("--debug", action="store_true", help="Print full exception tracebacks on errors.")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    debug = args.debug

    # Validate: at least one of --session-id or --collection required
    if not args.session_id and not args.collection:
        _err(
            "You must provide at least one of --session-id or --collection.\n\n"
            "  --session-id ID        : Look up Postgres repo_sessions to find collection + session metadata\n"
            "  --collection NAME      : Skip session lookup and query Qdrant collection directly\n\n"
            "Run with --help for full usage."
        )
        sys.exit(1)

    # Determine DB backend
    db_backend = os.environ.get("CODESEEK_DB_BACKEND", "").strip().lower()

    # Resolve Postgres DSN
    db_url = args.database_url or os.environ.get("CODESEEK_DATABASE_URL", "").strip()
    if not db_url:
        db_url = os.environ.get("DATABASE_URL", "").strip()

    # Resolve SQLite path
    db_path = (
        args.sqlite_path
        or os.environ.get("CODESEEK_SQLITE_PATH", "").strip()
        or os.environ.get("CODESEEK_DB_PATH", "").strip()
        or "./data/codeseek.db"
    )
    from pathlib import Path
    db_path = str(Path(db_path).resolve())

    # If backend not explicitly set, auto-detect or default to SQLite
    if not db_backend:
        if db_url.startswith("postgres"):
            db_backend = "postgres"
        else:
            db_backend = "sqlite"

    # Resolve Qdrant URL
    qdrant_url = (
        args.qdrant_url
        or os.environ.get("QDRANT_URL", "")
        or "http://localhost:6333"
    ).rstrip("/")

    # -----------------------------------------------------------------------
    # Step 1: Session lookup
    # -----------------------------------------------------------------------
    session_info: dict | None = None
    pg_chunks: list[dict] = []
    collection = args.collection

    db_conn = None
    if args.session_id:
        if db_backend == "postgres":
            if not db_url:
                _warn(
                    "No --database-url or CODESEEK_DATABASE_URL env var found.\n"
                    "  Skipping Postgres session lookup. Set database URL or use --collection."
                )
            else:
                db_conn = _pg_connect(db_url, debug=debug)
        else:
            if not db_path:
                _warn(
                    "No --sqlite-path or CODESEEK_SQLITE_PATH env var found.\n"
                    "  Skipping SQLite session lookup. Set SQLite path or use --collection."
                )
            else:
                db_conn = _sqlite_connect(db_path, debug=debug)

        if db_conn:
            session_info = fetch_session(db_conn, args.session_id, db_backend, debug=debug)
            if session_info is None:
                _err(f"No repo_sessions row found for session id: {args.session_id}")
                if not args.collection:
                    db_conn.close()
                    sys.exit(1)
                _warn("Continuing with Qdrant-only mode since --collection was also provided.")
            else:
                # Use session's collection if --collection not explicitly specified
                if not collection:
                    collection = session_info.get("collection", "")

                # Optional: fetch file→chunk Postgres/SQLite mappings
                pg_chunks = fetch_file_chunks(
                    db_conn, args.session_id, db_backend, limit=args.limit * 5, debug=debug
                )

    if not collection:
        _err(
            "Could not determine Qdrant collection name.\n"
            "  Provide --collection NAME or ensure --session-id resolves a valid session."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: Print session summary (if available)
    # -----------------------------------------------------------------------
    if session_info and not args.json:
        print_session_summary(session_info)

    if not args.json:
        _info(f"Querying Qdrant collection: '{collection}' at {qdrant_url}")
        _info(f"Limit: {args.limit}  |  Max scan: {args.max_scan}")
        if args.path:
            _info(f"Filter: path contains '{args.path}'")
        if args.label:
            _info(f"Filter: label contains '{args.label}'")
        if args.symbol:
            _info(f"Filter: symbol contains '{args.symbol}'")
        if args.missing_descriptions:
            _info("Filter: only missing descriptions")
        if args.missing_labels:
            _info("Filter: only missing labels")
        print()

    # -----------------------------------------------------------------------
    # Step 3: Qdrant scroll
    # -----------------------------------------------------------------------
    filters = {
        "path_substr": args.path or "",
        "label_substr": args.label or "",
        "symbol_substr": args.symbol or "",
        "missing_descriptions": args.missing_descriptions,
        "missing_labels": args.missing_labels,
    }

    chunks, scanned = qdrant_scroll(
        qdrant_url=qdrant_url,
        collection=collection,
        limit=args.limit,
        max_scan=args.max_scan,
        filters=filters,
        debug=debug,
    )

    # -----------------------------------------------------------------------
    # Step 4: Output
    # -----------------------------------------------------------------------
    if not chunks:
        if args.missing_descriptions or args.missing_labels or args.path or args.label or args.symbol:
            _info(f"No matching chunks found after scanning {scanned} points in '{collection}'.")
        else:
            _info(f"No points found in collection '{collection}'.")
        if db_conn:
            db_conn.close()
        return

    # Annotate with Postgres chunk metadata if available
    pg_chunk_map: dict[str, dict] = {}
    for pcc in pg_chunks:
        cid = pcc.get("chunk_id") or ""
        vid = pcc.get("vector_id") or ""
        if cid:
            pg_chunk_map[cid] = pcc
        if vid and vid not in pg_chunk_map:
            pg_chunk_map[vid] = pcc

    if args.json:
        output_json(session_info, chunks, scanned)
    else:
        print(f"Found {len(chunks)} chunks (scanned {scanned} points):\n")

        if pg_chunk_map and not args.json:
            _info(
                f"Postgres session_file_chunks enrichment: {len(pg_chunk_map)} chunk-id mappings loaded."
            )

        for i, norm in enumerate(chunks, 1):
            # Enrich with Postgres path / status if chunk_id matches
            pg_meta = pg_chunk_map.get(norm["chunk_id"]) or pg_chunk_map.get(norm["point_id"])
            if pg_meta and not norm["path"]:
                norm["path"] = pg_meta.get("repo_path", "")
            print_chunk(
                idx=i,
                norm=norm,
                full_description=args.full_description,
                raw=args.raw,
            )

        if not chunks[0]["_has_description"]:
            available_keys = sorted(chunks[0]["payload_keys"])
            _warn(
                f"No description-like field found in the first chunk.\n"
                f"  Available payload keys: {', '.join(available_keys)}"
            )

        if args.keys:
            print_keys_summary(chunks, scanned)
        else:
            # Always print a quick coverage summary at the end
            with_desc = sum(1 for c in chunks if c["_has_description"])
            with_lbl = sum(1 for c in chunks if c["_has_labels"])
            print()
            print(
                f"  Summary: {with_desc}/{len(chunks)} have description | "
                f"{with_lbl}/{len(chunks)} have labels | "
                f"{scanned} points scanned"
            )

    if db_conn:
        try:
            db_conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
