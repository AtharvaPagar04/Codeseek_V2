"""Index health checker for CodeSeek retrieval validation."""

import os
import sys
from pathlib import Path

# Load .env file before importing retrieval config
def _load_env_file():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                key = k.strip()
                if key not in os.environ:
                    os.environ[key] = v.strip()

_load_env_file()

from qdrant_client.http import exceptions as qdrant_exceptions
from retrieval.support.qdrant_config import create_qdrant_client

# Ensure backend directory is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.db import db_cursor

def is_file_type_required(rel_path: str, chunk_type: str) -> bool:
    """Determine if a chunk requires a file_type according to Task 2 rules."""
    if chunk_type == "repo_summary":
        return True
    
    path_lower = (rel_path or "").lower()
    filename = Path(path_lower).name
    
    # Dockerfiles
    if filename == "dockerfile" or "docker-compose" in path_lower:
        return True
    # Env files
    if filename.startswith(".env"):
        return True
    # Package / Manifest / pyproject / requirements / lock files
    if filename in (
        "pyproject.toml", 
        "requirements.txt", 
        "package.json", 
        "package-lock.json", 
        "poetry.lock", 
        "yarn.lock", 
        "pnpm-lock.yaml"
    ):
        return True
    # Config files
    if filename in ("config.py", "settings.py") or filename.endswith(".json") or filename.endswith(".yaml") or filename.endswith(".yml") or filename.endswith(".toml") or filename.endswith(".ini"):
        return True
    if "manifest" in path_lower:
        return True
        
    return False

def run_index_health_check(
    session_id: str | None = None,
    collection_name: str | None = None,
    repo_root: str | None = None
) -> dict:
    """Validate that the indexed collection and database state are healthy and fresh."""
    db_session = None
    resolved_collection = collection_name
    resolved_repo_root = repo_root
    indexed_commit = ""
    current_commit = ""

    # 1. Resolve parameters from database session if session_id is provided
    if session_id:
        try:
            with db_cursor() as (conn, cursor):
                cursor.execute(
                    "SELECT collection, repo_root, last_indexed_commit, current_commit_sha FROM repo_sessions WHERE id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                if row:
                    db_session = dict(row)
                    if not resolved_collection:
                        resolved_collection = db_session.get("collection")
                    if not resolved_repo_root:
                        resolved_repo_root = db_session.get("repo_root")
                    indexed_commit = db_session.get("last_indexed_commit", "")
                    current_commit = db_session.get("current_commit_sha", "")
                else:
                    return {
                        "status": "FAIL",
                        "error": f"Session ID '{session_id}' not found in database."
                    }
        except Exception as e:
            return {
                "status": "FAIL",
                "error": f"Database connection failed: {str(e)}"
            }

    # Fallback to defaults from environment if still unresolved
    if not resolved_collection:
        resolved_collection = os.getenv("QDRANT_COLLECTION_NAME", "repository_chunks")
    if not resolved_repo_root:
        resolved_repo_root = os.getenv("RETRIEVAL_REPO_ROOT", str(Path.cwd()))

    resolved_repo_root = str(Path(resolved_repo_root).resolve())

    # Get git HEAD if git repo is present
    if not current_commit and Path(resolved_repo_root).joinpath(".git").exists():
        try:
            import subprocess
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=resolved_repo_root,
                capture_output=True,
                text=True,
                check=True
            )
            current_commit = res.stdout.strip()
        except Exception:
            pass

    # 2. Connect to Qdrant
    try:
        client = create_qdrant_client()
    except Exception as e:
        return {
            "status": "FAIL",
            "error": f"Failed to connect to Qdrant: {str(e)}"
        }

    # 3. Check collection existence and count points
    try:
        coll_info = client.get_collection(collection_name=resolved_collection)
        points_count = coll_info.points_count
    except qdrant_exceptions.UnexpectedResponse as e:
        if e.status_code == 404:
            return {
                "status": "FAIL",
                "error": f"Qdrant collection '{resolved_collection}' does not exist."
            }
        return {
            "status": "FAIL",
            "error": f"Qdrant query failed: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "FAIL",
            "error": f"Failed to retrieve Qdrant collection: {str(e)}"
        }

    # 4. Scroll all points in the collection
    points = []
    offset = None
    try:
        while True:
            res = client.scroll(
                collection_name=resolved_collection,
                limit=1000,
                with_payload=True,
                with_vectors=False,
                offset=offset
            )
            if not res:
                break
            hits, next_offset = res
            points.extend(hits)
            if not next_offset:
                break
            offset = next_offset
    except Exception as e:
        return {
            "status": "FAIL",
            "error": f"Failed scrolling points from collection: {str(e)}"
        }

    total_points = len(points)
    if total_points == 0:
        return {
            "status": "FAIL",
            "error": "Collection is empty (0 points found)."
        }

    # 5. Audit fields
    missing_chunk_id = 0
    missing_relative_path = 0
    missing_line_range = 0
    missing_content = 0
    missing_chunk_type = 0
    
    missing_file_type_total = 0
    missing_file_type_required = 0
    file_type_optional_missing = 0
    
    missing_labels = 0
    missing_code_intent = 0
    missing_qualified_symbol = 0

    points_with_labels = 0
    points_with_code_intent = 0
    points_with_question_use = 0
    points_with_domain = 0
    points_with_chunk_type = 0
    points_with_qualified_symbol = 0

    eligible_for_symbol = 0
    deleted_file_chunks = 0

    # Cache file existence check to speed up freshness validation
    file_existence_cache = {}

    offenders = {
        "missing_chunk_id": [],
        "missing_relative_path": [],
        "missing_line_range": [],
        "missing_content": [],
        "missing_chunk_type": [],
        "missing_file_type_required": [],
        "missing_qualified_symbol": [],
        "missing_labels": [],
        "missing_code_intent": [],
        "deleted_file_chunks": [],
    }

    def add_offender(category: str, pt, reason: str):
        payload = pt.payload or {}
        if len(offenders[category]) < 5:
            offenders[category].append({
                "chunk_id": payload.get("chunk_id", "N/A"),
                "relative_path": payload.get("relative_path", "N/A"),
                "chunk_type": payload.get("chunk_type", "N/A"),
                "file_type": payload.get("file_type", "N/A"),
                "symbol_name": payload.get("symbol_name", "N/A"),
                "reason": reason
            })

    for pt in points:
        payload = pt.payload or {}
        
        # Check basic fields
        if not payload.get("chunk_id"):
            missing_chunk_id += 1
            add_offender("missing_chunk_id", pt, "Missing chunk_id field")
        
        rel_path = payload.get("relative_path")
        if not rel_path:
            missing_relative_path += 1
            add_offender("missing_relative_path", pt, "Missing relative_path field")
        else:
            # Check if file exists in repo (virtual file __repo_summary__.md always exists)
            if rel_path == "__repo_summary__.md":
                file_existence_cache[rel_path] = True
            elif rel_path not in file_existence_cache:
                full_path = Path(resolved_repo_root) / rel_path
                file_existence_cache[rel_path] = full_path.is_file()
            
            if not file_existence_cache[rel_path]:
                deleted_file_chunks += 1
                add_offender("deleted_file_chunks", pt, f"File no longer exists: {rel_path}")

        if payload.get("start_line") is None or payload.get("end_line") is None:
            missing_line_range += 1
            add_offender("missing_line_range", pt, "Missing start_line or end_line")

        # Content field check
        has_content = any(payload.get(k) for k in ("content", "content_excerpt", "text"))
        if not has_content:
            missing_content += 1
            add_offender("missing_content", pt, "No content excerpt/text found")

        # Check types & identifiers
        ctype = payload.get("chunk_type")
        if not ctype:
            missing_chunk_type += 1
            add_offender("missing_chunk_type", pt, "Missing chunk_type")
        else:
            points_with_chunk_type += 1
            if ctype in ("function", "method", "class"):
                eligible_for_symbol += 1
                if payload.get("qualified_symbol"):
                    points_with_qualified_symbol += 1
                else:
                    missing_qualified_symbol += 1
                    add_offender("missing_qualified_symbol", pt, f"Missing qualified_symbol for chunk_type '{ctype}'")

        # File type required/optional check
        file_type = payload.get("file_type")
        if not file_type:
            missing_file_type_total += 1
            if is_file_type_required(rel_path, ctype):
                missing_file_type_required += 1
                add_offender("missing_file_type_required", pt, "Missing file_type on a required/config file")
            else:
                file_type_optional_missing += 1

        # Check labels & code_intent
        labels = payload.get("labels")
        if not labels:
            missing_labels += 1
            add_offender("missing_labels", pt, "Missing labels array")
        else:
            points_with_labels += 1
            if any(l.startswith("question_use:") for l in labels):
                points_with_question_use += 1
            if any(l.startswith("domain:") for l in labels):
                points_with_domain += 1

        if not payload.get("code_intent"):
            missing_code_intent += 1
            add_offender("missing_code_intent", pt, "Missing code_intent field")
        else:
            points_with_code_intent += 1

    # Calculate percentages
    label_coverage_percent = (points_with_labels / total_points) * 100.0 if total_points else 0.0
    code_intent_coverage_percent = (points_with_code_intent / total_points) * 100.0 if total_points else 0.0
    question_use_label_coverage_percent = (points_with_question_use / total_points) * 100.0 if total_points else 0.0
    domain_label_coverage_percent = (points_with_domain / total_points) * 100.0 if total_points else 0.0
    chunk_type_coverage_percent = (points_with_chunk_type / total_points) * 100.0 if total_points else 0.0
    qualified_symbol_coverage_percent = (
        (points_with_qualified_symbol / eligible_for_symbol * 100.0)
        if eligible_for_symbol > 0 else 100.0
    )

    # Determine freshness status
    repo_freshness_status = "current"
    if current_commit and indexed_commit:
        if current_commit != indexed_commit:
            repo_freshness_status = "commit_changed"
    elif not current_commit or not indexed_commit:
        repo_freshness_status = "unknown"

    if deleted_file_chunks > 0:
        repo_freshness_status = "files_deleted"

    # Check config-gated label requirements
    try:
        from rag_ingestion.config import ENABLE_CHUNK_LABELS
    except ImportError:
        ENABLE_CHUNK_LABELS = True

    # Define validation thresholds
    is_healthy = (
        missing_chunk_id == 0 and
        missing_relative_path == 0 and
        missing_line_range == 0 and
        deleted_file_chunks == 0 and
        missing_chunk_type == 0 and
        missing_file_type_required == 0 and
        missing_qualified_symbol == 0
    )
    if ENABLE_CHUNK_LABELS:
        is_healthy = is_healthy and (
            label_coverage_percent >= 90.0 and
            code_intent_coverage_percent >= 90.0 and
            missing_labels == 0 and
            missing_code_intent == 0
        )

    verdict = "PASS" if is_healthy else "FAIL"

    # Collect re-index reasons
    reasons = []
    if total_points < 30:
        reasons.append(f"Qdrant collection has only {total_points} points (low points_count)")
    if missing_chunk_id > 0:
        reasons.append(f"{missing_chunk_id} chunks missing chunk_id")
    if missing_relative_path > 0:
        reasons.append(f"{missing_relative_path} chunks missing relative_path")
    if missing_line_range > 0:
        reasons.append(f"{missing_line_range} chunks missing line range")
    if deleted_file_chunks > 0:
        reasons.append(f"{deleted_file_chunks} stale/deleted file path chunks found in Qdrant")
    if missing_chunk_type > 0:
        reasons.append(f"{missing_chunk_type} chunks missing chunk_type")
    if missing_file_type_required > 0:
        reasons.append(f"{missing_file_type_required} required files missing file_type")
    if missing_qualified_symbol > 0:
        reasons.append(f"{missing_qualified_symbol} qualified symbols missing for function/class chunks")
    if ENABLE_CHUNK_LABELS:
        if label_coverage_percent < 90.0:
            reasons.append(f"Label coverage is only {label_coverage_percent:.2f}% (required >= 90.0%)")
        if code_intent_coverage_percent < 90.0:
            reasons.append(f"Code intent coverage is only {code_intent_coverage_percent:.2f}% (required >= 90.0%)")
        if missing_labels > 0:
            reasons.append(f"{missing_labels} chunks missing labels")
        if missing_code_intent > 0:
            reasons.append(f"{missing_code_intent} chunks missing code_intent")

    # Generate re-index guidance if failed
    reindex_remediation = ""
    if not is_healthy:
        from evals.reindex_guidance import get_reindex_guidance
        reindex_remediation = get_reindex_guidance(reasons, resolved_repo_root, resolved_collection)

    report = {
        "status": verdict,
        "session_id": session_id or "N/A",
        "repo_root": resolved_repo_root,
        "collection": resolved_collection,
        "points_count": total_points,
        "reindex_remediation": reindex_remediation,
        "offenders": offenders,
        "metrics": {
            "missing_chunk_id": missing_chunk_id,
            "missing_relative_path": missing_relative_path,
            "missing_line_range": missing_line_range,
            "missing_content": missing_content,
            "missing_chunk_type": missing_chunk_type,
            "missing_file_type_total": missing_file_type_total,
            "missing_file_type_required": missing_file_type_required,
            "file_type_optional_missing": file_type_optional_missing,
            "missing_labels": missing_labels,
            "missing_code_intent": missing_code_intent,
            "missing_qualified_symbol": missing_qualified_symbol,
            "deleted_file_chunks": deleted_file_chunks,
            "label_coverage_percent": round(label_coverage_percent, 2),
            "code_intent_coverage_percent": round(code_intent_coverage_percent, 2),
            "question_use_label_coverage_percent": round(question_use_label_coverage_percent, 2),
            "domain_label_coverage_percent": round(domain_label_coverage_percent, 2),
            "chunk_type_coverage_percent": round(chunk_type_coverage_percent, 2),
            "qualified_symbol_coverage_percent": round(qualified_symbol_coverage_percent, 2),
            "repo_freshness_status": repo_freshness_status,
            "current_commit": current_commit,
            "indexed_commit": indexed_commit,
        }
    }
    return report

def print_health_report(report: dict):
    """Print index health report in a clean format to stdout."""
    print("=" * 40)
    print("            INDEX HEALTH REPORT")
    print("=" * 40)
    if report.get("error"):
        print(f"Error: {report.get('error')}")
        print("=" * 40)
        return
    print(f"Verdict:      {report.get('status')}")
    print(f"Session ID:   {report.get('session_id')}")
    print(f"Repo Root:    {report.get('repo_root')}")
    print(f"Collection:   {report.get('collection')}")
    print(f"Points Count: {report.get('points_count')}")
    print("-" * 40)
    metrics = report.get("metrics", {})
    for k, v in metrics.items():
        if "percent" in k:
            print(f"{k.replace('_', ' ').title():<36}: {v}%")
        else:
            print(f"{k.replace('_', ' ').title():<36}: {v}")
    
    # Print offenders summary
    offenders = report.get("offenders", {})
    has_offenders = any(len(v) > 0 for v in offenders.values())
    if has_offenders:
        print("\n" + "=" * 40)
        print("          OFFENDING CHUNKS SAMPLE")
        print("=" * 40)
        for cat, list_samples in offenders.items():
            if list_samples:
                print(f"\nFailure: {cat.replace('_', ' ').title()} ({len(list_samples)} samples displayed):")
                for s in list_samples[:5]:
                    print(f"  - Chunk: {s['chunk_id']} | Path: {s['relative_path']} | Type: {s['chunk_type']} | reason: {s['reason']}")
    
    # Print stale cleanup guidance (Task 3)
    if metrics.get("deleted_file_chunks", 0) > 0:
        print("\n" + "=" * 40)
        print("    STALE/DELETED FILE CLEANUP GUIDANCE")
        print("=" * 40)
        print("Recommended remediation:")
        print("  - Run a full re-index with collection recreation, or")
        print("  - Run incremental deleted-file cleanup if supported by ingestion.")
        print("\nTo run incremental deleted-file cleanup, execute:")
        print(f"  QDRANT_RECREATE_COLLECTION=0 ./scripts/run_index_cpu_embeddings.sh {report.get('repo_root')} {report.get('collection')}")
        print("=" * 40)

    # Print re-index guidance
    reindex_remediation = report.get("reindex_remediation")
    if reindex_remediation:
        print("\n" + reindex_remediation)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run index health validation.")
    parser.add_argument("--session-id", help="Session ID from database.")
    parser.add_argument("--collection", help="Qdrant collection name.")
    parser.add_argument("--repo-root", help="Repository root directory.")
    args = parser.parse_args()
    
    report = run_index_health_check(
        session_id=args.session_id,
        collection_name=args.collection,
        repo_root=args.repo_root
    )
    print_health_report(report)
    if report.get("status") == "FAIL":
        sys.exit(1)
