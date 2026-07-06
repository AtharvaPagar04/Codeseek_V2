"""Evaluate shadow graph retrieval diagnostics against query expectations."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Callable, Iterable

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


QueryRunner = Callable[[str, str, int], dict]


@dataclass(frozen=True)
class EvaluationSession:
    session_id: str
    repo_root: str
    collection: str
    session_status: str
    graph_status: str | None
    graph_error: str = ""
    collection_points: int = 0


class EvaluationSessionValidationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        session_status: str | None = None,
        graph_status: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.session_status = session_status
        self.graph_status = graph_status


def load_query_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list):
        raise ValueError("Query file must be a JSON list or an object with a `cases` list.")

    normalized: list[dict] = []
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Query case #{index} must be an object.")
        query = str(case.get("query") or "").strip()
        if not query:
            raise ValueError(f"Query case #{index} is missing required field `query`.")
        normalized.append(
            {
                "id": str(case.get("id") or f"q{index}"),
                "query": query,
                "expected_files": _unique_strings(case.get("expected_files") or []),
                "expected_symbols": _unique_strings(case.get("expected_symbols") or []),
                "notes": str(case.get("notes") or "").strip(),
            }
        )
    return normalized


def evaluate_cases(
    cases: list[dict],
    *,
    session_id: str,
    top_k: int = 10,
    max_queries: int | None = None,
    runner: QueryRunner | None = None,
    allow_missing_graph: bool = False,
    validate_session: bool | None = None,
) -> dict:
    should_validate = (runner is None) if validate_session is None else validate_session
    if should_validate:
        try:
            load_evaluation_session(session_id, allow_missing_graph=allow_missing_graph)
        except EvaluationSessionValidationError as exc:
            return build_session_error_report(session_id, top_k=top_k, error=exc)

    selected_cases = cases[: max_queries or len(cases)]
    runner = runner or run_retrieval_shadow_query
    results: list[dict] = []

    with _temporary_env({"CODESEEK_GRAPH_RETRIEVAL_SHADOW": "true"}):
        for case in selected_cases:
            try:
                query_result = runner(session_id, case["query"], top_k)
                results.append(build_query_result(case, query_result, top_k=top_k))
            except Exception as exc:
                results.append(build_query_error_result(case, str(exc), top_k=top_k))

    summary = build_summary(results)
    return {
        "session_id": session_id,
        "top_k": top_k,
        "summary": summary,
        "results": results,
    }


def run_retrieval_shadow_query(session_id: str, query: str, top_k: int) -> dict:
    """Run retrieval and graph shadow expansion without generating an answer."""
    session = load_evaluation_session(session_id, allow_missing_graph=True)
    previous_repo_root = os.getenv("RETRIEVAL_REPO_ROOT", "")
    previous_collection = os.getenv("QDRANT_COLLECTION_NAME", "")
    try:
        os.environ["RETRIEVAL_REPO_ROOT"] = session.repo_root
        os.environ["QDRANT_COLLECTION_NAME"] = session.collection

        from retrieval.query.query_processor import process_query
        from retrieval.search.searcher import search
        from retrieval.graph.retrieval import run_graph_shadow_retrieval

        query_info = process_query(query)
        candidates = search(query_info)
        graph_shadow = run_graph_shadow_retrieval(session_id, candidates, query=query)
        return {
            "normal_candidates": candidates[:top_k],
            "graph_shadow": graph_shadow,
        }
    finally:
        _restore_env("RETRIEVAL_REPO_ROOT", previous_repo_root)
        _restore_env("QDRANT_COLLECTION_NAME", previous_collection)


def build_query_result(case: dict, query_result: dict, *, top_k: int = 10) -> dict:
    normal_candidates = list(query_result.get("normal_candidates") or [])[:top_k]
    graph_shadow = query_result.get("graph_shadow") if isinstance(query_result.get("graph_shadow"), dict) else {}
    candidate_chunks = list(graph_shadow.get("candidate_chunks") or [])
    external_packages = list(graph_shadow.get("external_packages") or [])
    unresolved_imports = list(graph_shadow.get("unresolved_imports") or [])
    graph_status = str(graph_shadow.get("status") or "").strip()

    normal_top_files = _unique_paths(item.get("relative_path") for item in normal_candidates)
    normal_top_symbols = _unique_strings(item.get("symbol_name") for item in normal_candidates if item.get("symbol_name"))
    graph_shadow_files = _unique_paths(item.get("relative_path") for item in candidate_chunks)
    graph_shadow_symbols = _unique_strings(item.get("symbol_name") for item in candidate_chunks if item.get("symbol_name"))
    graph_new_files = [path for path in graph_shadow_files if path not in normal_top_files]
    graph_new_symbols = [symbol for symbol in graph_shadow_symbols if symbol not in normal_top_symbols]
    expected_files = _unique_paths(case.get("expected_files") or [])
    expected_symbols = _unique_strings(case.get("expected_symbols") or [])
    graph_noisy_files = [path for path in graph_new_files if path not in expected_files]
    expected_file_hit_normal = _intersection(expected_files, normal_top_files)
    expected_file_hit_graph_shadow = _intersection(expected_files, graph_shadow_files)
    graph_added_expected_file = [path for path in expected_file_hit_graph_shadow if path not in expected_file_hit_normal]
    expected_symbol_hit_normal = _intersection_ci(expected_symbols, normal_top_symbols)
    expected_symbol_hit_graph_shadow = _intersection_ci(expected_symbols, graph_shadow_symbols)
    graph_added_expected_symbol = [
        symbol for symbol in expected_symbol_hit_graph_shadow
        if _norm(symbol) not in {_norm(item) for item in expected_symbol_hit_normal}
    ]

    if graph_status and graph_status != "ready" and not candidate_chunks:
        classification = "no_graph_available"
    else:
        classification = classify_query_result(
            graph_candidate_count=len(candidate_chunks),
            graph_added_expected_file=bool(graph_added_expected_file),
            graph_added_expected_symbol=bool(graph_added_expected_symbol),
            graph_new_files=graph_new_files,
            unresolved_import_count=len(unresolved_imports),
        )

    return {
        "id": case.get("id", ""),
        "query": case["query"],
        "notes": case.get("notes", ""),
        "status": graph_status,
        "classification": classification,
        "normal_top_candidates": [compact_candidate(item) for item in normal_candidates],
        "normal_top_files": normal_top_files,
        "normal_top_symbols": normal_top_symbols,
        "graph_shadow_files": graph_shadow_files,
        "graph_shadow_symbols": graph_shadow_symbols,
        "graph_new_files": graph_new_files,
        "graph_new_symbols": graph_new_symbols,
        "graph_noisy_files": graph_noisy_files,
        "expected_files": expected_files,
        "expected_symbols": expected_symbols,
        "expected_file_hit_normal": expected_file_hit_normal,
        "expected_file_hit_graph_shadow": expected_file_hit_graph_shadow,
        "graph_added_expected_file": graph_added_expected_file,
        "expected_symbol_hit_normal": expected_symbol_hit_normal,
        "expected_symbol_hit_graph_shadow": expected_symbol_hit_graph_shadow,
        "graph_added_expected_symbol": graph_added_expected_symbol,
        "graph_candidate_count": len(candidate_chunks),
        "graph_external_package_count": len(external_packages),
        "graph_unresolved_import_count": len(unresolved_imports),
        "graph_noisy_file_count": len(graph_noisy_files),
        "graph_candidate_details": [compact_graph_candidate(item) for item in candidate_chunks],
        "graph_shadow": {
            "enabled": bool(graph_shadow.get("enabled", False)),
            "status": graph_shadow.get("status", ""),
            "anchors": list(graph_shadow.get("anchors") or []),
            "expanded_nodes": list(graph_shadow.get("expanded_nodes") or []),
            "candidate_chunks": candidate_chunks,
            "unresolved_imports": unresolved_imports,
            "external_packages": external_packages,
            "overlap": dict(graph_shadow.get("overlap") or {}),
            "stats": dict(graph_shadow.get("stats") or {}),
        },
    }


def build_query_error_result(case: dict, error: str, *, top_k: int = 10) -> dict:
    return {
        "id": case.get("id", ""),
        "query": case.get("query", ""),
        "notes": case.get("notes", ""),
        "status": "error",
        "classification": "error",
        "error": error,
        "normal_top_candidates": [],
        "normal_top_files": [],
        "normal_top_symbols": [],
        "graph_shadow_files": [],
        "graph_shadow_symbols": [],
        "graph_new_files": [],
        "graph_new_symbols": [],
        "graph_noisy_files": [],
        "expected_files": _unique_paths(case.get("expected_files") or []),
        "expected_symbols": _unique_strings(case.get("expected_symbols") or []),
        "expected_file_hit_normal": [],
        "expected_file_hit_graph_shadow": [],
        "graph_added_expected_file": [],
        "expected_symbol_hit_normal": [],
        "expected_symbol_hit_graph_shadow": [],
        "graph_added_expected_symbol": [],
        "graph_candidate_count": 0,
        "graph_external_package_count": 0,
        "graph_unresolved_import_count": 0,
        "graph_noisy_file_count": 0,
        "graph_candidate_details": [],
        "graph_shadow": {
            "enabled": True,
            "status": "error",
            "anchors": [],
            "expanded_nodes": [],
            "candidate_chunks": [],
            "unresolved_imports": [],
            "external_packages": [],
            "overlap": {},
            "stats": {"top_k": top_k},
        },
    }


def build_session_error_report(
    session_id: str,
    *,
    top_k: int,
    error: EvaluationSessionValidationError,
) -> dict:
    return {
        "session_id": session_id,
        "top_k": top_k,
        "summary": _empty_summary(),
        "session_error": {
            "status": "failed_validation",
            "message": error.message,
            "session_status": error.session_status,
            "graph_status": error.graph_status,
        },
        "results": [],
    }


def build_summary(results: list[dict]) -> dict:
    total_queries = len(results)
    graph_candidate_counts = [int(item.get("graph_candidate_count", 0) or 0) for item in results]
    unresolved_counts = [int(item.get("graph_unresolved_import_count", 0) or 0) for item in results]
    added_files = Counter(
        path
        for item in results
        for path in item.get("graph_new_files", [])
    )
    noisy_files = Counter(
        path
        for item in results
        for path in item.get("graph_noisy_files", [])
    )
    unresolved_imports = Counter(
        str(unresolved.get("raw_reference") or "").strip()
        for item in results
        for unresolved in (item.get("graph_shadow", {}).get("unresolved_imports") or [])
        if str(unresolved.get("raw_reference") or "").strip()
    )
    classifications = Counter(str(item.get("classification") or "unknown") for item in results)

    return {
        "total_queries": total_queries,
        "queries_with_graph_candidates": sum(1 for count in graph_candidate_counts if count > 0),
        "queries_where_graph_added_expected_file": sum(1 for item in results if item.get("graph_added_expected_file")),
        "queries_where_graph_added_expected_symbol": sum(1 for item in results if item.get("graph_added_expected_symbol")),
        "queries_where_graph_added_new_file": sum(1 for item in results if item.get("graph_new_files")),
        "queries_where_graph_added_noisy_file": sum(1 for item in results if item.get("graph_noisy_files")),
        "average_graph_candidate_count": _average(graph_candidate_counts),
        "average_graph_noisy_file_count": _average([int(item.get("graph_noisy_file_count", 0) or 0) for item in results]),
        "average_unresolved_import_count": _average(unresolved_counts),
        "top_added_files": [{"file": file_path, "count": count} for file_path, count in added_files.most_common(10)],
        "top_noisy_files": [{"file": file_path, "count": count} for file_path, count in noisy_files.most_common(10)],
        "top_unresolved_imports": [
            {"raw_reference": raw_reference, "count": count}
            for raw_reference, count in unresolved_imports.most_common(10)
        ],
        "classification_counts": dict(classifications),
    }


def _empty_summary() -> dict:
    return {
        "total_queries": 0,
        "queries_with_graph_candidates": 0,
        "queries_where_graph_added_expected_file": 0,
        "queries_where_graph_added_expected_symbol": 0,
        "queries_where_graph_added_new_file": 0,
        "queries_where_graph_added_noisy_file": 0,
        "average_graph_candidate_count": 0.0,
        "average_graph_noisy_file_count": 0.0,
        "average_unresolved_import_count": 0.0,
        "top_added_files": [],
        "top_noisy_files": [],
        "top_unresolved_imports": [],
        "classification_counts": {},
    }


def classify_query_result(
    *,
    graph_candidate_count: int,
    graph_added_expected_file: bool,
    graph_added_expected_symbol: bool,
    graph_new_files: list[str],
    unresolved_import_count: int,
) -> str:
    if graph_added_expected_file or graph_added_expected_symbol:
        return "graph_helped"
    if unresolved_import_count > 0 and not graph_new_files:
        return "graph_added_noisy_unresolved_context"
    if graph_candidate_count <= 0:
        return "no_graph_candidates"
    return "graph_added_neutral_context"


def compact_candidate(item: dict) -> dict:
    compact: dict[str, object] = {}
    for key in (
        "chunk_id",
        "relative_path",
        "symbol_name",
        "qualified_symbol",
        "chunk_type",
        "retrieval_source",
        "support_kind",
    ):
        value = item.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    for key in ("start_line", "end_line"):
        try:
            value = int(item.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            compact[key] = value
    for key in ("retrieval_score", "fusion_score", "final_score"):
        if key in item:
            try:
                compact[key] = round(float(item.get(key) or 0.0), 4)
            except (TypeError, ValueError):
                pass
    return compact


def compact_graph_candidate(item: dict) -> dict:
    compact = compact_candidate(item)
    for key in ("candidate_score", "selection_rank", "anchor_rank", "hub_import_count"):
        if key not in item:
            continue
        try:
            compact[key] = round(float(item.get(key) or 0.0), 4)
        except (TypeError, ValueError):
            compact[key] = item.get(key)
    for key in ("expansion_reason", "edge_type", "direction", "selection_reason"):
        value = item.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    reasons = item.get("score_reasons")
    if isinstance(reasons, list):
        compact["score_reasons"] = [str(reason) for reason in reasons if str(reason or "").strip()]
    return compact


def render_markdown_report(report: dict) -> str:
    if report.get("session_error"):
        return _render_session_error_markdown(report)

    summary = report.get("summary", {})
    lines = [
        "# Graph Shadow Evaluation Report",
        "",
        "## Summary",
        f"- Total queries: {summary.get('total_queries', 0)}",
        f"- Queries with graph candidates: {summary.get('queries_with_graph_candidates', 0)}",
        f"- Queries where graph added expected file: {summary.get('queries_where_graph_added_expected_file', 0)}",
        f"- Average graph candidate count: {_format_float(summary.get('average_graph_candidate_count', 0.0))}",
        f"- Average noisy files added: {_format_float(summary.get('average_graph_noisy_file_count', 0.0))}",
        f"- Average unresolved imports: {_format_float(summary.get('average_unresolved_import_count', 0.0))}",
        "",
        "## Query Details",
    ]
    for result in report.get("results", []):
        lines.extend(
            [
                "",
                f"### {result.get('id') or 'query'}",
                f"- Query: {result.get('query', '')}",
                f"- Status: {result.get('status', '')}",
                f"- Classification: {result.get('classification', '')}",
                f"- Normal top files: {_format_list(result.get('normal_top_files', []))}",
                f"- Graph shadow added files: {_format_list(result.get('graph_new_files', []))}",
                f"- Graph shadow noisy files: {_format_list(result.get('graph_noisy_files', []))}",
                f"- Graph candidate scores: {_format_graph_candidate_scores(result)}",
                f"- Expected files hit by normal retrieval: {_format_list(result.get('expected_file_hit_normal', []))}",
                f"- Expected files added by graph shadow: {_format_list(result.get('graph_added_expected_file', []))}",
                f"- External packages: {_format_external_packages(result)}",
                f"- Unresolved imports: {_format_unresolved_imports(result)}",
            ]
        )
        notes = str(result.get("notes") or "").strip()
        if notes:
            lines.append(f"- Notes: {notes}")
        error = str(result.get("error") or "").strip()
        if error:
            lines.append(f"- Error: {error}")
    lines.append("")
    return "\n".join(lines)


def _render_session_error_markdown(report: dict) -> str:
    session_error = report.get("session_error", {})
    lines = [
        "# Graph Shadow Evaluation Report",
        "",
        "## Session Error",
        f"- Session: {report.get('session_id', '')}",
        f"- Error: {session_error.get('message', '')}",
        f"- Session status: {session_error.get('session_status') or 'none'}",
        f"- Graph status: {session_error.get('graph_status') or 'none'}",
        "",
    ]
    return "\n".join(lines)


def write_json_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_report(report), encoding="utf-8")


def load_evaluation_session(
    session_id: str,
    *,
    allow_missing_graph: bool = False,
    collection_point_count: Callable[[str], int] | None = None,
) -> EvaluationSession:
    session = _load_session(session_id)
    if not session:
        raise EvaluationSessionValidationError(f"Session not found: {session_id}")

    session_status = str(session.get("status") or "").strip()
    if session_status != "ready":
        raise EvaluationSessionValidationError(
            f"Session is not ready: {session_status or 'unknown'}",
            session_status=session_status or None,
        )

    repo_root = str(session.get("repo_root") or "").strip()
    if not repo_root:
        raise EvaluationSessionValidationError(
            "Session has no repo_root",
            session_status=session_status,
        )
    if not Path(repo_root).exists():
        raise EvaluationSessionValidationError(
            f"Session repo_root does not exist: {repo_root}",
            session_status=session_status,
        )

    collection = str(session.get("collection") or "").strip()
    if not collection:
        raise EvaluationSessionValidationError(
            "Session has no collection",
            session_status=session_status,
        )

    point_counter = collection_point_count or _collection_point_count
    collection_points = point_counter(collection)
    if collection_points <= 0:
        raise EvaluationSessionValidationError(
            f"Session collection is empty or unavailable: {collection}",
            session_status=session_status,
        )

    graph_status_row = _load_graph_status(session_id)
    graph_status = str(graph_status_row.get("status") or "not_built").strip() or "not_built"
    graph_error = str(graph_status_row.get("error") or "").strip()
    if graph_status == "not_built":
        if not allow_missing_graph:
            raise EvaluationSessionValidationError(
                "Graph is not built for this session",
                session_status=session_status,
                graph_status=graph_status,
            )
    elif graph_status == "failed":
        message = "Graph build failed for this session"
        if graph_error:
            message = f"{message}: {graph_error}"
        raise EvaluationSessionValidationError(
            message,
            session_status=session_status,
            graph_status=graph_status,
        )
    elif graph_status != "ready":
        raise EvaluationSessionValidationError(
            f"Graph is not ready for this session: {graph_status}",
            session_status=session_status,
            graph_status=graph_status,
        )

    return EvaluationSession(
        session_id=session_id,
        repo_root=repo_root,
        collection=collection,
        session_status=session_status,
        graph_status=graph_status,
        graph_error=graph_error,
        collection_points=collection_points,
    )


def _load_session(session_id: str) -> dict:
    from retrieval.db import db_cursor

    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT id, repo_root, collection, status
            FROM repo_sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        return dict(row) if row else {}


def _load_graph_status(session_id: str) -> dict:
    from retrieval.db import db_cursor

    with db_cursor() as (_conn, cursor):
        row = cursor.execute(
            """
            SELECT session_id, status, error
            FROM code_graph_builds
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row:
            return dict(row)
        return {
            "session_id": session_id,
            "status": "not_built",
            "error": "",
        }


def _collection_point_count(collection: str) -> int:
    try:
        from retrieval.support.qdrant_config import create_qdrant_client

        client = create_qdrant_client(timeout=5.0, check_compatibility=False)
        info = client.get_collection(collection)
    except Exception:
        return 0

    for field in ("points_count", "vectors_count", "indexed_vectors_count"):
        value = getattr(info, field, None)
        if value:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


@contextmanager
def _temporary_env(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            _restore_env(key, value)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _unique_paths(values: Iterable[object]) -> list[str]:
    return _unique_strings(str(value or "").strip().replace("\\", "/").strip("/") for value in values)


def _unique_strings(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = _norm(text)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def _intersection(expected: list[str], actual: list[str]) -> list[str]:
    actual_set = {item.strip() for item in actual}
    return [item for item in expected if item.strip() in actual_set]


def _intersection_ci(expected: list[str], actual: list[str]) -> list[str]:
    actual_map = {_norm(item): item for item in actual}
    return [item for item in expected if _norm(item) in actual_map]


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


def _average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _format_list(values: list[object]) -> str:
    cleaned = [str(value) for value in values if str(value or "").strip()]
    return ", ".join(cleaned) if cleaned else "none"


def _format_external_packages(result: dict) -> str:
    packages = [
        str(item.get("name") or item.get("qualified_name") or "").strip()
        for item in result.get("graph_shadow", {}).get("external_packages", [])
        if str(item.get("name") or item.get("qualified_name") or "").strip()
    ]
    return _format_list(_unique_strings(packages))


def _format_unresolved_imports(result: dict) -> str:
    unresolved = [
        str(item.get("raw_reference") or "").strip()
        for item in result.get("graph_shadow", {}).get("unresolved_imports", [])
        if str(item.get("raw_reference") or "").strip()
    ]
    return _format_list(_unique_strings(unresolved))


def _format_graph_candidate_scores(result: dict) -> str:
    formatted: list[str] = []
    for item in list(result.get("graph_candidate_details") or [])[:6]:
        path = str(item.get("relative_path") or item.get("chunk_id") or "").strip()
        if not path:
            continue
        score = item.get("candidate_score")
        reasons = item.get("score_reasons") if isinstance(item.get("score_reasons"), list) else []
        reason_text = ",".join(str(reason) for reason in reasons[:3] if str(reason or "").strip())
        if reason_text:
            formatted.append(f"{path} ({score}: {reason_text})")
        else:
            formatted.append(f"{path} ({score})")
    return _format_list(formatted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate graph shadow retrieval diagnostics.")
    parser.add_argument("--session-id", required=True, help="Repo session id to evaluate.")
    parser.add_argument("--queries-file", required=True, type=Path, help="JSON file containing query cases.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report output path.")
    parser.add_argument("--output-md", type=Path, default=None, help="Optional Markdown report output path.")
    parser.add_argument("--top-k", type=int, default=10, help="Normal retrieval top-k candidates to compare.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional maximum number of queries.")
    parser.add_argument(
        "--allow-missing-graph",
        action="store_true",
        help="Continue when the graph has not been built; query results will report no graph availability.",
    )
    parser.add_argument("--api-base", default=None, help="Reserved for API-based evaluation; currently unused.")
    parser.add_argument("--api-key", default=None, help="Reserved for API-based evaluation; currently unused.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.api_base or args.api_key:
        raise SystemExit("--api-base and --api-key are reserved; this evaluator uses backend functions directly.")

    cases = load_query_cases(args.queries_file)
    report = evaluate_cases(
        cases,
        session_id=args.session_id,
        top_k=max(1, int(args.top_k or 10)),
        max_queries=args.max_queries,
        allow_missing_graph=args.allow_missing_graph,
    )

    if args.output_json:
        write_json_report(report, args.output_json)
    if args.output_md:
        write_markdown_report(report, args.output_md)
    if not args.output_json and not args.output_md:
        print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("session_error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
