"""Manual Portfolio Graph RAG live evaluation.

This script calls a running CodeSeek backend and is intentionally not a pytest
test. It is meant for manual response-quality checks against a known indexed
Portfolio session.

Usage notes:
- Do not source the full backend/.env in your shell if it may override PATH.
- Prefer an explicit API key:
  CODESEEK_API_KEY=... python backend/scripts/evaluate_portfolio_graph_rag.py ...
- Or let the script read only CODESEEK_API_KEY from an env file:
  python backend/scripts/evaluate_portfolio_graph_rag.py --env-file backend/.env
- Print backend routes before evaluating:
  python backend/scripts/evaluate_portfolio_graph_rag.py --print-routes
- Active OFF backend mode:
  CODESEEK_GRAPH_RETRIEVAL_SHADOW=true CODESEEK_GRAPH_RETRIEVAL_ACTIVE=false
- Active ON backend mode:
  CODESEEK_GRAPH_RETRIEVAL_SHADOW=true CODESEEK_GRAPH_RETRIEVAL_ACTIVE=true

Examples:
  python backend/scripts/evaluate_portfolio_graph_rag.py \\
    --base-url http://127.0.0.1:8000 \\
    --query-path /api/v1/query \\
    --env-file backend/.env \\
    --output /tmp/portfolio_graph_rag_eval_off.md \\
    --json-output /tmp/portfolio_graph_rag_eval_off.json

  python backend/scripts/evaluate_portfolio_graph_rag.py \\
    --base-url http://127.0.0.1:8000 \\
    --query-path /query \\
    --env-file backend/.env \\
    --output /tmp/portfolio_graph_rag_eval_off_root.md \\
    --json-output /tmp/portfolio_graph_rag_eval_off_root.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_ID = "7dd4f22e40c848dbba307dda334a3cf2"
DEFAULT_DATASET = "backend/evals/datasets/eval_portfolio_graph_rag.json"
DEFAULT_ENV_FILE = "backend/.env"
DEFAULT_OUTPUT = "/tmp/portfolio_graph_rag_eval_report.md"
DEFAULT_JSON_OUTPUT = "/tmp/portfolio_graph_rag_eval_results.json"
DEFAULT_QUERY_PATH = "/api/v1/query"
RAW_RESPONSE_TEMPLATE = "/tmp/portfolio_graph_rag_{case_id}.json"
QUERY_PATH_404_HINT = (
    "Query endpoint returned 404. Check /openapi.json and try --query-path /query "
    "or --query-path /api/v1/query."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manual live Portfolio Graph RAG response-quality evaluation."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query-path", default=DEFAULT_QUERY_PATH)
    parser.add_argument("--auto-query-path", action="store_true")
    parser.add_argument("--print-routes", action="store_true")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--timeout", type=float, default=90)
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return REPO_ROOT / path


def resolve_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list):
        raise ValueError("Dataset must be a JSON list or an object with a `cases` list.")

    normalized: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case #{index} must be a JSON object.")
        case_id = str(case.get("id") or f"case-{index}").strip()
        query = str(case.get("query") or "").strip()
        if not case_id:
            raise ValueError(f"Case #{index} is missing required field `id`.")
        if not query:
            raise ValueError(f"Case {case_id!r} is missing required field `query`.")
        normalized.append(
            {
                "id": case_id,
                "query": query,
                "expected_answer_notes": str(case.get("expected_answer_notes") or "").strip(),
                "expected_source_paths": unique_strings(case.get("expected_source_paths") or []),
                "required_answer_substrings": unique_strings(
                    case.get("required_answer_substrings") or []
                ),
                "forbidden_answer_substrings": unique_strings(
                    case.get("forbidden_answer_substrings") or []
                ),
                "validation_focus": str(case.get("validation_focus") or "").strip(),
            }
        )
    return normalized


def unique_strings(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    if not isinstance(values, list):
        return result
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def load_api_key(env_file: Path) -> tuple[str, str]:
    env_value = os.getenv("CODESEEK_API_KEY", "").strip()
    if env_value:
        return env_value, "environment"

    file_value = read_key_from_env_file(env_file, "CODESEEK_API_KEY")
    if file_value:
        return file_value, str(env_file)

    raise RuntimeError(
        "CODESEEK_API_KEY is required. Set it in the environment or pass "
        f"--env-file with a file containing CODESEEK_API_KEY. Checked: {env_file}"
    )


def read_key_from_env_file(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    prefix = f"{key}="
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        return unquote_env_value(value)
    return ""


def unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value.split(" #", 1)[0].strip()


def get_openapi_paths(base_url: str, timeout: float = 20) -> dict[str, list[str]]:
    response = requests.get(resolve_url(base_url, "/openapi.json"), timeout=timeout)
    response.raise_for_status()
    body = response.json()
    paths = body.get("paths") if isinstance(body, dict) else {}
    if not isinstance(paths, dict):
        return {}
    result: dict[str, list[str]] = {}
    for path, methods in paths.items():
        if not isinstance(path, str):
            continue
        if isinstance(methods, dict):
            result[path] = sorted(method.upper() for method in methods.keys())
        else:
            result[path] = []
    return result


def print_routes(base_url: str, timeout: float) -> int:
    try:
        paths = get_openapi_paths(base_url, timeout=timeout)
    except Exception as exc:
        print(f"Failed to load OpenAPI routes from {resolve_url(base_url, '/openapi.json')}: {exc}")
        return 1

    interesting_terms = ("query", "health", "sessions")
    matches = [
        (path, methods)
        for path, methods in sorted(paths.items())
        if any(term in path.casefold() for term in interesting_terms)
    ]
    if not matches:
        print("No query, health, or sessions routes found in OpenAPI.")
        return 0

    print(f"Routes from {resolve_url(base_url, '/openapi.json')}:")
    for path, methods in matches:
        method_text = ", ".join(methods) if methods else "methods unavailable"
        print(f"- {path}: {method_text}")
    return 0


def health_preflight(base_url: str, timeout: float) -> dict[str, str]:
    attempts: list[str] = []
    for path in ("/api/v1/health", "/health"):
        url = resolve_url(base_url, path)
        try:
            response = requests.get(url, timeout=min(timeout, 20))
            if response.ok:
                return {"path": path, "url": url, "status_code": str(response.status_code)}
            attempts.append(f"{url} -> HTTP {response.status_code}: {preview(response.text, 200)}")
        except requests.RequestException as exc:
            attempts.append(f"{url} -> {exc.__class__.__name__}: {exc}")
    raise RuntimeError("Backend health preflight failed:\n" + "\n".join(f"- {item}" for item in attempts))


def select_query_path(
    *,
    base_url: str,
    requested_path: str,
    auto_query_path: bool,
    api_key: str,
    timeout: float,
) -> tuple[str, str, dict[str, Any]]:
    normalized_requested = normalize_path(requested_path)
    if not auto_query_path:
        return normalized_requested, resolve_url(base_url, normalized_requested), {
            "mode": "manual",
            "source": "--query-path",
        }

    candidates = unique_strings([normalized_requested, "/query", DEFAULT_QUERY_PATH])
    route_info: dict[str, Any] = {"mode": "auto", "openapi_checked": False, "probes": []}
    try:
        paths = get_openapi_paths(base_url, timeout=min(timeout, 20))
        route_info["openapi_checked"] = True
        route_info["openapi_query_paths"] = [
            path for path in sorted(paths) if "query" in path.casefold()
        ]
        for candidate in candidates:
            if candidate in paths:
                route_info["source"] = "openapi"
                return candidate, resolve_url(base_url, candidate), route_info
    except Exception as exc:
        route_info["openapi_error"] = str(exc)

    for candidate in candidates:
        url = resolve_url(base_url, candidate)
        probe = probe_query_route(url=url, api_key=api_key, timeout=min(timeout, 20))
        route_info["probes"].append(probe)
        if probe["status_code"] != 404:
            route_info["source"] = "probe"
            return candidate, url, route_info

    route_info["source"] = "fallback_to_requested"
    return normalized_requested, resolve_url(base_url, normalized_requested), route_info


def normalize_path(path: str) -> str:
    value = (path or DEFAULT_QUERY_PATH).strip()
    return "/" + value.lstrip("/")


def probe_query_route(*, url: str, api_key: str, timeout: float) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Request-Id": "portfolio-graph-rag-route-probe",
    }
    try:
        response = requests.post(url, headers=headers, json={}, timeout=timeout)
        return {
            "url": url,
            "status_code": response.status_code,
            "body_preview": preview(response.text, 200),
        }
    except requests.RequestException as exc:
        return {
            "url": url,
            "status_code": None,
            "error": str(exc),
        }


def query_backend(
    *,
    query_url: str,
    api_key: str,
    session_id: str,
    case: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Request-Id": f"portfolio-graph-rag-{case['id']}",
    }
    payload = {
        "session_id": session_id,
        "query": case["query"],
        "debug": True,
    }

    try:
        response = requests.post(query_url, headers=headers, json=payload, timeout=timeout)
        body = parse_response_body(response)
        return {
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            "request_url": query_url,
            "response": body,
            "response_preview": response_preview(body),
            "http_error_hint": QUERY_PATH_404_HINT if response.status_code == 404 else "",
        }
    except requests.RequestException as exc:
        body = {
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
        return {
            "ok": False,
            "status_code": None,
            "request_url": query_url,
            "response": body,
            "response_preview": response_preview(body),
            "http_error_hint": "",
        }


def parse_response_body(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw_body": response.text}


def response_preview(body: Any, limit: int = 600) -> str:
    if isinstance(body, str):
        return preview(body, limit)
    try:
        rendered = json.dumps(body, sort_keys=True)
    except TypeError:
        rendered = str(body)
    return preview(rendered, limit)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_case(case: dict[str, Any], raw_result: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    response = raw_result.get("response") if isinstance(raw_result.get("response"), dict) else {}
    http_ok = bool(raw_result.get("ok"))
    answer = str(response.get("answer") or "") if http_ok else ""
    sources = response.get("sources") if http_ok and isinstance(response.get("sources"), list) else []
    diagnostics = response.get("diagnostics") if http_ok else None
    if http_ok and not isinstance(diagnostics, dict):
        diagnostics = response.get("meta") if isinstance(response.get("meta"), dict) else None

    actual_sources = unique_paths(extract_paths(sources))
    expected_sources = list(case["expected_source_paths"])
    missing_sources = [
        expected for expected in expected_sources if not path_present(expected, actual_sources)
    ]

    missing_required = [
        needle
        for needle in case["required_answer_substrings"]
        if not contains_case_insensitive(answer, needle)
    ]
    forbidden_found = [
        needle
        for needle in case["forbidden_answer_substrings"]
        if contains_case_insensitive(answer, needle)
    ]

    alignment_value = extract_source_alignment(diagnostics)
    diagnostics_available = diagnostics is not None
    source_alignment_ok = alignment_value is True or (
        alignment_value is None and not diagnostics_available
    )

    graph_shadow = diagnostics.get("graph_shadow") if isinstance(diagnostics, dict) else None
    graph_active = diagnostics.get("graph_active") if isinstance(diagnostics, dict) else None
    graph_active_added_paths = extract_graph_active_added_paths(graph_active)
    graph_active_added_count = extract_int(graph_active, "added_count")
    graph_active_enabled = extract_value(graph_active, "enabled")
    graph_active_reason = extract_value(graph_active, "reason")
    graph_shadow_status = extract_value(graph_shadow, "status")

    expected_sources_present = not missing_sources
    required_substrings_present = not missing_required
    forbidden_substrings_absent = not forbidden_found

    if not http_ok:
        verdict = "FAIL"
    elif (
        expected_sources_present
        and required_substrings_present
        and forbidden_substrings_absent
        and source_alignment_ok
    ):
        verdict = "PASS"
    elif expected_sources_present and missing_required and forbidden_substrings_absent and source_alignment_ok:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "id": case["id"],
        "query": case["query"],
        "validation_focus": case["validation_focus"],
        "verdict": verdict,
        "http_ok": http_ok,
        "http_status_code": raw_result.get("status_code"),
        "request_url": raw_result.get("request_url"),
        "response_preview": raw_result.get("response_preview") or "",
        "http_error_hint": raw_result.get("http_error_hint") or "",
        "expected_sources": expected_sources,
        "actual_sources": actual_sources,
        "missing_sources": missing_sources,
        "required_substrings": list(case["required_answer_substrings"]),
        "missing_required_substrings": missing_required,
        "forbidden_substrings": list(case["forbidden_answer_substrings"]),
        "forbidden_substrings_found": forbidden_found,
        "expected_sources_present": expected_sources_present,
        "required_substrings_present": required_substrings_present,
        "forbidden_substrings_absent": forbidden_substrings_absent,
        "source_alignment": alignment_value,
        "source_alignment_ok": source_alignment_ok,
        "graph_shadow_status": graph_shadow_status,
        "graph_shadow_ready": graph_shadow_status == "ready",
        "graph_active_enabled": graph_active_enabled,
        "graph_active_reason": graph_active_reason,
        "graph_active_added_count": graph_active_added_count,
        "graph_active_added_paths": graph_active_added_paths,
        "answer_preview": preview(answer),
        "raw_response_path": str(raw_path),
    }


def extract_paths(items: list[Any]) -> list[str]:
    paths: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        path = first_present_string(
            item,
            ("relative_path", "path", "file_path", "source_path", "filepath"),
        )
        if not path:
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                path = first_present_string(
                    metadata,
                    ("relative_path", "path", "file_path", "source_path", "filepath"),
                )
        if path:
            paths.append(path)
    return paths


def extract_graph_active_added_paths(graph_active: Any) -> list[str]:
    if not isinstance(graph_active, dict):
        return []
    added_chunks = graph_active.get("added_chunks")
    if not isinstance(added_chunks, list):
        return []
    return unique_paths(extract_paths(added_chunks))


def extract_source_alignment(diagnostics: Any) -> bool | None:
    if not isinstance(diagnostics, dict):
        return None
    source_alignment = diagnostics.get("source_alignment")
    if not isinstance(source_alignment, dict):
        return None
    aligned = source_alignment.get("aligned")
    return aligned if isinstance(aligned, bool) else None


def extract_value(data: Any, key: str) -> Any:
    if not isinstance(data, dict):
        return None
    return data.get(key)


def extract_int(data: Any, key: str) -> int:
    value = extract_value(data, key)
    return value if isinstance(value, int) else 0


def first_present_string(data: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def unique_paths(paths: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = canonical_path(path)
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def canonical_path(path: str) -> str:
    value = path.strip().replace("\\", "/")
    if value.startswith("file://"):
        value = value.removeprefix("file://")
    value = value.removeprefix("./")
    match = re.match(r"^(.+\.(?:py|js|jsx|ts|tsx|json|md|css|html|ya?ml))(?::\d+)?$", value)
    if match:
        value = match.group(1)
    return value


def path_present(expected: str, actual_paths: list[str]) -> bool:
    expected_path = canonical_path(expected)
    for actual in actual_paths:
        actual_path = canonical_path(actual)
        if actual_path == expected_path or actual_path.endswith(f"/{expected_path}"):
            return True
    return False


def contains_case_insensitive(haystack: str, needle: str) -> bool:
    return needle.casefold() in haystack.casefold()


def preview(answer: str, limit: int = 600) -> str:
    collapsed = re.sub(r"\s+", " ", answer).strip()
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = sum(1 for result in results if result["verdict"] == "PASS")
    partial_count = sum(1 for result in results if result["verdict"] == "PARTIAL")
    fail_count = sum(1 for result in results if result["verdict"] == "FAIL")
    source_alignment_failures = sum(
        1 for result in results if result["source_alignment_ok"] is False
    )
    graph_shadow_ready_count = sum(1 for result in results if result["graph_shadow_ready"])
    graph_active_added_count_total = sum(
        int(result.get("graph_active_added_count") or 0) for result in results
    )
    cases_with_graph_active_paths = [
        {
            "id": result["id"],
            "added_paths": result["graph_active_added_paths"],
        }
        for result in results
        if result["graph_active_added_paths"]
    ]
    return {
        "total": len(results),
        "pass": pass_count,
        "partial": partial_count,
        "fail": fail_count,
        "source_alignment_failures": source_alignment_failures,
        "graph_shadow_ready_count": graph_shadow_ready_count,
        "graph_active_added_count_total": graph_active_added_count_total,
        "cases_where_graph_active_added_paths": cases_with_graph_active_paths,
        "graph_shadow_statuses": dict(
            Counter(str(result["graph_shadow_status"]) for result in results)
        ),
        "graph_active_enabled_values": dict(
            Counter(str(result["graph_active_enabled"]) for result in results)
        ),
        "graph_active_reasons": dict(
            Counter(str(result["graph_active_reason"]) for result in results)
        ),
    }


def render_markdown(
    *,
    runtime: dict[str, Any],
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Portfolio Graph RAG Manual Eval",
        "",
        "## Runtime",
        f"- base_url: `{runtime['base_url']}`",
        f"- query_path: `{runtime['query_path']}`",
        f"- resolved_query_url: `{runtime['resolved_query_url']}`",
        f"- health endpoint used: `{runtime['health_endpoint_used']}`",
        f"- session_id: `{runtime['session_id']}`",
        f"- dataset: `{runtime['dataset']}`",
        f"- markdown output: `{runtime['output']}`",
        f"- json output: `{runtime['json_output']}`",
        f"- raw response template: `{RAW_RESPONSE_TEMPLATE}`",
        f"- graph_shadow statuses observed: {format_mapping(summary['graph_shadow_statuses'])}",
        f"- graph_active enabled values observed: {format_mapping(summary['graph_active_enabled_values'])}",
        f"- graph_active reasons observed: {format_mapping(summary['graph_active_reasons'])}",
        "",
        "## Summary",
        f"- total: {summary['total']}",
        f"- pass: {summary['pass']}",
        f"- partial: {summary['partial']}",
        f"- fail: {summary['fail']}",
        f"- source alignment failures: {summary['source_alignment_failures']}",
        f"- graph shadow ready count: {summary['graph_shadow_ready_count']}",
        f"- graph active added count total: {summary['graph_active_added_count_total']}",
        "- cases where graph_active added paths:",
    ]
    graph_added_cases = summary["cases_where_graph_active_added_paths"]
    if graph_added_cases:
        for item in graph_added_cases:
            lines.append(f"  - {item['id']}: {format_list(item['added_paths'])}")
    else:
        lines.append("  - none")

    lines.extend(["", "## Case results"])
    for result in results:
        lines.extend(
            [
                "",
                f"### {result['id']}",
                f"- query: {result['query']}",
                f"- verdict: {result['verdict']}",
                f"- expected sources: {format_list(result['expected_sources'])}",
                f"- actual sources: {format_list(result['actual_sources'])}",
                f"- missing sources: {format_list(result['missing_sources'])}",
                "- missing required substrings: "
                f"{format_list(result['missing_required_substrings'])}",
                "- forbidden substrings found: "
                f"{format_list(result['forbidden_substrings_found'])}",
                f"- source_alignment: {result['source_alignment']}",
                f"- graph_shadow.status: {result['graph_shadow_status']}",
                f"- graph_active.enabled: {result['graph_active_enabled']}",
                f"- graph_active.reason: {result['graph_active_reason']}",
                f"- graph_active.added_count: {result['graph_active_added_count']}",
                f"- graph_active.added_paths: {format_list(result['graph_active_added_paths'])}",
                f"- answer preview: {result['answer_preview']}",
                f"- raw response path: `{result['raw_response_path']}`",
            ]
        )
        if not result["http_ok"]:
            lines.extend(
                [
                    f"- status code: {result['http_status_code']}",
                    f"- request URL: `{result['request_url']}`",
                    f"- response preview: {result['response_preview']}",
                ]
            )
            if result["http_error_hint"]:
                lines.append(f"- endpoint hint: {result['http_error_hint']}")
    return "\n".join(lines) + "\n"


def format_list(values: list[Any]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{value}`" for value in values)


def format_mapping(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{key}`={value}" for key, value in sorted(values.items()))


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    if args.print_routes:
        return print_routes(base_url, args.timeout)

    env_file = resolve_path(args.env_file)
    try:
        api_key, api_key_source = load_api_key(env_file)
        health = health_preflight(base_url, args.timeout)
        query_path, query_url, route_info = select_query_path(
            base_url=base_url,
            requested_path=args.query_path,
            auto_query_path=args.auto_query_path,
            api_key=api_key,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"Portfolio Graph RAG eval preflight failed: {exc}", file=sys.stderr)
        return 2

    dataset_path = resolve_path(args.dataset)
    cases = load_dataset(dataset_path)

    print(f"Health endpoint: {health['url']} ({health['status_code']})")
    print(f"Query URL: {query_url}")
    print(f"API key source: {api_key_source}")
    if args.auto_query_path:
        print(f"Query path auto-selection: {route_info}")

    results: list[dict[str, Any]] = []
    for case in cases:
        raw_path = Path(RAW_RESPONSE_TEMPLATE.format(case_id=case["id"]))
        raw_result = query_backend(
            query_url=query_url,
            api_key=api_key,
            session_id=args.session_id,
            case=case,
            timeout=args.timeout,
        )
        write_json(raw_path, raw_result)
        results.append(evaluate_case(case, raw_result, raw_path))

    summary = build_summary(results)
    runtime = {
        "base_url": base_url,
        "query_path": query_path,
        "resolved_query_url": query_url,
        "health_endpoint_used": health["path"],
        "health_url": health["url"],
        "session_id": args.session_id,
        "dataset": str(dataset_path),
        "output": args.output,
        "json_output": args.json_output,
        "api_key_source": api_key_source,
        "auto_query_path": bool(args.auto_query_path),
        "route_info": route_info,
    }
    json_report = {
        "runtime": runtime,
        "summary": summary,
        "results": results,
    }
    write_json(Path(args.json_output), json_report)

    markdown = render_markdown(runtime=runtime, results=results, summary=summary)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Wrote Markdown report: {output_path}")
    print(f"Wrote JSON results: {args.json_output}")
    return 0 if summary["partial"] == 0 and summary["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
