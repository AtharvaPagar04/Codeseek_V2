"""Structured logging and lightweight in-process metrics for retrieval."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

    class _NoopMetric:
        def labels(self, **_kwargs):
            return self

        def inc(self, _value: float = 1.0) -> None:
            return None

        def observe(self, _value: float) -> None:
            return None

        def set(self, _value: float) -> None:
            return None

    def Counter(*_args, **_kwargs):  # type: ignore
        return _NoopMetric()

    def Gauge(*_args, **_kwargs):  # type: ignore
        return _NoopMetric()

    def Histogram(*_args, **_kwargs):  # type: ignore
        return _NoopMetric()

    def generate_latest() -> bytes:  # type: ignore
        return b"# prometheus_client not installed\n"


REQUESTS_TOTAL = Counter(
    "codeseek_api_requests_total",
    "Total API requests by path and status",
    ["path", "status"],
)
REQUEST_LATENCY_SECONDS = Histogram(
    "codeseek_api_request_latency_seconds",
    "End-to-end API request latency",
    ["path"],
)
RETRIEVAL_STAGE_LATENCY_SECONDS = Histogram(
    "codeseek_retrieval_stage_latency_seconds",
    "Latency per retrieval stage",
    ["stage"],
)
RETRIEVAL_ERRORS_TOTAL = Counter(
    "codeseek_retrieval_errors_total",
    "Total retrieval errors by type",
    ["error_type"],
)
RETRIEVAL_SOURCES_SELECTED = Gauge(
    "codeseek_retrieval_sources_selected",
    "Sources selected after filtering for latest request",
)
RETRIEVAL_CONTEXT_TOKENS = Gauge(
    "codeseek_retrieval_context_tokens",
    "Context tokens used for latest request",
)

_SENSITIVE_FIELD_MARKERS = (
    "api_key",
    "token",
    "secret",
    "authorization",
    "password",
    "cookie",
    "ciphertext",
)
_BEARER_RE = re.compile(r"bearer\s+[a-z0-9_\-\.]+", re.IGNORECASE)
_URL_CREDS_RE = re.compile(r'([a-zA-Z0-9+.-]+://)([^@/]+)(@)', re.IGNORECASE)


def new_request_id() -> str:
    return uuid.uuid4().hex


def sanitize_credentials_in_string(text: str) -> str:
    if not isinstance(text, str):
        return text
    # 1. Redact bearer token
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    # 2. Redact URL credentials
    def redact_url_match(match):
        scheme = match.group(1)
        creds = match.group(2)
        if ":" in creds:
            return f"{scheme}[redacted]:[redacted]@"
        return f"{scheme}[redacted]@"
    text = _URL_CREDS_RE.sub(redact_url_match, text)
    return text


def sanitize_for_log(value):
    if isinstance(value, dict):
        sanitized = {}
        for key, inner in value.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in _SENSITIVE_FIELD_MARKERS):
                sanitized[key_text] = "[redacted]"
            else:
                sanitized[key_text] = sanitize_for_log(inner)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [sanitize_for_log(item) for item in value]
    if isinstance(value, str):
        redacted = sanitize_credentials_in_string(value)
        if len(redacted) > 512:
            return redacted[:509] + "..."
        return redacted
    return value


def log_event(event: str, request_id: str, **fields) -> None:
    payload = {
        "ts_ms": int(time.time() * 1000),
        "event": event,
        "request_id": request_id,
    }
    payload.update(sanitize_for_log(fields))
    print(json.dumps(payload, ensure_ascii=True))


def render_prometheus_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


def observe_api_request(path: str, status: str, latency_ms: int) -> None:
    REQUESTS_TOTAL.labels(path=path, status=status).inc()
    REQUEST_LATENCY_SECONDS.labels(path=path).observe(max(0.0, latency_ms / 1000.0))


def observe_retrieval_meta(meta: dict, source_count: int, context_tokens: int) -> None:
    stage_latency = meta.get("stage_latency_ms", {}) if isinstance(meta, dict) else {}
    for stage, latency_ms in stage_latency.items():
        RETRIEVAL_STAGE_LATENCY_SECONDS.labels(stage=str(stage)).observe(
            max(0.0, float(latency_ms) / 1000.0)
        )
    RETRIEVAL_SOURCES_SELECTED.set(max(0, int(source_count)))
    RETRIEVAL_CONTEXT_TOKENS.set(max(0, int(context_tokens)))


@dataclass
class StageMetrics:
    request_id: str
    started_at: float = field(default_factory=time.perf_counter)
    stage_latency_ms: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def add_stage(self, name: str, started: float) -> None:
        self.stage_latency_ms[name] = int((time.perf_counter() - started) * 1000)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def total_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)
