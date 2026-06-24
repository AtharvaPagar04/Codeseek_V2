"""Local Ollama model warmup and readiness tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from retrieval.config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_COMPLEX_MODEL,
    LOCAL_LLM_PRIMARY_MODEL,
    LOCAL_LLM_TIMEOUT_SECONDS,
)

OLLAMA_KEEP_ALIVE = "30m"

_state_lock = threading.Lock()
_states: dict[str, "LocalModelState"] = {}


@dataclass
class LocalModelState:
    model: str
    status: str = "idle"
    detail: str = ""
    last_attempt_at: float | None = None
    last_ready_at: float | None = None
    last_error: str = ""
    thread: threading.Thread | None = None
    event: threading.Event = field(default_factory=threading.Event)


def background_prime_primary_model() -> dict[str, Any]:
    """Start warming the default local model without blocking the caller."""
    return warm_model_background(LOCAL_LLM_PRIMARY_MODEL, reason="provider_selected")


def warm_model_background(model: str, *, reason: str = "background") -> dict[str, Any]:
    model = _normalize_model(model)
    if not model:
        return {"model": "", "status": "idle", "detail": "no model selected"}

    with _state_lock:
        state = _states.setdefault(model, LocalModelState(model=model))
        if state.status == "ready" and _is_model_running(model):
            return _snapshot(state)
        if state.thread and state.thread.is_alive():
            state.detail = reason
            return _snapshot(state)
        state.status = "loading"
        state.detail = reason
        state.last_attempt_at = time.time()
        state.last_error = ""
        state.event.clear()
        thread = threading.Thread(target=_warm_model_worker, args=(model,), daemon=True)
        state.thread = thread
        thread.start()
        return _snapshot(state)


def wait_for_model_ready(model: str, *, timeout_seconds: float | None = None, reason: str = "manual") -> dict[str, Any]:
    model = _normalize_model(model)
    if not model:
        return {"model": "", "status": "idle", "detail": "no model selected"}

    snapshot = warm_model_background(model, reason=reason)
    if snapshot.get("status") == "ready":
        return snapshot

    state = _get_state(model)
    timeout = timeout_seconds if timeout_seconds is not None else LOCAL_LLM_TIMEOUT_SECONDS
    if not state.event.wait(timeout):
        raise TimeoutError(f"Local model {model} is still initializing")

    final = get_model_status(model)
    if final.get("status") == "error":
        raise RuntimeError(final.get("detail") or f"Local model {model} failed to load")
    return final


def get_model_status(model: str) -> dict[str, Any]:
    model = _normalize_model(model)
    if not model:
        return {"model": "", "status": "idle", "detail": "no model selected"}

    with _state_lock:
        state = _states.get(model)
        if state and state.status == "ready" and _is_model_running(model):
            return _snapshot(state)
        if state and state.status in {"loading", "ready", "error"}:
            if state.status == "ready" and not _is_model_running(model):
                state.status = "idle"
                state.detail = "model no longer running"
            return _snapshot(state)

    if _is_model_running(model):
        with _state_lock:
            state = _states.setdefault(model, LocalModelState(model=model))
            state.status = "ready"
            state.detail = "model already loaded"
            state.last_ready_at = time.time()
            state.event.set()
            return _snapshot(state)

    return {"model": model, "status": "idle", "detail": "model not loaded"}


def get_provider_runtime_state(provider: str, provider_model: str = "") -> dict[str, Any]:
    provider = (provider or "").strip().lower()
    if provider != "local":
        return {"provider": provider, "status": "ready", "detail": "remote provider"}

    selected = _normalize_model(provider_model) or LOCAL_LLM_PRIMARY_MODEL
    primary = get_model_status(LOCAL_LLM_PRIMARY_MODEL)
    selected_state = get_model_status(selected)
    status = selected_state["status"]
    detail = selected_state.get("detail", "")
    if selected == LOCAL_LLM_PRIMARY_MODEL:
        status = primary["status"]
        detail = primary.get("detail", detail)
    elif status == "idle" and primary["status"] in {"loading", "ready"}:
        status = primary["status"]
        base_detail = primary.get("detail", "").strip()
        if base_detail:
            detail = f"{base_detail}. Selected model {selected} waits on demand."
        else:
            detail = f"Selected model {selected} waits on demand."
    return {
        "provider": "local",
        "selected_model": selected,
        "primary_model": LOCAL_LLM_PRIMARY_MODEL,
        "status": status,
        "detail": detail,
        "selected_status": selected_state["status"],
        "selected_detail": selected_state.get("detail", ""),
        "primary_status": primary["status"],
        "primary_detail": primary.get("detail", ""),
    }


def _get_state(model: str) -> LocalModelState:
    model = _normalize_model(model)
    with _state_lock:
        return _states.setdefault(model, LocalModelState(model=model))


def _snapshot(state: LocalModelState) -> dict[str, Any]:
    return {
        "model": state.model,
        "status": state.status,
        "detail": state.detail,
        "last_attempt_at": state.last_attempt_at,
        "last_ready_at": state.last_ready_at,
        "last_error": state.last_error,
    }


def _normalize_model(model: str) -> str:
    normalized = (model or "").strip()
    if normalized in {"", "default"}:
        return LOCAL_LLM_PRIMARY_MODEL
    if normalized == "auto":
        return LOCAL_LLM_PRIMARY_MODEL
    return normalized


def _warm_model_worker(model: str) -> None:
    state = _get_state(model)
    try:
        _load_model_into_ollama(model)
        with _state_lock:
            state.status = "ready"
            state.detail = "model loaded"
            state.last_ready_at = time.time()
            state.last_error = ""
            state.event.set()
    except Exception as exc:
        with _state_lock:
            state.status = "error"
            state.detail = str(exc)
            state.last_error = str(exc)
            state.event.set()


def _load_model_into_ollama(model: str) -> dict[str, Any]:
    url = f"{_ollama_api_root()}/api/chat"
    response = httpx.post(
        url,
        json={
            "model": model,
            "messages": [],
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        },
        timeout=LOCAL_LLM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _is_model_running(model: str) -> bool:
    try:
        response = httpx.get(f"{_ollama_api_root()}/api/ps", timeout=2.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return False
    models = payload.get("models") if isinstance(payload, dict) else []
    if not isinstance(models, list):
        return False
    for item in models:
        if not isinstance(item, dict):
            continue
        if item.get("model") == model or item.get("name") == model:
            return True
    return False


def _ollama_api_root() -> str:
    base = LOCAL_LLM_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    return base
