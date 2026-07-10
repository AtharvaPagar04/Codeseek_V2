"""In-memory indexing event bus for live progress tracking.

Events are stored per session in a bounded ring buffer. Subscribers
receive events through thread-safe queues so the SSE endpoint can
stream them without polling.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Generator


_lock = threading.Lock()
_events_by_session: dict[str, list[dict]] = {}
_next_id_by_session: dict[str, int] = {}
_subscribers_by_session: dict[str, list[queue.Queue]] = {}

MAX_EVENTS_PER_SESSION = 500


@dataclass
class IndexingEvent:
    """A single indexing progress event."""

    id: int = 0
    session_id: str = ""
    stage: str = ""
    level: str = "info"
    message: str = ""
    progress: int | None = None
    total: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


def emit_indexing_event(
    session_id: str,
    stage: str,
    message: str,
    *,
    level: str = "info",
    progress: int | None = None,
    total: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
    """Emit an indexing event and notify all subscribers.

    Returns the event dict that was stored.
    """
    with _lock:
        event_id = _next_id_by_session.get(session_id, 1)
        _next_id_by_session[session_id] = event_id + 1

        event = IndexingEvent(
            id=event_id,
            session_id=session_id,
            stage=stage,
            level=level,
            message=message,
            progress=progress,
            total=total,
            metadata=metadata or {},
        )
        event_dict = asdict(event)

        buf = _events_by_session.setdefault(session_id, [])
        buf.append(event_dict)
        if len(buf) > MAX_EVENTS_PER_SESSION:
            _events_by_session[session_id] = buf[-MAX_EVENTS_PER_SESSION:]

        # Push to all subscribers.
        for sub_queue in _subscribers_by_session.get(session_id, []):
            try:
                sub_queue.put_nowait(event_dict)
            except queue.Full:
                pass  # subscriber is slow; drop the event for them

    return event_dict


def get_indexing_events(session_id: str, *, after_id: int = 0) -> list[dict]:
    """Return stored events for a session, optionally filtering by id > after_id."""
    with _lock:
        events = _events_by_session.get(session_id, [])
        if after_id > 0:
            return [e for e in events if e["id"] > after_id]
        return list(events)


def subscribe_indexing_events(session_id: str) -> Generator[dict, None, None]:
    """Yield events as they arrive.  Blocking generator.

    A sentinel ``None`` is returned when the caller should stop (e.g.
    ``clear_indexing_events`` was called or a terminal event was received).
    """
    sub: queue.Queue[dict | None] = queue.Queue(maxsize=200)
    with _lock:
        _subscribers_by_session.setdefault(session_id, []).append(sub)
    try:
        while True:
            try:
                event = sub.get(timeout=15)
            except queue.Empty:
                # Heartbeat timeout — yield None so caller can send ``: heartbeat``
                yield {"_heartbeat": True}
                continue
            if event is None:
                return
            yield event
            if event.get("stage") in {"complete", "failed"}:
                return
    finally:
        with _lock:
            subs = _subscribers_by_session.get(session_id, [])
            if sub in subs:
                subs.remove(sub)


def clear_indexing_events(session_id: str) -> None:
    """Remove all events for a session and signal subscribers to stop."""
    with _lock:
        _events_by_session.pop(session_id, None)
        _next_id_by_session.pop(session_id, None)
        subs = _subscribers_by_session.pop(session_id, [])
        for sub_queue in subs:
            try:
                sub_queue.put_nowait(None)
            except queue.Full:
                pass
