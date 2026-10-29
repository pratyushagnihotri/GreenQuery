"""Per-query telemetry for local model inference."""

from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from copy import deepcopy
from typing import Any


_current_events: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "measurement_local_inference_events", default=None
)
_current_operator: ContextVar[dict[str, Any] | None] = ContextVar(
    "measurement_local_inference_operator", default=None
)


def start_collection():
    return _current_events.set([])


def stop_collection(token) -> list[dict[str, Any]]:
    events = deepcopy(_current_events.get() or [])
    _current_events.reset(token)
    return events


def snapshot_index() -> int:
    return len(_current_events.get() or [])


def events_since(index: int) -> list[dict[str, Any]]:
    return deepcopy((_current_events.get() or [])[index:])


@contextmanager
def operator_context(operator: dict[str, Any]):
    """Attach a physical-plan operator identity and API phase to nested calls."""
    token = _current_operator.set(deepcopy(operator))
    try:
        from .api_usage import phase_context

        phase = "execution" if operator.get("phase") == "execution" else "planning"
        with phase_context(phase):
            yield
    finally:
        _current_operator.reset(token)


def record_local_inference(event: dict[str, Any]) -> None:
    events = _current_events.get()
    if events is not None:
        recorded = deepcopy(event)
        operator = _current_operator.get()
        if operator is not None:
            recorded["operator"] = deepcopy(operator)
        events.append(recorded)


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "request_count": sum(int(event.get("request_count", 0)) for event in events),
        "processed_rows": sum(int(event.get("processed_rows", 0)) for event in events),
        "answer_cache_hits": sum(
            int(event.get("answer_cache_hits", 0)) for event in events
        ),
        "kv_cache_hits": sum(int(event.get("kv_cache_hits", 0)) for event in events),
        "input_tokens": sum(int(event.get("input_tokens", 0)) for event in events),
        "output_tokens": sum(int(event.get("output_tokens", 0)) for event in events),
        "inference_seconds": sum(
            float(event.get("inference_seconds", 0.0)) for event in events
        ),
        "events": deepcopy(events),
    }
