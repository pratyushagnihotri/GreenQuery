"""OpenAI/API usage tracking for measurement runs."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import time


@dataclass
class APICallEvent:
    """One model call or cache hit observed during a measurement run."""

    provider: str
    model_name: str
    runtime_seconds: float
    requested_model_name: Optional[str] = None
    resolved_model_name: Optional[str] = None
    phase: str = "unattributed"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cost_usd: float = 0.0
    completion_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    cached: bool = False
    energy_kwh: Optional[float] = None
    energy_kwh_min: Optional[float] = None
    energy_kwh_max: Optional[float] = None
    co2_g: Optional[float] = None
    co2_g_min: Optional[float] = None
    co2_g_max: Optional[float] = None
    energy_source: str = "not_available"
    co2_source: str = "not_available"
    metric_source: str = "measured"
    observed_at: float = 0.0
    # Model EcoLogits used for the estimate when it differs from the called model.
    ecologits_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_current_events: ContextVar[Optional[List[APICallEvent]]] = ContextVar(
    "measurement_api_events", default=None
)
_current_phase: ContextVar[str] = ContextVar("measurement_api_phase", default="unattributed")


@contextmanager
def phase_context(phase: str):
    """Tag nested API calls as planning or execution work."""
    token = _current_phase.set(phase)
    try:
        yield
    finally:
        _current_phase.reset(token)


def start_collection():
    """Start collecting API events for the current context."""
    return _current_events.set([])


def stop_collection(token) -> List[APICallEvent]:
    """Stop collecting and return all events collected in the context."""
    events = list(_current_events.get() or [])
    _current_events.reset(token)
    return events


def snapshot_index() -> int:
    """Return the current event-list position for per-query deltas."""
    return len(_current_events.get() or [])


def events_since(index: int) -> List[APICallEvent]:
    """Return events recorded since a previous snapshot."""
    return list((_current_events.get() or [])[index:])


def record_openai_call(
    *,
    model_name: str,
    provider: str = "openai",
    requested_model_name: Optional[str] = None,
    resolved_model_name: Optional[str] = None,
    runtime_seconds: float,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cost_usd: float,
    completion_cost_usd: float,
    energy_kwh: Optional[float] = None,
    energy_kwh_min: Optional[float] = None,
    energy_kwh_max: Optional[float] = None,
    co2_g: Optional[float] = None,
    co2_g_min: Optional[float] = None,
    co2_g_max: Optional[float] = None,
    energy_source: str = "estimated_by_ecologits",
    co2_source: str = "estimated_by_ecologits",
    ecologits_model: Optional[str] = None,
):
    """Record an uncached provider API call when measurement collection is active."""
    events = _current_events.get()
    if events is None:
        return

    total_tokens = prompt_tokens + completion_tokens
    events.append(
        APICallEvent(
            provider=provider,
            model_name=model_name,
            requested_model_name=requested_model_name or model_name,
            resolved_model_name=resolved_model_name or model_name,
            phase=_current_phase.get(),
            runtime_seconds=runtime_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_cost_usd=prompt_cost_usd,
            completion_cost_usd=completion_cost_usd,
            total_cost_usd=prompt_cost_usd + completion_cost_usd,
            cached=False,
            energy_kwh=energy_kwh,
            energy_kwh_min=energy_kwh if energy_kwh_min is None else energy_kwh_min,
            energy_kwh_max=energy_kwh if energy_kwh_max is None else energy_kwh_max,
            co2_g=co2_g,
            co2_g_min=co2_g if co2_g_min is None else co2_g_min,
            co2_g_max=co2_g if co2_g_max is None else co2_g_max,
            energy_source=energy_source if energy_kwh is not None else "not_available",
            co2_source=co2_source if co2_g is not None else "not_available",
            metric_source=f"measured_from_{provider}_usage",
            observed_at=time.time(),
            ecologits_model=ecologits_model,
        )
    )


# Provider-neutral name; record_openai_call is kept for existing callers.
record_api_call = record_openai_call


def record_cache_hit(*, provider: str, model_name: str, runtime_seconds: float):
    """Record a ReasonDB LLM cache hit without charging it as a new API call."""
    events = _current_events.get()
    if events is None:
        return

    events.append(
        APICallEvent(
            provider=provider,
            model_name=model_name,
            requested_model_name=model_name,
            resolved_model_name=model_name,
            phase=_current_phase.get(),
            runtime_seconds=runtime_seconds,
            cached=True,
            metric_source="measured_cache_hit_no_api_call",
            observed_at=time.time(),
        )
    )


def _summary(events: List[APICallEvent], *, include_events: bool) -> Dict[str, Any]:
    uncached = [event for event in events if not event.cached]
    cached = [event for event in events if event.cached]
    models_used = sorted({event.model_name for event in events})
    providers = sorted({event.provider for event in events})

    energy_events = [event for event in uncached if event.energy_kwh is not None]
    co2_events = [event for event in uncached if event.co2_g is not None]
    energy_values = [event.energy_kwh for event in energy_events]
    co2_values = [event.co2_g for event in co2_events]

    return {
        "total_calls": len(uncached),
        "cached_calls": len(cached),
        "models_used": models_used,
        "providers": providers,
        "total_prompt_tokens": sum(event.prompt_tokens for event in uncached),
        "total_completion_tokens": sum(event.completion_tokens for event in uncached),
        "total_tokens": sum(event.total_tokens for event in uncached),
        "prompt_cost_usd": sum(event.prompt_cost_usd for event in uncached),
        "completion_cost_usd": sum(event.completion_cost_usd for event in uncached),
        "total_cost_usd": sum(event.total_cost_usd for event in uncached),
        "total_runtime_seconds": sum(event.runtime_seconds for event in uncached),
        "energy_kwh": sum(energy_values) if energy_values else 0.0,
        "energy_kwh_min": sum(event.energy_kwh_min or 0.0 for event in energy_events),
        "energy_kwh_max": sum(event.energy_kwh_max or 0.0 for event in energy_events),
        "co2_g": sum(co2_values) if co2_values else 0.0,
        "co2_g_min": sum(event.co2_g_min or 0.0 for event in co2_events),
        "co2_g_max": sum(event.co2_g_max or 0.0 for event in co2_events),
        "energy_source": energy_events[0].energy_source if energy_events else "not_available",
        "co2_source": co2_events[0].co2_source if co2_events else "not_available",
        "ecologits_proxy_models": sorted({event.ecologits_model for event in uncached if event.ecologits_model}),
        "impact_complete": bool(uncached) and len(energy_events) == len(uncached) and len(co2_events) == len(uncached),
        "events": [event.to_dict() for event in events] if include_events else [],
    }


def summarize(events: List[APICallEvent]) -> Dict[str, Any]:
    """Aggregate API events, retaining phase-specific EcoLogits ranges."""
    summary = _summary(events, include_events=True)
    summary["phase_metrics"] = {
        phase: _summary([event for event in events if event.phase == phase], include_events=False)
        for phase in ("planning", "execution", "unattributed")
    }
    return summary
