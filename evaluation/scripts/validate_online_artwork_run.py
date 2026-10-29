"""Validate one online Artwork preflight and optionally write its compact artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.artifacts import load_measurement
from evaluation.scripts.artwork_quality import EXPECTED_QUERIES

QUERY_ID = EXPECTED_QUERIES[5]


def _positive_phase(phase: dict[str, Any], name: str) -> None:
    section = phase.get(name, {})
    assert section.get("total_calls", 0) > 0
    assert section.get("total_tokens", 0) > 0
    assert section.get("total_runtime_seconds", 0) > 0
    assert section.get("total_cost_usd", 0) > 0
    assert section.get("energy_kwh", 0) > 0
    assert section.get("energy_kwh_min", 0) > 0
    assert section.get("energy_kwh_max", 0) > 0
    assert section.get("co2_g", 0) > 0
    assert section.get("co2_g_min", 0) > 0
    assert section.get("co2_g_max", 0) > 0
    assert section.get("impact_complete") is True


def _assert_phase_totals(query: dict[str, Any]) -> None:
    api = query.get("api_calls", {})
    phase = (query.get("result_summary") or {}).get("api_phase_metrics", {})
    for field in ("total_calls", "total_prompt_tokens", "total_completion_tokens", "total_tokens"):
        assert phase["planning"].get(field, 0) + phase["execution"].get(field, 0) == api.get(field, 0)
    assert phase.get("unattributed", {}).get("total_calls", 0) == 0


def compact_artifact(data: dict[str, Any]) -> dict[str, Any]:
    query = data["queries"][0]
    summary = query.get("result_summary") or {}
    api = query.get("api_calls") or {}
    phase = summary.get("api_phase_metrics") or {}
    events = query.get("api_events") or []
    return {
        "artifact_version": "artwork-online-preflight-v1",
        "measurement_id": data.get("measurement_id"),
        "query": query.get("query_name"),
        "model": query.get("model_name"),
        "resolved_models": sorted({event.get("resolved_model_name") for event in events if not event.get("cached")}),
        "operators": query.get("physical_operators", []),
        "runtime_seconds": query.get("execution_time_seconds"),
        "api_calls": api.get("total_calls", 0),
        "cached_calls": api.get("cached_calls", 0),
        "input_tokens": api.get("total_prompt_tokens", 0),
        "output_tokens": api.get("total_completion_tokens", 0),
        "total_tokens": api.get("total_tokens", 0),
        "api_cost_usd": query.get("api_cost_usd", 0),
        "energy_kwh": query.get("energy_kwh", 0),
        "energy_kwh_min": summary.get("api_energy_kwh_min"),
        "energy_kwh_max": summary.get("api_energy_kwh_max"),
        "co2_g": query.get("co2_g", 0),
        "co2_g_min": summary.get("api_co2_g_min"),
        "co2_g_max": summary.get("api_co2_g_max"),
        "planning": phase.get("planning", {}),
        "execution": phase.get("execution", {}),
        "quality": summary.get("quality", {}),
        "execution_plan": query.get("execution_plan", {}),
    }


def validate(measurement_path: Path, requested_model: str) -> dict[str, Any]:
    data = load_measurement(measurement_path, hydrate=True)
    config = data.get("config", {})
    assert config.get("backend_profile") == "api"
    assert config.get("model_name") == requested_model
    assert config.get("planner_model_name") == requested_model
    assert config.get("online_cache_policy") == "disabled"
    assert config.get("cache_state") == "disabled"
    assert config.get("model_cache_state") == "disabled"
    assert config.get("require_ecologits") is True
    assert str(config.get("reasondb_database_name", "")).startswith("online_artwork_")

    queries = data.get("queries", [])
    assert [query.get("query_name") for query in queries] == [QUERY_ID]
    query = queries[0]
    assert query.get("status") == "success", query.get("error")
    assert query.get("model_name") == requested_model
    assert query.get("energy", {}).get("metric_source") == "estimated_by_ecologits"
    assert query.get("energy_kwh", 0) > 0 and query.get("co2_g", 0) > 0
    summary = query.get("result_summary")
    assert summary
    assert summary.get("api_energy_kwh_min", 0) > 0 and summary.get("api_energy_kwh_max", 0) > 0
    assert summary.get("api_co2_g_min", 0) > 0 and summary.get("api_co2_g_max", 0) > 0
    assert summary.get("quality", {}).get("status") == "evaluated"
    assert summary.get("quality", {}).get("metric") == "artwork_selection_and_century"
    assert summary.get("expected_query_ids") == [QUERY_ID]

    api = query.get("api_calls", {})
    assert api.get("total_calls", 0) > 0
    assert api.get("cached_calls", 0) == 0
    assert api.get("total_tokens", 0) > 0
    assert query.get("api_cost_usd", 0) > 0
    phase = summary.get("api_phase_metrics", {})
    _positive_phase(phase, "planning")
    _positive_phase(phase, "execution")
    _assert_phase_totals(query)

    events = query.get("api_events", [])
    assert events
    assert {event.get("phase") for event in events} <= {"planning", "execution"}
    assert {event.get("requested_model_name") for event in events if not event.get("cached")} == {requested_model}
    resolved_by_phase = {
        name: {event.get("resolved_model_name") for event in events if event.get("phase") == name and not event.get("cached")}
        for name in ("planning", "execution")
    }
    assert resolved_by_phase["planning"] and resolved_by_phase["planning"] == resolved_by_phase["execution"]
    assert all(
        (event.get("energy_kwh") or 0) > 0
        and (event.get("energy_kwh_min") or 0) > 0
        and (event.get("energy_kwh_max") or 0) > 0
        and (event.get("co2_g") or 0) > 0
        and (event.get("co2_g_min") or 0) > 0
        and (event.get("co2_g_max") or 0) > 0
        for event in events if not event.get("cached")
    )

    plan = query.get("execution_plan", {})
    assert plan.get("schema_version") == 1
    assert plan.get("logical", {}).get("steps")
    assert plan.get("physical", {}).get("steps")
    assert plan.get("edges")
    assert query.get("physical_operators")
    return compact_artifact(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifact = validate(args.measurement, args.model)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print("online Artwork validation passed")


if __name__ == "__main__":
    main()
