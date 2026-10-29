"""Validate one complete online Medical measurement before analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.artifacts import load_measurement
from reasondb.reasoning.providers import get_model_spec, model_requires_ecologits


EXPECTED_QUERIES = {
    "Q1_allergy_patients",
    "Q3_cancer_family_lung_problems_patients",
    "Q4_average_age_acne_patients",
    "Q9_sick_patients_skin_xray",
}


def _positive_phase(phase: dict, name: str, energy_expected: bool = True) -> None:
    section = phase.get(name, {})
    assert section.get("total_calls", 0) > 0
    assert section.get("total_tokens", 0) > 0
    assert section.get("total_runtime_seconds", 0) > 0
    assert section.get("total_cost_usd", 0) >= 0
    if energy_expected:
        assert section.get("energy_kwh", 0) > 0
        assert section.get("co2_g", 0) > 0
        assert section.get("impact_complete") is True


def validate(measurement_path: Path, requested_model: str) -> None:
    data = load_measurement(measurement_path, hydrate=True)
    config = data.get("config", {})
    assert config.get("backend_profile") == "api"
    assert config.get("model_name") == requested_model
    assert config.get("planner_model_name") == requested_model
    assert config.get("online_cache_policy") == "disabled"
    assert config.get("cache_state") == "disabled"
    assert config.get("model_cache_state") == "disabled"
    # Energy expectations follow the model registry (see reasondb/reasoning/providers.py).
    energy_expected = model_requires_ecologits(requested_model)
    spec = get_model_spec(requested_model)
    requested_names = {requested_model, spec.api_model} if spec else {requested_model}
    assert config.get("require_ecologits") is energy_expected
    assert str(config.get("reasondb_database_name", "")).startswith("online_medical_")

    queries = data.get("queries", [])
    assert {query.get("query_name") for query in queries} == EXPECTED_QUERIES
    for query in queries:
        assert query.get("status") == "success", query.get("error")
        assert query.get("model_name") == requested_model
        if energy_expected:
            assert query.get("energy_kwh", 0) > 0
            assert query.get("co2_g", 0) > 0
        assert query.get("physical_operators")
        assert query.get("operator_plan")

        api = query.get("api_calls", {})
        assert api.get("total_calls", 0) > 0
        assert api.get("cached_calls", 0) == 0
        assert api.get("total_tokens", 0) > 0
        if energy_expected:
            assert query.get("energy", {}).get("metric_source") == "estimated_by_ecologits"

        summary = query.get("result_summary") or {}
        quality = summary.get("quality") or {}
        assert quality.get("status") == "evaluated"
        if query["query_name"] == "Q4_average_age_acne_patients":
            assert quality.get("metric") == "exact_aggregate_average_age"
        else:
            assert quality.get("metric") == "patient_id_set_precision_recall"

        phase = summary.get("api_phase_metrics", {})
        _positive_phase(phase, "planning", energy_expected)
        _positive_phase(phase, "execution", energy_expected)
        assert phase.get("unattributed", {}).get("total_calls", 0) == 0

        events = query.get("api_events", [])
        assert events
        assert {event.get("phase") for event in events} <= {"planning", "execution"}
        assert {event.get("requested_model_name") for event in events if not event.get("cached")} <= requested_names
        assert all(event.get("resolved_model_name") for event in events if not event.get("cached"))
        if energy_expected:
            assert all(event.get("energy_kwh", 0) > 0 and event.get("co2_g", 0) > 0 for event in events if not event.get("cached"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", type=Path)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    validate(args.measurement, args.model)
    print("online Medical validation passed")


if __name__ == "__main__":
    main()
