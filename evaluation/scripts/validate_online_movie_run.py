"""Validate one complete online Movie measurement before it enters analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.artifacts import load_measurement
from reasondb.reasoning.providers import get_model_spec, model_requires_ecologits


EXPECTED_QUERIES = {
    "Q1_positive_reviews",
    "Q2_positive_reviews_for_taken_3",
    "Q3_reviews_same_sentiment",
    "Q4_positive_reviews_for_taken_3_ratio",
}


def validate(
    measurement_path: Path,
    summary_path: Path,
    requested_model: str,
    only_query: str | None = None,
) -> None:
    measurement = load_measurement(measurement_path, hydrate=True)
    config = measurement.get("config", {})
    assert config.get("backend_profile") == "api"
    assert config.get("model_name") == requested_model
    assert config.get("planner_model_name") == requested_model
    assert config.get("online_cache_policy") == "disabled"
    # Energy and cost expectations follow the model registry: EcoLogits-backed
    # API models must report both; free local servers report neither.
    energy_expected = model_requires_ecologits(requested_model)
    spec = get_model_spec(requested_model)
    paid = spec is None or spec.price_in > 0 or spec.price_out > 0
    assert config.get("require_ecologits") is energy_expected

    queries = measurement.get("queries", [])
    expected_queries = {only_query} if only_query else EXPECTED_QUERIES
    assert {query.get("query_name") for query in queries} == expected_queries
    for query in queries:
        assert query.get("status") == "success", query.get("error")
        api = query.get("api_calls", {})
        assert api.get("total_calls", 0) > 0
        assert api.get("cached_calls", 0) == 0
        assert api.get("total_tokens", 0) > 0
        if paid:
            assert query.get("api_cost_usd", 0) > 0
        if energy_expected:
            assert query.get("energy", {}).get("metric_source") == "estimated_by_ecologits"
            assert query.get("energy_kwh", 0) > 0
            assert query.get("co2_g", 0) > 0
        phase = query.get("result_summary", {}).get("api_phase_metrics", {})
        assert phase.get("planning", {}).get("total_calls", 0) > 0
        assert phase.get("execution", {}).get("total_calls", 0) > 0
        assert phase["planning"].get("total_tokens", 0) > 0
        assert phase["execution"].get("total_tokens", 0) > 0
        if paid:
            assert phase["planning"].get("total_cost_usd", 0) > 0
            assert phase["execution"].get("total_cost_usd", 0) > 0
        if energy_expected:
            assert phase["planning"].get("impact_complete") is True
            assert phase["execution"].get("impact_complete") is True
        events = query.get("api_events", [])
        assert events and {event.get("phase") for event in events} <= {"planning", "execution"}
        assert all(event.get("resolved_model_name") for event in events if not event.get("cached"))
        plan = query.get("execution_plan", {})
        assert plan.get("schema_version") == 1
        assert plan.get("logical", {}).get("steps")
        assert plan.get("physical", {}).get("steps")
        assert plan.get("edges")
        logical_ids = {step["id"] for step in plan["logical"]["steps"]}
        implemented_ids = {edge["from"] for edge in plan["edges"]}
        assert logical_ids <= implemented_ids

    summary = json.loads(summary_path.read_text())
    quality = {query["query"]: query.get("quality", {}) for query in summary.get("queries", [])}
    assert set(quality) == expected_queries
    assert all(value.get("status") == "evaluated" for value in quality.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--only-query", choices=sorted(EXPECTED_QUERIES))
    args = parser.parse_args()
    validate(args.measurement, args.summary, args.model, args.only_query)
    print("online Movie validation passed")


if __name__ == "__main__":
    main()
