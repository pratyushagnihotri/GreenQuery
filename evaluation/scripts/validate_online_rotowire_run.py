"""Validate one complete online RotoWire measurement before analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.artifacts import load_measurement
from reasondb.reasoning.providers import get_model_spec, model_requires_ecologits


EXPECTED_QUERIES = {"R1_player_points_assists", "R2_player_points_home_team_won"}


def validate(measurement_path: Path, requested_model: str) -> None:
    data = load_measurement(measurement_path, hydrate=True)
    config = data.get("config", {})
    assert config.get("backend_profile") == "api"
    assert config.get("model_name") == requested_model
    assert config.get("planner_model_name") == requested_model
    assert config.get("online_cache_policy") == "disabled"
    energy_expected = model_requires_ecologits(requested_model)
    assert config.get("require_ecologits") is energy_expected

    queries = data.get("queries", [])
    assert {query.get("query_name") for query in queries} == EXPECTED_QUERIES
    for query in queries:
        assert query.get("status") == "success", query.get("error")
        api = query.get("api_calls", {})
        assert api.get("total_calls", 0) > 0
        assert api.get("cached_calls", 0) == 0
        if energy_expected:
            assert query.get("energy", {}).get("metric_source") == "estimated_by_ecologits"
            assert query.get("energy_kwh", 0) > 0 and query.get("co2_g", 0) > 0
        phase = (query.get("result_summary") or {}).get("api_phase_metrics", {})
        assert phase.get("planning", {}).get("total_calls", 0) > 0
        assert phase.get("execution", {}).get("total_calls", 0) > 0
        if energy_expected:
            assert phase["planning"].get("impact_complete") is True
            assert phase["execution"].get("impact_complete") is True
        assert query.get("api_events")

    r1 = next(query for query in queries if query["query_name"] == "R1_player_points_assists")
    assert (r1.get("result_summary") or {}).get("quality", {}).get("checked_fields", 0) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", type=Path)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    validate(args.measurement, args.model)
    print("online RotoWire validation passed")


if __name__ == "__main__":
    main()
