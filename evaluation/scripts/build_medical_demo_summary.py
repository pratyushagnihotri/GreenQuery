"""Build one Movie-layout-compatible Medical run summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.artifacts import load_measurement


def _phase(query: dict, name: str) -> dict:
    return ((query.get("result_summary") or {}).get("api_phase_metrics") or {}).get(name, {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    measurement = load_measurement(args.measurement, hydrate=True)
    config = measurement.get("config") or {}
    queries = []
    for query in measurement.get("queries", []):
        api = query.get("api_calls") or {}
        local = query.get("local_inference") or {}
        summary = query.get("result_summary") or {}
        events = query.get("api_events") or []
        queries.append({
            "query": query.get("query_name"),
            "model": query.get("model_name"),
            "operators": query.get("physical_operators", []),
            "operator_plan": query.get("operator_plan", []),
            "runtime_seconds": query.get("execution_time_seconds"),
            "energy": {
                "energy_kwh": query.get("energy_kwh"),
                "energy_kwh_min": summary.get("api_energy_kwh_min"),
                "energy_kwh_max": summary.get("api_energy_kwh_max"),
                "co2_g": query.get("co2_g"),
                "co2_g_min": summary.get("api_co2_g_min"),
                "co2_g_max": summary.get("api_co2_g_max"),
                "energy_source": (query.get("energy") or {}).get("metric_source"),
                "co2_source": (query.get("metric_sources") or {}).get("co2"),
            },
            "online": {
                "api_calls": api.get("total_calls", 0),
                "cached_calls": api.get("cached_calls", 0),
                "input_tokens": api.get("total_prompt_tokens", 0),
                "output_tokens": api.get("total_completion_tokens", 0),
                "api_cost_usd": query.get("api_cost_usd"),
                "requested_models": sorted({event.get("requested_model_name") for event in events if event.get("requested_model_name")}),
                "resolved_models": sorted({event.get("resolved_model_name") for event in events if event.get("resolved_model_name")}),
                "planning": _phase(query, "planning"),
                "execution": _phase(query, "execution"),
            },
            "offline": {
                "local_inference_requests": local.get("request_count", 0),
                "planner_requests": local.get("planner_request_count", 0),
                "input_tokens": local.get("execution_input_tokens", 0),
                "output_tokens": local.get("execution_output_tokens", 0),
            },
            "quality": (query.get("result_summary") or {}).get("quality"),
            "status": query.get("status"),
        })
    summary = {
        "summary_layout_version": "medical-demo-summary-v1",
        "measurement_id": measurement.get("measurement_id"),
        "cache_policy": config.get("online_cache_policy") or config.get("local_cache_policy"),
        "raw_measurement": {
            "path": str(args.measurement),
            "sha256": hashlib.sha256(args.measurement.read_bytes()).hexdigest(),
        },
        "ground_truth": {
            "directory": "reasondb/evaluation/ground_truth/medical",
            "query_files": {
                name: hashlib.sha256((Path("reasondb/evaluation/ground_truth/medical") / name).read_bytes()).hexdigest()
                for name in ("Q1_1000.csv", "Q3_1000.csv", "Q4_1000.csv", "Q9_1000.csv")
            },
        },
        "queries": queries,
    }
    output = args.output or args.measurement.with_name(f"{args.measurement.stem}_summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
