"""Build compact graph-ready Medical records from validated online measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.artifacts import load_measurement


def _phase(query: dict, name: str) -> dict:
    return ((query.get("result_summary") or {}).get("api_phase_metrics") or {}).get(name, {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = []
    for model_dir in sorted(path for path in args.raw_root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in model_dir.glob("run_*") if path.is_dir()):
            measurements = sorted(
                path
                for path in run_dir.glob("medical_*.json")
                if not path.name.endswith("_summary.json")
            )
            if len(measurements) != 1:
                raise ValueError(f"Expected one measurement in {run_dir}, found {len(measurements)}")
            measurement_path = measurements[0]
            measurement = load_measurement(measurement_path, hydrate=True)
            for query in measurement["queries"]:
                api = query["api_calls"]
                summary = query.get("result_summary") or {}
                events = query.get("api_events") or []
                records.append(
                    {
                        "demo": "medical",
                        "run": int(run_dir.name.removeprefix("run_")),
                        "query": query["query_name"],
                        "model": query["model_name"],
                        "planner_model": measurement["config"]["planner_model_name"],
                        "run_namespace": measurement["config"].get("reasondb_database_name"),
                        "raw_measurement": str(measurement_path),
                        "raw_measurement_sha256": hashlib.sha256(measurement_path.read_bytes()).hexdigest(),
                        "resolved_models": sorted({event["resolved_model_name"] for event in events if event.get("resolved_model_name")}),
                        "runtime_seconds": query["execution_time_seconds"],
                        "cache_hits": api["cached_calls"],
                        "api_calls": api["total_calls"],
                        "input_tokens": api["total_prompt_tokens"],
                        "output_tokens": api["total_completion_tokens"],
                        "total_tokens": query["total_tokens"],
                        "api_cost_usd": query["api_cost_usd"],
                        "energy_kwh": query["energy_kwh"],
                        "energy_kwh_min": summary.get("api_energy_kwh_min"),
                        "energy_kwh_max": summary.get("api_energy_kwh_max"),
                        "co2_g": query["co2_g"],
                        "co2_g_min": summary.get("api_co2_g_min"),
                        "co2_g_max": summary.get("api_co2_g_max"),
                        "energy_source": query["energy"]["metric_source"],
                        "co2_source": query["metric_sources"]["co2"],
                        "planning": _phase(query, "planning"),
                        "execution": _phase(query, "execution"),
                        "physical_operators": query.get("physical_operators", []),
                        "operator_plan": query.get("operator_plan", []),
                        "quality": summary.get("quality", {}),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"{args.output}: {len(records)} records")


if __name__ == "__main__":
    main()
