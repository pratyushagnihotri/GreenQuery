"""Build graph-ready online RotoWire records from validated measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.measurement.artifacts import load_measurement
from evaluation.scripts.build_online_movie_query_metrics import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = []
    for model_dir in sorted(path for path in args.raw_root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in model_dir.glob("run_*") if path.is_dir()):
            measurements = sorted(path for path in run_dir.glob("rotowire_*.json") if not path.name.endswith("_summary.json"))
            if len(measurements) != 1:
                raise ValueError(f"Expected one measurement in {run_dir}, found {len(measurements)}")
            measurement = load_measurement(measurements[0], hydrate=True)
            for query in measurement["queries"]:
                api = query["api_calls"]
                energy = query["energy"]
                summary = query.get("result_summary") or {}
                records.append(
                    {
                        "demo": "rotowire",
                        "run": int(run_dir.name.removeprefix("run_")),
                        "query": query["query_name"],
                        "model": query["model_name"],
                        "planner_model": measurement["config"]["planner_model_name"],
                        "resolved_models": sorted({event["resolved_model_name"] for event in query.get("api_events", []) if event.get("resolved_model_name")}),
                        "runtime_seconds": query["execution_time_seconds"],
                        "cache_hits": api["cached_calls"],
                        "api_calls": api["total_calls"],
                        "input_tokens": api["total_prompt_tokens"],
                        "output_tokens": api["total_completion_tokens"],
                        "total_tokens": query["total_tokens"],
                        "api_cost_usd": query["api_cost_usd"],
                        "energy_kwh": query["energy_kwh"],
                        "co2_g": query["co2_g"],
                        "energy_source": energy["metric_source"],
                        "co2_source": query["metric_sources"]["co2"],
                        "planning": summary.get("api_phase_metrics", {}).get("planning", {}),
                        "execution": summary.get("api_phase_metrics", {}).get("execution", {}),
                        "physical_operators": query.get("physical_operators", []),
                        "quality": summary.get("quality", {}),
                        **run_provenance(measurement["config"]),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"{args.output}: {len(records)} records")


if __name__ == "__main__":
    main()
