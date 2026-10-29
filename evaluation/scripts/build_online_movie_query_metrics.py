"""Build graph-ready Movie records from validated online run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_provenance(config: dict) -> dict:
    """Optimizer and operator-model mode recorded by MeasurementWrapper.

    Runs measured before 2026-10-16 did not record them: they stay None rather
    than being guessed.
    """
    return {
        "optimizer": config.get("optimizer"),
        "operator_mode": config.get("operator_mode"),
        "operator_models": config.get("operator_models"),
    }


def phase(record: dict, name: str) -> dict:
    return record.get("online", {}).get(name, {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = []
    for model_dir in sorted(path for path in args.raw_root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("run_")):
            summaries = sorted(run_dir.glob("*_summary.json"))
            if len(summaries) != 1:
                raise ValueError(f"Expected one summary in {run_dir}, found {len(summaries)}")
            summary = json.loads(summaries[0].read_text())
            measurement_path = summaries[0].with_name(summaries[0].name.replace("_summary.json", ".json"))
            config = json.loads(measurement_path.read_text()).get("config", {}) if measurement_path.exists() else {}
            for query in summary["queries"]:
                online = query["online"]
                energy = query["energy"]
                records.append(
                    {
                        "demo": "movie",
                        "run": int(run_dir.name.removeprefix("run_")),
                        "query": query["query"],
                        "model": query["model"],
                        "planner_model": online["requested_models"][0],
                        "resolved_models": online["resolved_models"],
                        "runtime_seconds": query["runtime_seconds"],
                        "cache_hits": online["cached_calls"],
                        "api_calls": online["api_calls"],
                        "input_tokens": online["input_tokens"],
                        "output_tokens": online["output_tokens"],
                        "total_tokens": online["input_tokens"] + online["output_tokens"],
                        "api_cost_usd": online["api_cost_usd"],
                        "energy_kwh": energy["energy_kwh"],
                        "energy_kwh_min": energy["energy_kwh_min"],
                        "energy_kwh_max": energy["energy_kwh_max"],
                        "co2_g": energy["co2_g"],
                        "co2_g_min": energy["co2_g_min"],
                        "co2_g_max": energy["co2_g_max"],
                        "energy_source": energy["energy_source"],
                        "co2_source": energy["co2_source"],
                        "planning": phase(query, "planning"),
                        "execution": phase(query, "execution"),
                        "physical_operators": query["operators"],
                        "execution_plan": query.get("execution_plan", {}),
                        "quality": query["quality"],
                        **run_provenance(config),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"{args.output}: {len(records)} records")


if __name__ == "__main__":
    main()
