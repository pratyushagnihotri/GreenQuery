"""Build one compact, chart-ready summary from matched-planner measurements."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evaluation.measurement.artifacts import load_measurement
from build_movie_demo_summary import _labels, _quality as movie_quality


ROOT = Path(__file__).parents[2]


def counter_fields(query: dict) -> dict:
    """Hardware-counter energy recorded next to the nvidia-smi sampler (if enabled)."""
    counters = (query.get("result_summary") or {}).get("energy_counters")
    if not counters:
        return {}
    phases = counters.get("phases") or {}
    return {
        "counter_method": counters.get("method"),
        "counter_gpu_energy_kwh": counters.get("gpu_energy_kwh"),
        "counter_cpu_package_energy_kwh": counters.get("cpu_package_energy_kwh"),
        "counter_dram_energy_kwh": counters.get("dram_energy_kwh"),
        "counter_measured_energy_kwh": counters.get("measured_energy_kwh"),
        "counter_measured_boundary": counters.get("measured_boundary"),
        "counter_gpu_dynamic_energy_kwh": counters.get("gpu_dynamic_energy_kwh"),
        "counter_cpu_package_dynamic_energy_kwh": counters.get("cpu_package_dynamic_energy_kwh"),
        "counter_planner_gpu_energy_kwh": (phases.get("planner") or {}).get("gpu_energy_kwh"),
        "counter_execution_gpu_energy_kwh": (phases.get("execution") or {}).get("gpu_energy_kwh"),
        "counter_planner_cpu_package_energy_kwh": (phases.get("planner") or {}).get("cpu_package_energy_kwh"),
        "counter_execution_cpu_package_energy_kwh": (phases.get("execution") or {}).get("cpu_package_energy_kwh"),
        "counter_unattributed_gpu_energy_kwh": phases.get("unattributed_gpu_energy_kwh"),
        "counter_measured_co2_g": counters.get("measured_co2_g"),
        "counter_sample_count": counters.get("sample_count"),
        "counter_errors": [error for error in (counters.get("error"), counters.get("gpu_error"), counters.get("rapl_error")) if error],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    movie_labels = _labels(ROOT / "reasondb/evaluation/benchmarks/files/reviews_100.csv")
    movie_q4_labels = _labels(ROOT / "reasondb/evaluation/benchmarks/files/reviews_1000.csv")
    rows = []
    for path in sorted(args.root.rglob("*.json")):
        if path.name.endswith("_summary.json") or path.name == "query_metrics.json":
            continue
        # Compact query fields live in the main JSON; loading sidecar prediction rows
        # would make dashboard aggregation needlessly slow and fragile.
        measurement = load_measurement(path, hydrate=True)
        if "queries" not in measurement:
            continue
        config = measurement.get("config", {})
        run = next((part for part in path.parts if re.fullmatch(r"run_\d+", part)), "run_unknown")
        for query in measurement["queries"]:
            local = query.get("local_inference") or {}
            phase_energy = (query.get("result_summary") or {}).get("phase_energy") or {}
            planner_energy = phase_energy.get("planner") or {}
            execution_energy = phase_energy.get("execution") or {}
            quality = (
                movie_quality(query, movie_labels, movie_q4_labels)
                if measurement.get("demo_name") == "movie"
                else (query.get("result_summary") or {}).get("quality")
            )
            selection_quality = (quality or {}).get("selection") or quality or {}
            century_quality = (quality or {}).get("century_extraction") or {}
            rows.append(
                {
                    "demo": measurement.get("demo_name"),
                    "model": query.get("model_name"),
                    "planner_model": config.get("planner_model_name"),
                    "run": run,
                    "query": query.get("query_name"),
                    "runtime_seconds": query.get("execution_time_seconds"),
                    "energy_kwh": query.get("energy_kwh"),
                    "co2_g": query.get("co2_g"),
                    "execution_input_tokens": local.get("execution_input_tokens", 0),
                    "execution_output_tokens": local.get("execution_output_tokens", 0),
                    "planner_input_tokens": local.get("planner_input_tokens", 0),
                    "planner_output_tokens": local.get("planner_output_tokens", 0),
                    "total_tokens": query.get("total_tokens", 0),
                    "local_requests": local.get("request_count", 0),
                    "planner_requests": local.get("planner_request_count", 0),
                    "cache_hits": local.get("answer_cache_hits", 0) + local.get("kv_cache_hits", 0),
                    "direct_qa_batch_size": config.get("direct_qa_batch_size"),
                    "vision_qa_batch_size": config.get("vision_qa_batch_size"),
                    "vision_model": config.get("local_vision_model_id"),
                    "run_namespace": config.get("run_namespace"),
                    "direct_qa_batch_count": sum(
                        int(event.get("inference_batch_count", 0))
                        for event in local.get("events", [])
                        if event.get("execution_mode") == "direct_no_cache"
                    ),
                    "quality": quality,
                    "quality_selection_f1": selection_quality.get("f1"),
                    "quality_century_accuracy": century_quality.get("accuracy"),
                    "quality_century_checked_rows": century_quality.get("checked_rows"),
                    "physical_operators": query.get("physical_operators", []),
                    "execution_plan": query.get("execution_plan", {}),
                    "energy_source": (query.get("energy") or {}).get("metric_source"),
                    "phase_energy_valid": phase_energy.get("valid", False),
                    "phase_energy_provenance": phase_energy.get("provenance"),
                    "phase_energy_source": phase_energy.get("energy_source"),
                    "phase_co2_source": phase_energy.get("co2_source"),
                    "phase_energy_invalid_reasons": phase_energy.get(
                        "invalid_reasons", []
                    ),
                    "planner_energy_kwh": planner_energy.get("energy_kwh"),
                    "planner_co2_g": planner_energy.get("co2_g"),
                    "planner_nvml_sample_count": planner_energy.get("sample_count"),
                    "execution_energy_kwh": execution_energy.get("energy_kwh"),
                    "execution_co2_g": execution_energy.get("co2_g"),
                    "execution_nvml_sample_count": execution_energy.get("sample_count"),
                    "phase_gpu_roles": phase_energy.get("gpu_roles"),
                    "query_energy_kwh_by_gpu": phase_energy.get(
                        "query_energy_kwh_by_gpu"
                    ),
                    "planner_energy_kwh_by_gpu_role": planner_energy.get(
                        "energy_kwh_by_gpu_role"
                    ),
                    "execution_energy_kwh_by_gpu_role": execution_energy.get(
                        "energy_kwh_by_gpu_role"
                    ),
                    "unattributed_energy_kwh": phase_energy.get(
                        "unattributed_query_energy_kwh"
                    ),
                    "unattributed_co2_g": phase_energy.get("unattributed_query_co2_g"),
                    **counter_fields(query),
                }
            )
    output = args.output or args.root / "query_metrics.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{output} ({len(rows)} query rows)")


if __name__ == "__main__":
    main()
