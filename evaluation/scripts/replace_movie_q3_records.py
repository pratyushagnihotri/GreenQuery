"""Replace one Movie Q3 record per existing repetition and rebuild summaries."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.artifacts import load_measurement, save_measurement


QUERY = "Q3_reviews_same_sentiment"


def measurement_file(run_dir: Path) -> Path:
    matches = sorted(path for path in run_dir.glob("movie_*.json") if not path.name.endswith("_summary.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one measurement in {run_dir}, found {len(matches)}")
    return matches[0]


def totals(queries: list[dict]) -> dict:
    return {
        "total_queries": len(queries),
        "total_api_calls": sum((query.get("api_calls") or {}).get("total_calls", 0) for query in queries),
        "total_cached_calls": sum((query.get("api_calls") or {}).get("cached_calls", 0) for query in queries),
        "total_tokens": sum(query.get("total_tokens", 0) or 0 for query in queries),
        "total_local_inference_requests": sum((query.get("local_inference") or {}).get("request_count", 0) for query in queries),
        "total_energy_kwh": sum(query.get("energy_kwh", 0) or 0 for query in queries),
        "total_carbon_gco2eq": sum(query.get("co2_g", 0) or 0 for query in queries),
        "total_co2_g": sum(query.get("co2_g", 0) or 0 for query in queries),
        "total_cost_usd": sum(query.get("api_cost_usd", 0) or 0 for query in queries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replacement_root", type=Path)
    parser.add_argument("target_root", type=Path)
    parser.add_argument("model")
    args = parser.parse_args()
    source_model = args.replacement_root / args.model
    target_model = args.target_root / args.model
    backup_model = args.target_root / "_superseded_q3" / args.model
    runner = Path(__file__).with_name("build_movie_demo_summary.py")

    for source_run in sorted(path for path in source_model.glob("run_*") if path.is_dir()):
        target_run = target_model / source_run.name
        source_path = measurement_file(source_run)
        target_path = measurement_file(target_run)
        source = load_measurement(source_path, hydrate=True)
        target = load_measurement(target_path, hydrate=True)
        replacement = next(query for query in source["queries"] if query.get("query_name") == QUERY)
        if len(source["queries"]) != 1 or not replacement.get("execution_plan", {}).get("physical", {}).get("steps"):
            raise ValueError(f"Invalid Q3 replacement: {source_path}")
        backup_run = backup_model / source_run.name
        backup_run.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_path, backup_run / target_path.name)
        summary_path = target_path.with_name(f"{target_path.stem}_summary.json")
        if summary_path.exists():
            shutil.copy2(summary_path, backup_run / summary_path.name)
        target["queries"] = [replacement if query.get("query_name") == QUERY else query for query in target["queries"]]
        target["totals"] = totals(target["queries"])
        target["q3_replacement_source"] = str(source_path)
        target_path.unlink()
        save_measurement(target, target_path)
        subprocess.run([sys.executable, str(runner), str(target_path)], check=True)
        print(f"replaced {args.model}/{source_run.name}")


if __name__ == "__main__":
    main()
