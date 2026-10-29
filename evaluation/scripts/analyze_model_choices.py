"""Which model did each optimizer use, and when? (multi-model runs of run_sembench_measured.sbatch)

  python evaluation/scripts/analyze_model_choices.py tests/results/sembench-measured-49853 ...

Per query window: calls per served model, the call sequence as run lengths in time order
(S = smallest operator model, L = any larger model), energy, and quality. A typical pattern is
"L1 S60 L138": one planning call, the small model tried on a sample, then the bulk on the large one.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def size_b(model: str) -> float:
    match = re.search(r":(\d+(?:\.\d+)?)b", model or "")
    return float(match.group(1)) if match else 0.0


def run_lengths(sequence: str) -> str:
    parts, previous, count = [], None, 0
    for symbol in sequence:
        if symbol == previous:
            count += 1
        else:
            if previous:
                parts.append(f"{previous}{count}")
            previous, count = symbol, 1
    if previous:
        parts.append(f"{previous}{count}")
    return " ".join(parts)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(run: Path) -> list:
    config = dict(line.split("=", 1) for line in (run / "run_config.txt").read_text().splitlines() if "=" in line)
    calls = read_jsonl(run / "calls.jsonl")
    windows = [w for w in read_jsonl(run / "windows.jsonl") if w["kind"] == "query"]
    energy = {(r["system"], r["query"], r["repetition"]): r
              for r in json.loads((run / "energy_records.json").read_text())}
    operators = [m.replace("ollama/", "") for m in config.get("operator_models", "").split(",") if m]
    smallest = min(operators, key=size_b) if operators else None
    rows = []
    for window in windows:
        inside = sorted((c for c in calls if window["started_at"] <= c["started_at"] <= window["finished_at"]),
                        key=lambda c: c["started_at"])
        metric_path = run / "metrics" / f"rep_{window['repetition']}" / f"{window['system']}_{window['query']}.json"
        metric = next(iter(json.loads(metric_path.read_text()).values()), {}) if metric_path.exists() else {}
        record = energy[(window["system"], window["query"], window["repetition"])]
        rows.append({
            "run": run.name, "planner": config.get("local_model"), "operators": " + ".join(operators),
            "target": config.get("quality_target", "0.8"), "system": window["system"], "query": window["query"],
            "repetition": window["repetition"],
            "calls": dict(Counter(c["served_model"] for c in inside)),
            "small_share": (sum(c["served_model"] == smallest for c in inside) / len(inside)) if inside else None,
            "sequence": run_lengths("".join("S" if c["served_model"] == smallest else "L" for c in inside)),
            "gpu_wh": record["gpu_energy_j"] / 3600 if record["gpu_energy_j"] is not None else None,
            "cpu_wh": record["cpu_job_energy_j"] / 3600 if record["cpu_job_energy_j"] is not None else None,
            "status": metric.get("status"),
            "quality": metric.get("f1_score", metric.get("relative_error")),
            "quality_metric": "f1" if "f1_score" in metric else "relative_error" if "relative_error" in metric else "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="also write all rows as JSON")
    args = parser.parse_args()
    rows = [row for run in args.runs for row in analyze(run)]
    for row in rows:
        quality = "–" if row["quality"] is None else f"{row['quality_metric']} {row['quality']:.2f}"
        print(f"{row['run'][-5:]} {row['operators']:32} t={row['target']} {row['system']:14} {row['query']} "
              f"small={row['small_share'] or 0:4.0%}  {row['sequence'][:40]:40} GPU {row['gpu_wh'] or 0:5.1f} Wh  {quality}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
