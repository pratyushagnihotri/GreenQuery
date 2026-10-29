"""Export the combined SemBench table as the dashboard's bundled snapshot (works without the DGX).

  python evaluation/scripts/export_dashboard_snapshot.py reports/sembench_measured.csv \
      --out dashboard/src/data/sembench_measured.json

Reads the CSV written by combine_sembench_measured.py; numbers stay as recorded (no rounding
beyond what the combine step did). Local runs have energy_source "measured"; API runs "api" with
no energy value (GPU energy happens at the provider; tokens, calls and quality are still shown).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

NUMERIC = ["quality", "score", "runtime_s", "llm_calls", "llm_errors", "prompt_tokens", "completion_tokens",
           "gpu_wh", "gpu_dynamic_wh", "cpu_job_wh", "gpu_idle_w", "repetition", "api_cost_usd"]


def number(value: str):
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() and "." not in value else parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", type=Path)
    parser.add_argument("--out", type=Path, default=Path("dashboard/src/data/sembench_measured.json"))
    args = parser.parse_args()
    rows = []
    with args.table.open() as handle:
        for raw in csv.DictReader(handle):
            row = {key: (number(value) if key in NUMERIC else value) for key, value in raw.items()}
            row["valid"] = raw["valid"].lower() == "true"
            api = row["run"].startswith("sembench-openai")
            # API runs: the model runs at the provider, so there is no GPU measurement; showing the
            # engine's CPU share alone as "energy" would look misleadingly small.
            row["energy_source"] = "api" if api else "measured"
            gpu, cpu = row["gpu_wh"], row["cpu_job_wh"]
            row["energy_wh"] = None if api or (gpu is None and cpu is None) else (gpu or 0) + (cpu or 0)
            rows.append(row)
    snapshot = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "benchmark": "SemBench Movie, scale factor 1000 (1,000 reviews)",
        "hardware": "DGX A100-40GB (1 GPU per run), Ollama 4-bit models, OLLAMA_NUM_PARALLEL=4",
        "measurement": "GPU: NVML energy counter (incl. idle); CPU: perf energy-pkg x job CPU share; "
                       "LLM calls and tokens: logging proxy (same for every system)",
        "query_timeout_s": 3600,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=1) + "\n")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
