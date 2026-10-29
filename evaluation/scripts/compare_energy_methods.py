"""Compare nvidia-smi power sampling with hardware energy counters.

Input: compact records from build_matched_demo_summary.py (one or more files).
Per (query, model) it reports, as medians over repetitions:

* sampler GPU energy vs NVML energy-counter GPU energy (relative difference),
* CPU package + DRAM share of the measured GPU+CPU+DRAM energy,
* share of GPU energy above the idle baseline (when an idle baseline was used),
* sampler overhead: counter energy with the sampler on vs off
  (records whose energy_source is the counter were run with the sampler off).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, Optional

SAMPLER_SOURCE = "measured_from_nvml_query_window"
COUNTER_ONLY_SOURCE = "measured_from_nvml_energy_counter_query_window"


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if isinstance(value, (int, float))]
    return median(clean) if clean else None


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:+.1f}%"


def _share(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def compare(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        groups[(record.get("demo"), record.get("query"), record.get("model"))].append(record)
    rows = []
    for (demo, query, model), items in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        with_sampler = [r for r in items if r.get("energy_source") == SAMPLER_SOURCE]
        counter_only = [r for r in items if r.get("energy_source") == COUNTER_ONLY_SOURCE]
        both = [r for r in with_sampler if r.get("counter_gpu_energy_kwh")]
        diff = _median(
            r["energy_kwh"] / r["counter_gpu_energy_kwh"] - 1 for r in both if r.get("energy_kwh")
        )
        cpu_share = _median(
            ((r.get("counter_cpu_package_energy_kwh") or 0) + (r.get("counter_dram_energy_kwh") or 0))
            / r["counter_measured_energy_kwh"]
            for r in items if r.get("counter_measured_energy_kwh")
        )
        dynamic_share = _median(
            r["counter_gpu_dynamic_energy_kwh"] / r["counter_gpu_energy_kwh"]
            for r in items if r.get("counter_gpu_dynamic_energy_kwh") is not None and r.get("counter_gpu_energy_kwh")
        )
        unattributed_share = _median(
            r["counter_unattributed_gpu_energy_kwh"] / r["counter_gpu_energy_kwh"]
            for r in items if r.get("counter_unattributed_gpu_energy_kwh") is not None and r.get("counter_gpu_energy_kwh")
        )
        on = _median(r.get("counter_measured_energy_kwh") for r in with_sampler)
        off = _median(r.get("counter_measured_energy_kwh") for r in counter_only)
        rows.append({
            "demo": demo, "query": query, "model": model,
            "runs_with_sampler": len(with_sampler), "runs_counter_only": len(counter_only),
            "sampler_vs_counter_gpu": diff,
            "cpu_dram_share": cpu_share,
            "gpu_above_idle_share": dynamic_share,
            "gpu_unattributed_share": unattributed_share,
            "sampler_overhead": (on / off - 1) if on and off else None,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", nargs="+", type=Path, help="compact query_metrics.json files")
    parser.add_argument("--json", type=Path, help="also write the comparison rows as JSON")
    args = parser.parse_args()
    records = [record for path in args.records for record in json.loads(path.read_text())]
    rows = compare(records)
    header = "| Demo | Query | Model | n (on/off) | Sampler vs counter (GPU) | CPU+DRAM share | GPU above idle | GPU unattributed | Sampler overhead |"
    print(header)
    print("|" + "---|" * 9)
    for row in rows:
        print(
            f"| {row['demo']} | {row['query']} | {row['model']} | {row['runs_with_sampler']}/{row['runs_counter_only']} "
            f"| {_pct(row['sampler_vs_counter_gpu'])} | {_share(row['cpu_dram_share'])} "
            f"| {_share(row['gpu_above_idle_share'])} | {_share(row['gpu_unattributed_share'])} "
            f"| {_pct(row['sampler_overhead'])} |"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
