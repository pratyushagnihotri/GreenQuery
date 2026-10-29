"""Measurement validation (V4): trace overhead and NVML counter vs nvidia-smi power sampling.

  python evaluation/scripts/validate_measurement.py FAST_RUN SLOW_RUN

FAST_RUN: run_sembench_measured.sbatch with TRACE_INTERVAL=0.1 and NVSMI_SAMPLE=1.
SLOW_RUN: the same systems/queries/model with TRACE_INTERVAL=5 (near-zero tracing overhead).
Prints per query: runtime and GPU energy of both runs (overhead and run-to-run spread), and, for the
fast run, GPU energy from the NVML counter vs trapezoid-integrated nvidia-smi power samples.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path


def records(run: Path) -> dict:
    return {(r["system"], r["query"]): r for r in json.loads((run / "energy_records.json").read_text())}


def nvsmi_samples(path: Path) -> list:
    samples = []
    for line in path.read_text().splitlines():
        parts = [part.strip() for part in line.split(",")]
        try:
            stamp = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            samples.append((stamp, float(parts[1])))
        except (ValueError, IndexError):
            continue
    return samples


def integrate(samples: list, start: float, end: float) -> tuple:
    points = [(t, w) for t, w in samples if start <= t <= end]
    energy = sum((t2 - t1) * (w1 + w2) / 2 for (t1, w1), (t2, w2) in zip(points, points[1:]))
    return energy, len(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fast", type=Path)
    parser.add_argument("slow", type=Path)
    args = parser.parse_args()
    fast, slow = records(args.fast), records(args.slow)

    print("Trace overhead / run-to-run: 0.1 s vs 5 s sampling")
    print(f"{'system':11} {'query':5} {'runtime fast/slow (s)':>22} {'GPU Wh fast/slow':>18} {'diff':>7} {'calls':>11}")
    for key in sorted(fast):
        if key not in slow:
            continue
        f, s = fast[key], slow[key]
        diff = (f["gpu_energy_j"] - s["gpu_energy_j"]) / s["gpu_energy_j"] * 100
        print(f"{key[0]:11} {key[1]:5} {f['window_s']:10.1f}/{s['window_s']:<10.1f} "
              f"{f['gpu_energy_j'] / 3600:8.2f}/{s['gpu_energy_j'] / 3600:<8.2f} {diff:+6.1f}% "
              f"{f['llm_calls']:>5}/{s['llm_calls']:<5}")

    samples_path = args.fast / "nvsmi_power.csv"
    if not samples_path.exists():
        return
    samples = nvsmi_samples(samples_path)
    windows = [json.loads(line) for line in (args.fast / "windows.jsonl").read_text().splitlines() if line.strip()]
    print("\nNVML energy counter vs nvidia-smi power.draw (0.1 s, trapezoid)")
    for window in windows:
        if window["kind"] != "query":
            continue
        counter = fast[(window["system"], window["query"])]["gpu_energy_j"]
        sampled, n = integrate(samples, window["started_at"], window["finished_at"])
        print(f"{window['system']:11} {window['query']:5} counter {counter / 3600:7.3f} Wh   "
              f"nvidia-smi {sampled / 3600:7.3f} Wh   diff {(sampled - counter) / counter * 100:+5.1f}%   samples {n}")


if __name__ == "__main__":
    main()
