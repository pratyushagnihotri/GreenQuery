"""Continuous energy trace for a whole job (system-agnostic).

Writes one JSON line per interval with cumulative counters:

* ``gpu_j``: NVML total-energy counter per visible GPU (joules),
* ``cpu_pkg_j``: CPU package energy for the whole node from ``perf stat -a -e power/energy-pkg/``
  (works without root when perf_event_paranoid <= 0; includes other users' processes),
* ``job_cpu_s``: CPU time consumed by this SLURM job's cgroup, and ``node_busy_cpu_s``: busy CPU
  time of the whole node; job share of CPU energy = Δjob_cpu_s / Δnode_busy_cpu_s.

Query and LLM-call energy is computed afterwards from these cumulative series by
interpolating at window edges (see evaluation/scripts/join_energy.py). Unavailable
sources are recorded as null with a reason, never as zero.

Run:  python -m evaluation.measurement.energy_trace --out trace.jsonl --interval 0.1
Stop with SIGTERM/SIGINT.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional


class PerfPackageEnergy:
    """Cumulative node CPU package energy from `perf stat -I` interval output."""

    def __init__(self, interval_ms: int = 200):
        self.total_j: Optional[float] = None
        self.error: Optional[str] = None
        try:
            self.process = subprocess.Popen(
                ["perf", "stat", "-a", "-e", "power/energy-pkg/", "-I", str(interval_ms), "-x", ","],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                # C locale: otherwise e.g. German locales print "95,42" and break the CSV fields.
                env={**os.environ, "LC_ALL": "C"},
            )
        except FileNotFoundError:
            self.process = None
            self.error = "perf not installed"
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.process and self.process.stderr
        for line in self.process.stderr:
            parts = line.strip().split(",")
            # interval-mode CSV: time,value,unit,event,...
            if len(parts) >= 4 and "energy-pkg" in parts[3]:
                try:
                    delta = float(parts[1])
                except ValueError:
                    self.error = f"perf: {line.strip()[:120]}"
                    continue
                self.total_j = (self.total_j or 0.0) + delta
            elif line.strip() and ("not supported" in line or "Permission" in line):
                self.error = line.strip()[:160]

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()


def node_busy_cpu_seconds() -> Optional[float]:
    """Busy CPU time of the whole node (all users), from /proc/stat."""
    try:
        fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        values = [int(v) for v in fields]
    except (OSError, ValueError, IndexError):
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
    return (sum(values[:8]) - idle) / os.sysconf("SC_CLK_TCK")


def job_cgroup_cpu_seconds() -> Optional[float]:
    """CPU time of this process's cgroup (SLURM job), cgroup v2 then v1."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        hierarchy, controllers, path = line.split(":", 2)
        if hierarchy == "0":  # cgroup v2
            stat = Path("/sys/fs/cgroup") / path.lstrip("/") / "cpu.stat"
            try:
                for row in stat.read_text().splitlines():
                    if row.startswith("usage_usec"):
                        return int(row.split()[1]) / 1e6
            except OSError:
                pass
        elif "cpuacct" in controllers:
            usage = Path("/sys/fs/cgroup/cpu,cpuacct") / path.lstrip("/") / "cpuacct.usage"
            try:
                return int(usage.read_text()) / 1e9
            except OSError:
                pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--no-gpu", action="store_true", help="CPU only (API runs: GPUs belong to other jobs)")
    args = parser.parse_args()

    from evaluation.measurement.energy_counters import GpuEnergyReader

    raw = os.environ.get("GREENQUERY_GPU_INDEX", "")
    gpu = GpuEnergyReader([v.strip() for v in raw.split(",") if v.strip()] or None)
    if args.no_gpu:
        gpu.available, gpu.error = False, "disabled (--no-gpu)"
    perf = PerfPackageEnergy()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        handle.write(json.dumps({
            "meta": True, "interval_s": args.interval, "gpu_error": gpu.error,
            "gpu_devices": gpu.device_info() if gpu.available else {},
            "cpu_source": "perf power/energy-pkg (node-wide, all sockets)",
        }) + "\n")
        while not stop.is_set():
            record: Dict = {"t": time.time()}
            record["gpu_j"] = gpu.read() if gpu.available else None
            record["cpu_pkg_j"] = perf.total_j
            record["job_cpu_s"] = job_cgroup_cpu_seconds()
            record["node_busy_cpu_s"] = node_busy_cpu_seconds()
            if perf.error:
                record["cpu_error"] = perf.error
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            stop.wait(args.interval)
    perf.stop()


if __name__ == "__main__":
    main()
