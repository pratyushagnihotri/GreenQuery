"""Cumulative hardware energy counters: NVML GPU energy and Linux RAPL.

Runs next to the nvidia-smi power sampler (``nvml.py``) so both methods measure
the same windows. Unlike power sampling, these counters report energy directly:

* GPU: ``nvmlDeviceGetTotalEnergyConsumption`` (mJ since driver load, Volta+).
* CPU package / DRAM: ``/sys/class/powercap/*-rapl*/energy_uj`` (µJ, wraps at
  ``max_energy_range_uj``). Package counters cover the whole socket, i.e. every
  process on the node, not only this benchmark.

Readings are taken in-process every ``interval_seconds`` (no subprocesses), and
the energy of any window is the interpolated counter difference at its edges.
Unreadable counters are reported as unavailable, never as zero.

CLI: ``python -m evaluation.measurement.energy_counters idle --seconds 60 --out idle.json``
measures the idle baseline used to report energy above idle.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

POWERCAP_ROOT = Path("/sys/class/powercap")
METHOD = "nvml_total_energy_counter+rapl_powercap_v1"
JOULES_PER_KWH = 3_600_000.0


def counters_enabled() -> bool:
    return os.environ.get("GREENQUERY_ENERGY_COUNTERS", "0") == "1"


# --------------------------------------------------------------------- readers
class GpuEnergyReader:
    """Reads NVML cumulative energy (joules) for the selected GPUs."""

    def __init__(self, gpu_indices: Optional[List[str]] = None):
        self.error: Optional[str] = None
        self.handles: Dict[str, Any] = {}
        try:
            import pynvml

            self._nvml = pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            wanted = gpu_indices or [str(index) for index in range(count)]
            for key in wanted:
                handle = (
                    pynvml.nvmlDeviceGetHandleByIndex(int(key))
                    if key.isdigit()
                    else pynvml.nvmlDeviceGetHandleByUUID(key)
                )
                pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # unsupported -> raises
                self.handles[key] = handle
        except Exception as error:  # no driver, no GPU, pre-Volta, missing pynvml
            self.error = f"nvml_energy_counter_unavailable: {error}"
            self.handles = {}

    @property
    def available(self) -> bool:
        return bool(self.handles)

    def read(self) -> Dict[str, float]:
        return {
            key: self._nvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0
            for key, handle in self.handles.items()
        }

    def device_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for key, handle in self.handles.items():
            try:
                name = self._nvml.nvmlDeviceGetName(handle)
                info[key] = {"name": name.decode() if isinstance(name, bytes) else name}
            except Exception:
                info[key] = {}
        try:
            driver = self._nvml.nvmlSystemGetDriverVersion()
            info["driver_version"] = driver.decode() if isinstance(driver, bytes) else driver
        except Exception:
            pass
        return info


@dataclass
class RaplDomain:
    name: str  # e.g. "package-0", "package-0/dram"
    kind: str  # "package" | "dram" | other subdomain name
    energy_path: Path
    max_range_uj: int


class RaplReader:
    """Reads RAPL energy (joules) with wraparound handling."""

    def __init__(self, root: Path = POWERCAP_ROOT):
        self.error: Optional[str] = None
        self.domains: List[RaplDomain] = []
        self._last: Dict[str, int] = {}
        self._offset: Dict[str, int] = {}
        if not root.exists():
            self.error = "rapl_unavailable: no /sys/class/powercap"
            return
        for zone in sorted(root.glob("*-rapl:*")):
            if zone.name.count(":") != 1:
                continue  # subzones are visited from their parent
            self._add(zone, parent=None)
            for sub in sorted(zone.glob(f"{zone.name}:*")):
                self._add(sub, parent=self._name(zone))
        if not self.domains and not self.error:
            self.error = "rapl_unavailable: no readable RAPL zones (check permissions on energy_uj)"

    @staticmethod
    def _name(zone: Path) -> str:
        return (zone / "name").read_text().strip()

    def _add(self, zone: Path, parent: Optional[str]) -> None:
        try:
            label = self._name(zone)
            energy = zone / "energy_uj"
            int(energy.read_text())  # permission check
            max_range = int((zone / "max_energy_range_uj").read_text())
        except (OSError, ValueError) as error:
            self.error = f"rapl_unreadable: {zone}: {error}"
            return
        name = f"{parent}/{label}" if parent else label
        kind = "dram" if label == "dram" else ("package" if label.startswith("package") else label)
        self.domains.append(RaplDomain(name, kind, energy, max_range))

    @property
    def available(self) -> bool:
        return bool(self.domains)

    def read(self) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for domain in self.domains:
            raw = int(domain.energy_path.read_text())
            last = self._last.get(domain.name)
            if last is not None and raw < last:
                self._offset[domain.name] = self._offset.get(domain.name, 0) + domain.max_range_uj
            self._last[domain.name] = raw
            values[domain.name] = (raw + self._offset.get(domain.name, 0)) / 1e6
        return values


# --------------------------------------------------------------------- sampler
@dataclass
class CounterSample:
    timestamp: float
    gpu_joules: Dict[str, float]
    rapl_joules: Dict[str, float]


@dataclass
class CounterSampler:
    """Background thread recording cumulative counters at a fixed interval."""

    interval_seconds: float = 0.1
    gpu_indices: Optional[List[str]] = None
    samples: List[CounterSample] = field(default_factory=list)
    error: Optional[str] = None

    def __post_init__(self):
        if self.gpu_indices is None:
            raw = os.environ.get("GREENQUERY_GPU_INDEX", "")
            self.gpu_indices = [value.strip() for value in raw.split(",") if value.strip()] or None
        self.gpu = GpuEnergyReader(self.gpu_indices)
        self.rapl = RaplReader()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def available(self) -> bool:
        return self.gpu.available or self.rapl.available

    def _sample(self) -> None:
        self.samples.append(
            CounterSample(
                time.time(),
                self.gpu.read() if self.gpu.available else {},
                self.rapl.read() if self.rapl.available else {},
            )
        )

    def start(self) -> "CounterSampler":
        if not self.available:
            return self
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._sample()
            except Exception as error:  # keep what we have, but say so
                self.error = f"counter_read_failed: {error}"
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.available and self.error is None:
            try:
                self._sample()
            except Exception as error:
                self.error = f"counter_read_failed: {error}"

    # ------------------------------------------------------------ integration
    def _value_at(self, when: float, getter) -> Optional[float]:
        """Linearly interpolate a cumulative counter at ``when``."""
        times = [sample.timestamp for sample in self.samples]
        if not times or when < times[0] or when > times[-1]:
            return None
        index = bisect_left(times, when)
        if times[index] == when:
            return getter(self.samples[index])
        before, after = self.samples[index - 1], self.samples[index]
        v0, v1 = getter(before), getter(after)
        if v0 is None or v1 is None:
            return None
        ratio = (when - before.timestamp) / (after.timestamp - before.timestamp)
        return v0 + (v1 - v0) * ratio

    def joules_between(self, start: float, end: float) -> Dict[str, Dict[str, float]]:
        """Energy per GPU and per RAPL domain in [start, end]; missing keys = unavailable."""
        result: Dict[str, Dict[str, float]] = {"gpu": {}, "rapl": {}}
        if not self.samples or end <= start:
            return result
        start = max(start, self.samples[0].timestamp)
        end = min(end, self.samples[-1].timestamp)
        if end <= start:
            return result
        for group, attr in (("gpu", "gpu_joules"), ("rapl", "rapl_joules")):
            for key in getattr(self.samples[-1], attr):
                getter = lambda sample, key=key, attr=attr: getattr(sample, attr).get(key)
                v0, v1 = self._value_at(start, getter), self._value_at(end, getter)
                if v0 is not None and v1 is not None:
                    result[group][key] = max(v1 - v0, 0.0)
        return result

    def joules_over_windows(self, windows: List[Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
        total: Dict[str, Dict[str, float]] = {"gpu": {}, "rapl": {}}
        for start, end in windows:
            part = self.joules_between(start, end)
            for group in total:
                for key, value in part[group].items():
                    total[group][key] = total[group].get(key, 0.0) + value
        return total


# --------------------------------------------------------------------- summary
def _kwh(joules: Dict[str, float]) -> Dict[str, float]:
    return {key: value / JOULES_PER_KWH for key, value in joules.items()}


def _rapl_by_kind(rapl: Dict[str, float], domains: List[RaplDomain]) -> Dict[str, float]:
    kinds: Dict[str, float] = {}
    for domain in domains:
        if domain.name in rapl:
            kinds[domain.kind] = kinds.get(domain.kind, 0.0) + rapl[domain.name]
    return kinds


def load_idle_baseline() -> Optional[Dict[str, Any]]:
    path = os.environ.get("GREENQUERY_IDLE_BASELINE")
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def summarize_window(
    sampler: CounterSampler,
    start: float,
    end: float,
    phase_windows: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    co2_g_per_kwh: Optional[float] = None,
) -> Dict[str, Any]:
    """Query-level (and optional per-phase) counter energy, gross and above idle."""
    duration = max(end - start, 0.0)
    joules = sampler.joules_between(start, end)
    gpu_kwh = sum(_kwh(joules["gpu"]).values()) if joules["gpu"] else None
    kinds = _rapl_by_kind(joules["rapl"], sampler.rapl.domains)
    cpu_kwh = kinds["package"] / JOULES_PER_KWH if "package" in kinds else None
    dram_kwh = kinds["dram"] / JOULES_PER_KWH if "dram" in kinds else None
    # DRAM is a subdomain reported alongside (not inside) the package counter.
    node_parts = [value for value in (gpu_kwh, cpu_kwh, dram_kwh) if value is not None]
    measured_kwh = sum(node_parts) if gpu_kwh is not None and cpu_kwh is not None else None

    summary: Dict[str, Any] = {
        "method": METHOD,
        "interval_seconds": sampler.interval_seconds,
        "sample_count": len(sampler.samples),
        "window_seconds": duration,
        "error": sampler.error,
        "gpu_error": sampler.gpu.error,
        "rapl_error": sampler.rapl.error,
        "gpu_devices": sampler.gpu.device_info() if sampler.gpu.available else {},
        "gpu_energy_kwh": gpu_kwh,
        "gpu_energy_kwh_by_gpu": _kwh(joules["gpu"]) or None,
        "cpu_package_energy_kwh": cpu_kwh,
        "dram_energy_kwh": dram_kwh,
        "rapl_energy_kwh_by_domain": _kwh(joules["rapl"]) or None,
        # GPU + CPU packages + DRAM. Excludes fans, NICs, storage, PSU loss and PUE.
        "measured_energy_kwh": measured_kwh,
        "measured_boundary": "gpu+cpu_package+dram" if measured_kwh is not None else None,
    }

    idle = load_idle_baseline()
    if idle:
        idle_gpu_w = sum((idle.get("gpu_watts_by_gpu") or {}).get(key, 0.0) for key in joules["gpu"])
        idle_cpu_w = (idle.get("rapl_watts_by_kind") or {}).get("package")
        idle_dram_w = (idle.get("rapl_watts_by_kind") or {}).get("dram")
        def above(value, watts):
            if value is None or watts is None:
                return None
            return max(value - watts * duration / JOULES_PER_KWH, 0.0)
        summary["idle_baseline"] = {
            "source": os.environ.get("GREENQUERY_IDLE_BASELINE"),
            "gpu_watts": idle_gpu_w, "cpu_package_watts": idle_cpu_w, "dram_watts": idle_dram_w,
            "measured_at": idle.get("measured_at"), "seconds": idle.get("seconds"),
        }
        summary["gpu_dynamic_energy_kwh"] = above(gpu_kwh, idle_gpu_w)
        summary["cpu_package_dynamic_energy_kwh"] = above(cpu_kwh, idle_cpu_w)
        summary["dram_dynamic_energy_kwh"] = above(dram_kwh, idle_dram_w)

    if phase_windows:
        phases: Dict[str, Any] = {}
        for phase, windows in phase_windows.items():
            part = sampler.joules_over_windows(windows)
            phase_kinds = _rapl_by_kind(part["rapl"], sampler.rapl.domains)
            phases[phase] = {
                "gpu_energy_kwh": sum(part["gpu"].values()) / JOULES_PER_KWH if part["gpu"] else None,
                "cpu_package_energy_kwh": phase_kinds["package"] / JOULES_PER_KWH if "package" in phase_kinds else None,
                "duration_seconds": sum(end_ - start_ for start_, end_ in windows),
            }
        attributed = sum(phase["gpu_energy_kwh"] or 0.0 for phase in phases.values())
        phases["unattributed_gpu_energy_kwh"] = max(gpu_kwh - attributed, 0.0) if gpu_kwh is not None else None
        summary["phases"] = phases

    if co2_g_per_kwh is not None:
        summary["co2_g_per_kwh"] = co2_g_per_kwh
        summary["gpu_co2_g"] = gpu_kwh * co2_g_per_kwh if gpu_kwh is not None else None
        summary["measured_co2_g"] = measured_kwh * co2_g_per_kwh if measured_kwh is not None else None
        summary["co2_scope"] = "operational_location_based"
    return summary


# ------------------------------------------------------------------------- CLI
def measure_idle(seconds: float, interval: float = 0.5) -> Dict[str, Any]:
    sampler = CounterSampler(interval_seconds=interval).start()
    if not sampler.available:
        raise SystemExit(f"No energy counters available: {sampler.gpu.error}; {sampler.rapl.error}")
    time.sleep(seconds)
    sampler.stop()
    start, end = sampler.samples[0].timestamp, sampler.samples[-1].timestamp
    joules = sampler.joules_between(start, end)
    duration = end - start
    return {
        "method": METHOD,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": duration,
        "gpu_watts_by_gpu": {key: value / duration for key, value in joules["gpu"].items()},
        "rapl_watts_by_domain": {key: value / duration for key, value in joules["rapl"].items()},
        "rapl_watts_by_kind": {
            key: value / duration for key, value in _rapl_by_kind(joules["rapl"], sampler.rapl.domains).items()
        },
        "gpu_devices": sampler.gpu.device_info(),
        "gpu_error": sampler.gpu.error,
        "rapl_error": sampler.rapl.error,
        "note": "Measure with models loaded and no queries running.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    idle = sub.add_parser("idle", help="Measure idle power from the counters")
    idle.add_argument("--seconds", type=float, default=60.0)
    idle.add_argument("--out", type=Path, required=True)
    probe = sub.add_parser("probe", help="Report which counters are readable on this machine")
    args = parser.parse_args()
    if args.command == "probe":
        sampler = CounterSampler()
        print(json.dumps({
            "gpu_available": sampler.gpu.available, "gpu_error": sampler.gpu.error,
            "gpu_devices": sampler.gpu.device_info() if sampler.gpu.available else {},
            "rapl_available": sampler.rapl.available, "rapl_error": sampler.rapl.error,
            "rapl_domains": [domain.name for domain in sampler.rapl.domains],
        }, indent=2))
        return
    result = measure_idle(args.seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
