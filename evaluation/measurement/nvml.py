"""Lightweight GPU energy sampling through nvidia-smi.

The sampler is deliberately best-effort. If the current process does not have a
visible GPU allocation or power telemetry is unavailable, callers receive a
`not_available` result instead of fabricated energy values.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


PHASE_ENERGY_PROVENANCE = "nvml_non_overlapping_local_inference_windows_v1"
ROLE_PHASE_ENERGY_PROVENANCE = "nvml_per_gpu_role_local_inference_windows_v2"


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(windows):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def derive_phase_windows(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build disjoint planner/execution windows from local inference timestamps."""
    intervals: list[tuple[str, float, float]] = []
    invalid_events = 0
    for event in events:
        try:
            start = float(event["inference_started_at"])
            end = float(event["inference_finished_at"])
        except (KeyError, TypeError, ValueError):
            invalid_events += 1
            continue
        if end <= start:
            invalid_events += 1
            continue
        phase = "planner" if event.get("event_kind") == "planner" else "execution"
        intervals.append((phase, start, end))

    phase_windows = {"planner": [], "execution": []}
    cross_phase_overlap_seconds = 0.0
    boundaries = sorted({value for _, start, end in intervals for value in (start, end)})
    for start, end in zip(boundaries, boundaries[1:]):
        phases = {
            phase
            for phase, interval_start, interval_end in intervals
            if interval_start < end and interval_end > start
        }
        if len(phases) == 1:
            phase_windows[phases.pop()].append((start, end))
        elif len(phases) > 1:
            cross_phase_overlap_seconds += end - start

    phase_windows = {
        phase: _merge_windows(windows) for phase, windows in phase_windows.items()
    }
    reasons = []
    if invalid_events:
        reasons.append("local_inference_event_missing_valid_timestamps")
    if cross_phase_overlap_seconds:
        reasons.append("planner_execution_windows_overlap")
    for phase in phase_windows:
        if not phase_windows[phase]:
            reasons.append(f"missing_{phase}_window")
    return {
        "windows": phase_windows,
        "valid": not reasons,
        "invalid_reasons": reasons,
        "invalid_event_count": invalid_events,
        "cross_phase_overlap_seconds": cross_phase_overlap_seconds,
    }


@dataclass
class NVMLMeasurement:
    energy_kwh: float
    co2_g: float
    energy_source: str
    co2_source: str
    sample_count: int
    error: Optional[str] = None
    sample_started_at: Optional[float] = None
    sample_finished_at: Optional[float] = None
    per_gpu_energy_kwh: Optional[dict[str, float]] = None


class NvidiaSmiEnergySampler:
    """Sample GPU power draw and integrate it over a query window."""

    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, float | dict[str, float]]] = []
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> NVMLMeasurement:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))

        if len(self.samples) < 2:
            return NVMLMeasurement(
                energy_kwh=0.0,
                co2_g=0.0,
                energy_source="not_available",
                co2_source="not_available",
                sample_count=len(self.samples),
                error=self.error or "fewer_than_two_power_samples",
                sample_started_at=self.samples[0][0] if self.samples else None,
                sample_finished_at=self.samples[-1][0] if self.samples else None,
            )

        start, end = self.samples[0][0], self.samples[-1][0]
        energy_kwh, _ = self.energy_between(start, end)
        gpu_ids = sorted({gpu for _, powers in self.samples if isinstance(powers, dict) for gpu in powers})
        per_gpu_energy_kwh = {
            gpu: self.energy_between(start, end, gpu)[0] for gpu in gpu_ids
        }

        factor_raw = os.getenv("GREENQUERY_CO2_G_PER_KWH")
        if factor_raw:
            try:
                co2_g = energy_kwh * float(factor_raw)
                co2_source = "estimated_from_configured_emissions_factor"
            except ValueError:
                co2_g = 0.0
                co2_source = "not_available"
                self.error = f"invalid GREENQUERY_CO2_G_PER_KWH={factor_raw!r}"
        else:
            co2_g = 0.0
            co2_source = "not_available"

        return NVMLMeasurement(
            energy_kwh=energy_kwh,
            co2_g=co2_g,
            energy_source="measured_from_nvml",
            co2_source=co2_source,
            sample_count=len(self.samples),
            error=self.error,
            sample_started_at=start,
            sample_finished_at=end,
            per_gpu_energy_kwh=per_gpu_energy_kwh or None,
        )

    @staticmethod
    def _power(sample: float | dict[str, float], gpu_id: str | None = None) -> float:
        if isinstance(sample, dict):
            return sample.get(gpu_id, 0.0) if gpu_id is not None else sum(sample.values())
        return sample if gpu_id is None else 0.0

    def energy_between(
        self, start: float, end: float, gpu_id: str | None = None
    ) -> tuple[float, int]:
        """Integrate samples whose intervals overlap the requested window."""
        if end <= start or len(self.samples) < 2:
            return 0.0, 0
        watt_seconds = 0.0
        used = set()
        for index, ((t0, sample0), (t1, sample1)) in enumerate(
            zip(self.samples, self.samples[1:])
        ):
            overlap_start = max(t0, start)
            overlap_end = min(t1, end)
            if overlap_end <= overlap_start or t1 <= t0:
                continue
            p0 = self._power(sample0, gpu_id)
            p1 = self._power(sample1, gpu_id)
            start_ratio = (overlap_start - t0) / (t1 - t0)
            end_ratio = (overlap_end - t0) / (t1 - t0)
            power_start = p0 + (p1 - p0) * start_ratio
            power_end = p0 + (p1 - p0) * end_ratio
            watt_seconds += (
                (power_start + power_end) / 2.0
            ) * (overlap_end - overlap_start)
            used.update((index, index + 1))
        return watt_seconds / 3_600_000.0, len(used)

    def energy_over_windows(
        self, windows: list[tuple[float, float]], gpu_id: str | None = None
    ) -> tuple[float, int]:
        """Integrate the union of timestamp windows without double counting."""
        merged = _merge_windows(windows)
        energy_kwh = 0.0
        used_samples: set[int] = set()
        for start, end in merged:
            window_energy, _ = self.energy_between(start, end, gpu_id)
            energy_kwh += window_energy
            for index, ((t0, _), (t1, _)) in enumerate(
                zip(self.samples, self.samples[1:])
            ):
                if max(t0, start) < min(t1, end):
                    used_samples.update((index, index + 1))
        return energy_kwh, len(used_samples)

    def serialized_samples(self) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": timestamp,
                "power_watts_by_gpu": (
                    dict(powers) if isinstance(powers, dict) else {"aggregate": powers}
                ),
            }
            for timestamp, powers in self.samples
        ]

    def _sample_loop(self):
        while not self._stop.is_set():
            try:
                command = [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,power.draw",
                    "--format=csv,noheader,nounits",
                ]
                gpu_indices = os.getenv("GREENQUERY_GPU_INDEX")
                if gpu_indices:
                    command.insert(1, f"--id={gpu_indices}")

                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                requested = [value.strip() for value in (gpu_indices or "").split(",") if value.strip()]
                powers = {}
                for line in completed.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 3 or parts[2] == "[Not Supported]":
                        continue
                    index, uuid, power = parts
                    key = next((value for value in requested if value in {index, uuid}), index)
                    powers[key] = float(power)
                if powers:
                    self.samples.append((time.time(), powers))
            except Exception as exc:
                self.error = str(exc)
                return
            self._stop.wait(self.interval_seconds)


def build_phase_energy_metrics(
    events: list[dict[str, Any]],
    sampler: NvidiaSmiEnergySampler,
    query_energy_kwh: float,
    query_co2_g: float,
    co2_source: str,
    gpu_roles: dict[str, str] | None = None,
    planner_required: bool = True,
) -> dict[str, Any]:
    """Measure disjoint planner/execution inference energy within one query."""
    if gpu_roles:
        return _build_role_phase_energy_metrics(
            events, sampler, query_energy_kwh, query_co2_g, co2_source, gpu_roles,
            planner_required,
        )
    derived = derive_phase_windows(events)
    if query_energy_kwh <= 0 or len(sampler.samples) < 2:
        derived["valid"] = False
        derived["invalid_reasons"].append("query_nvml_measurement_not_available")
    factor = query_co2_g / query_energy_kwh if query_energy_kwh else 0.0
    phases: dict[str, Any] = {}
    for phase, windows in derived["windows"].items():
        energy_kwh, sample_count = sampler.energy_over_windows(windows)
        phases[phase] = {
            "energy_kwh": energy_kwh,
            "co2_g": energy_kwh * factor,
            "sample_count": sample_count,
            "window_count": len(windows),
            "duration_seconds": sum(end - start for start, end in windows),
            "windows": [
                {"started_at": start, "finished_at": end}
                for start, end in windows
            ],
        }

    attributed = sum(item["energy_kwh"] for item in phases.values())
    if attributed > query_energy_kwh and attributed > 0:
        scale = query_energy_kwh / attributed
        for item in phases.values():
            item["energy_kwh"] *= scale
            item["co2_g"] *= scale
        attributed = query_energy_kwh
        derived["valid"] = False
        derived["invalid_reasons"].append("phase_energy_exceeds_query_energy")

    unattributed_energy = max(query_energy_kwh - attributed, 0.0)
    return {
        "provenance": PHASE_ENERGY_PROVENANCE,
        "valid": derived["valid"],
        "invalid_reasons": derived["invalid_reasons"],
        "invalid_event_count": derived["invalid_event_count"],
        "cross_phase_overlap_seconds": derived["cross_phase_overlap_seconds"],
        "energy_source": "measured_from_nvml_phase_windows",
        "co2_source": co2_source,
        "planner": phases["planner"],
        "execution": phases["execution"],
        "unattributed_query_energy_kwh": unattributed_energy,
        "unattributed_query_co2_g": unattributed_energy * factor,
    }


def _build_role_phase_energy_metrics(
    events: list[dict[str, Any]],
    sampler: NvidiaSmiEnergySampler,
    query_energy_kwh: float,
    query_co2_g: float,
    co2_source: str,
    gpu_roles: dict[str, str],
    planner_required: bool = True,
) -> dict[str, Any]:
    """Attribute each local inference event to its assigned physical GPU role."""
    roles = {str(role): str(gpu) for role, gpu in gpu_roles.items()}
    reasons = []
    if "text" not in roles or not all(roles.values()) or len(set(roles.values())) != len(roles):
        reasons.append("invalid_gpu_role_mapping")

    windows = {
        "planner": {"text": []},
        "execution": {role: [] for role in roles},
    }
    invalid_events = 0
    for event in events:
        try:
            interval = (
                float(event["inference_started_at"]),
                float(event["inference_finished_at"]),
            )
        except (KeyError, TypeError, ValueError):
            invalid_events += 1
            continue
        if interval[1] <= interval[0]:
            invalid_events += 1
        elif event.get("event_kind") == "planner" and event.get("execution_mode") == "local_planner_generation":
            windows["planner"]["text"].append(interval)
        elif event.get("execution_mode") == "direct_no_cache":
            role = event.get("event_kind") if event.get("event_kind") in roles else "text"
            if role in roles:
                windows["execution"][role].append(interval)
            else:
                invalid_events += 1
        else:
            invalid_events += 1

    if invalid_events:
        reasons.append("unclassified_or_invalid_local_inference_event")
    if planner_required and not windows["planner"]["text"]:
        reasons.append("missing_planner_window")
    if not any(windows["execution"].values()):
        reasons.append("missing_execution_window")

    available_gpus = {
        gpu for _, powers in sampler.samples if isinstance(powers, dict) for gpu in powers
    }
    if any(gpu not in available_gpus for gpu in roles.values()):
        reasons.append("gpu_role_missing_from_nvml_samples")

    planner_windows = _merge_windows(windows["planner"]["text"])
    execution_windows = {
        role: _merge_windows(role_windows)
        for role, role_windows in windows["execution"].items()
    }
    overlap_seconds = sum(
        max(0.0, min(p_end, e_end) - max(p_start, e_start))
        for p_start, p_end in planner_windows
        for e_start, e_end in execution_windows.get("text", [])
    )
    if overlap_seconds:
        reasons.append("text_gpu_planner_execution_windows_overlap")

    factor = query_co2_g / query_energy_kwh if query_energy_kwh else 0.0
    planner_energy, planner_samples = sampler.energy_over_windows(
        planner_windows, roles.get("text", "")
    )
    execution_by_role = {
        role: sampler.energy_over_windows(role_windows, roles[role])[0]
        for role, role_windows in execution_windows.items()
    }
    execution_energy = sum(execution_by_role.values())
    all_execution_windows = [window for role_windows in execution_windows.values() for window in role_windows]
    _, execution_samples = sampler.energy_over_windows(all_execution_windows)
    attributed = planner_energy + execution_energy
    if query_energy_kwh <= 0 or len(sampler.samples) < 2:
        reasons.append("query_nvml_measurement_not_available")
    if attributed > query_energy_kwh + 1e-12:
        reasons.append("phase_energy_exceeds_query_energy")
    unattributed = max(query_energy_kwh - attributed, 0.0)

    def phase(energy: float, sample_count: int, role_energy: dict[str, float], role_windows: dict[str, list[tuple[float, float]]]) -> dict[str, Any]:
        merged = _merge_windows([window for items in role_windows.values() for window in items])
        return {
            "energy_kwh": energy,
            "co2_g": energy * factor,
            "sample_count": sample_count,
            "window_count": len(merged),
            "duration_seconds": sum(end - start for start, end in merged),
            "energy_kwh_by_gpu_role": role_energy,
            "windows_by_gpu_role": {
                role: [{"started_at": start, "finished_at": end} for start, end in items]
                for role, items in role_windows.items()
            },
        }

    per_gpu_query_energy = {
        gpu: sampler.energy_between(sampler.samples[0][0], sampler.samples[-1][0], gpu)[0]
        for gpu in available_gpus
    } if len(sampler.samples) >= 2 else {}
    return {
        "provenance": ROLE_PHASE_ENERGY_PROVENANCE,
        "valid": not reasons,
        "invalid_reasons": reasons,
        "invalid_event_count": invalid_events,
        "cross_phase_overlap_seconds": overlap_seconds,
        "energy_source": "measured_from_nvml_per_gpu_role_windows",
        "co2_source": co2_source,
        "gpu_roles": roles,
        "planner_required": planner_required,
        "query_energy_kwh_by_gpu": per_gpu_query_energy,
        "planner": phase(planner_energy, planner_samples, {"text": planner_energy}, {"text": planner_windows}),
        "execution": phase(execution_energy, execution_samples, execution_by_role, execution_windows),
        "unattributed_query_energy_kwh": unattributed,
        "unattributed_query_co2_g": unattributed * factor,
    }
