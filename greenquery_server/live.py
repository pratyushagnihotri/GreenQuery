"""Live progress of a running measured SemBench job (Challenge mode).

Reads the job's output folder while run_sembench_measured.sbatch is running:
  trace.jsonl    cumulative GPU energy (NVML counter), one line per 0.1 s
  windows.jsonl  the idle-baseline window (written before the queries start) and finished query windows
  calls.jsonl    one line per LLM call (logging proxy)
Energy so far = GPU counter now - GPU counter at the end of the idle baseline (GPU only; the final
record adds the CPU share after join_energy.py has run). The heartbeat is the GPU power of the last
HEARTBEAT_S seconds (1 s buckets from the counter) with the start times of the LLM calls in that span.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HEARTBEAT_S = 180
TAIL_BYTES = 512 * 1024  # about 6 min of trace at 0.1 s
# job output dir -> (idle end time, GPU joules at that time, idle power in W); found once per job.
_baseline_cache: Dict[str, Tuple[float, float, Optional[float]]] = {}


def _gpu_joules(line: str) -> Optional[Tuple[float, float]]:
    try:
        row = json.loads(line)
    except ValueError:
        return None
    if row.get("meta") or not row.get("gpu_j"):
        return None
    return row["t"], float(sum(row["gpu_j"].values()))


def _tail_points(path: Path, size: int = TAIL_BYTES) -> List[Tuple[float, float]]:
    """Complete trace points at the end of the file (the trace grows to megabytes)."""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        total = handle.tell()
        handle.seek(max(total - size, 0))
        lines = handle.read().decode(errors="replace").splitlines()
    points = [_gpu_joules(line) for line in (lines[1:] if total > size else lines)]
    return [point for point in points if point]


def _baseline(out_dir: Path, idle: dict) -> Tuple[Optional[float], Optional[float]]:
    """GPU joules at the end of the idle baseline and the mean idle power."""
    key, idle_start, idle_end = str(out_dir), idle["started_at"], idle["finished_at"]
    if key in _baseline_cache and _baseline_cache[key][0] == idle_end:
        return _baseline_cache[key][1], _baseline_cache[key][2]
    first = None
    with (out_dir / "trace.jsonl").open() as handle:
        for line in handle:
            point = _gpu_joules(line)
            if not point or point[0] < idle_start:
                continue
            first = first or point
            if point[0] >= idle_end:
                idle_w = (point[1] - first[1]) / (point[0] - first[0]) if point[0] > first[0] else None
                _baseline_cache[key] = (idle_end, point[1], idle_w)
                return point[1], idle_w
    return None, None


def _heartbeat(points: List[Tuple[float, float]], origin: float) -> List[List[float]]:
    """GPU power in 1 s buckets over the last HEARTBEAT_S seconds, as [seconds since origin, W]."""
    if len(points) < 2:
        return []
    since = points[-1][0] - HEARTBEAT_S
    beats, bucket = [], None
    for (t0, j0), (t1, j1) in zip(points, points[1:]):
        if t1 < since or t1 <= t0:
            continue
        second = int(t1 - origin)
        if bucket is None or bucket[0] != second:
            if bucket:
                beats.append([float(bucket[0]), round(bucket[1] / bucket[2], 1)])
            bucket = [second, 0.0, 0.0]
        bucket[1] += j1 - j0
        bucket[2] += t1 - t0
    if bucket and bucket[2] > 0:
        beats.append([float(bucket[0]), round(bucket[1] / bucket[2], 1)])
    return beats


def live_progress(out_dir: Path) -> dict:
    """Phase, energy so far (Wh, GPU), LLM calls and tokens of a measured job."""
    progress = {"phase": "waiting for a GPU", "energy_wh": 0.0, "llm_calls": 0, "llm_errors": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "elapsed_s": 0.0, "done": False,
                "power": [], "call_starts": [], "idle_w": None, "power_w": None}
    if not (out_dir / "trace.jsonl").exists():
        if (out_dir / "pulling").exists():
            progress["phase"] = f"downloading {(out_dir / 'pulling').read_text().strip()}"
        elif (out_dir / "run_config.txt").exists():
            progress["phase"] = "starting the model server"
        return progress
    windows = []
    if (out_dir / "windows.jsonl").exists():
        windows = [json.loads(line) for line in (out_dir / "windows.jsonl").read_text().splitlines() if line.strip()]
    idle = next((w for w in windows if w.get("kind") == "idle"), None)
    points = _tail_points(out_dir / "trace.jsonl")
    if idle is None:
        progress["phase"] = "measuring the idle baseline"
        # Seconds before the query starts are negative; the origin is fixed once the idle window is known.
        if points:
            progress["power"] = _heartbeat(points, points[-1][0])
            progress["power_w"] = progress["power"][-1][1] if progress["power"] else None
        return progress
    progress["phase"] = "running the query"
    origin = idle["finished_at"]
    if (out_dir / "calls.jsonl").exists():
        for line in (out_dir / "calls.jsonl").read_text().splitlines():
            try:
                call = json.loads(line)
            except ValueError:
                continue
            if call.get("started_at", 0) < idle["finished_at"]:
                continue  # model warm-up before the query
            progress["llm_calls"] += 1
            progress["llm_errors"] += int((call.get("status") or 0) >= 400)
            progress["prompt_tokens"] += call.get("prompt_tokens") or 0
            progress["completion_tokens"] += call.get("completion_tokens") or 0
            if points and call["started_at"] >= points[-1][0] - HEARTBEAT_S:
                progress["call_starts"].append(round(call["started_at"] - origin, 1))
    last = points[-1] if points else None
    start, progress["idle_w"] = _baseline(out_dir, idle)
    progress["power"] = _heartbeat(points, origin)
    progress["power_w"] = progress["power"][-1][1] if progress["power"] else None
    if last and start is not None:
        progress["energy_wh"] = max(last[1] - start, 0.0) / 3600
        progress["elapsed_s"] = max(last[0] - origin, 0.0)
    if any(w.get("kind") == "query" for w in windows):
        progress["phase"] = "scoring and joining energy"
    if (out_dir / "energy_records.json").exists():
        progress["phase"] = "finished"
        progress["done"] = True
    return progress
