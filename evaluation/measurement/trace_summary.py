"""Condensed power timeline and call activity of one query window (run inspector).

A query window can span an hour, with an energy-trace line every 0.1 s and tens of thousands of LLM
calls. summarize_window() bins it into at most MAX_POINTS time bins:
  t      bin start, seconds since the window start
  w      average GPU power in the bin (W), from the cumulative NVML counter
  busy   fraction of the bin with at least one LLM call in flight
  calls  LLM calls started in the bin
  tok    tokens (prompt + completion) of the calls started in the bin
plus the first FIRST_CALLS calls in detail and the idle baseline. Used by
evaluation/scripts/export_run_traces.py (recorded runs) and the GreenQuery service (live runs).
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional

MAX_POINTS = 600
FIRST_CALLS = 40


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _gpu_series(trace: List[Dict]) -> tuple:
    times, joules = [], []
    for row in trace:
        if row.get("meta") or not row.get("gpu_j"):
            continue
        times.append(row["t"])
        joules.append(float(sum(row["gpu_j"].values())))
    return times, joules


def _value_at(times: List[float], values: List[float], when: float) -> Optional[float]:
    if not times or when < times[0] or when > times[-1]:
        return None
    i = bisect_left(times, when)
    if times[i] == when or i == 0:
        return values[i]
    t0, t1 = times[i - 1], times[i]
    return values[i - 1] + (values[i] - values[i - 1]) * (when - t0) / (t1 - t0)


def summarize_window(trace: List[Dict], calls: List[Dict], start: float, end: float,
                     idle_w: Optional[float] = None) -> Dict:
    duration = max(end - start, 1e-6)
    step = max(1, math.ceil(duration / MAX_POINTS))
    bins = max(1, math.ceil(duration / step))
    times, joules = _gpu_series(trace)
    inside = sorted((c for c in calls if start <= c.get("started_at", 0) <= end), key=lambda c: c["started_at"])

    points = []
    for k in range(bins):
        b0, b1 = start + k * step, min(start + (k + 1) * step, end)
        e0, e1 = _value_at(times, joules, b0), _value_at(times, joules, b1)
        power = (e1 - e0) / (b1 - b0) if e0 is not None and e1 is not None and b1 > b0 else None
        points.append({"t": round(k * step, 1), "w": None if power is None else round(max(power, 0.0), 1),
                       "busy": 0.0, "calls": 0, "tok": 0})
    for call in inside:
        s, f = call["started_at"], min(call.get("finished_at", call["started_at"]), end)
        k = min(int((s - start) // step), bins - 1)
        points[k]["calls"] += 1
        points[k]["tok"] += (call.get("prompt_tokens") or 0) + (call.get("completion_tokens") or 0)
        # in-flight time per bin (union of overlapping calls approximated by clipping at 1)
        first, last = k, min(int((f - start) // step), bins - 1)
        for j in range(first, last + 1):
            b0, b1 = start + j * step, start + (j + 1) * step
            overlap = max(0.0, min(f, b1) - max(s, b0))
            points[j]["busy"] = min(1.0, points[j]["busy"] + overlap / step)
    for point in points:
        point["busy"] = round(point["busy"], 2)

    detail = [{
        "start": round(c["started_at"] - start, 2),
        "dur": round(c.get("finished_at", c["started_at"]) - c["started_at"], 2),
        "in": c.get("prompt_tokens"), "out": c.get("completion_tokens"),
        "status": c.get("status"), "model": c.get("served_model"),
    } for c in inside[:FIRST_CALLS]]
    return {"duration_s": round(duration, 1), "step_s": step, "idle_w": None if idle_w is None else round(idle_w, 1),
            "calls_total": len(inside), "points": points, "first_calls": detail}


def summarize_run_dir(run_dir: Path) -> List[Dict]:
    """One summary per query window of a measured run folder (run_sembench_measured.sbatch layout)."""
    trace = _read_jsonl(run_dir / "trace.jsonl")
    calls = _read_jsonl(run_dir / "calls.jsonl")
    windows = _read_jsonl(run_dir / "windows.jsonl")
    times, joules = _gpu_series(trace)
    idle_w = None
    for w in windows:
        if w.get("kind") == "idle":
            e0, e1 = _value_at(times, joules, w["started_at"]), _value_at(times, joules, w["finished_at"])
            if e0 is not None and e1 is not None:
                idle_w = (e1 - e0) / (w["finished_at"] - w["started_at"])
    out = []
    for w in windows:
        if w.get("kind") != "query":
            continue
        summary = summarize_window(trace, calls, w["started_at"], w["finished_at"], idle_w)
        out.append({"system": w.get("system"), "query": w.get("query"), "repetition": w.get("repetition"), **summary})
    return out
