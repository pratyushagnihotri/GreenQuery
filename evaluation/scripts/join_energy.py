"""Join query windows, LLM call log and energy trace into per-query energy records.

Inputs (written by run_sembench_measured.sbatch):
  trace.jsonl    cumulative GPU / CPU-package energy and CPU times (energy_trace.py)
  calls.jsonl    one line per LLM call (llm_proxy.py)
  windows.jsonl  one line per measured window: {"kind": "query"|"idle", "system", "query",
                 "repetition", "model", "started_at", "finished_at", "exit_code", ...}

Output: one JSON record per query window with
  gpu_energy_j, gpu_idle_w, gpu_dynamic_energy_j, gpu_energy_in_llm_calls_j,
  cpu_job_energy_j (node package energy x job share of busy CPU time),
  llm_calls, prompt_tokens, completion_tokens, llm_active_s, window_s.
Every system is measured identically; nothing here depends on the engine.
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


class Series:
    """Cumulative counter sampled over time; value() interpolates linearly."""

    def __init__(self, samples: Sequence[Tuple[float, Optional[float]]]):
        clean = [(t, v) for t, v in samples if v is not None]
        self.t = [t for t, _ in clean]
        self.v = [v for _, v in clean]

    def value(self, when: float, tolerance: float = 1.0) -> Optional[float]:
        if not self.t:
            return None
        # Window edges a few samples outside the trace (process start/stop jitter) use the edge value.
        if self.t[0] - tolerance <= when < self.t[0]:
            when = self.t[0]
        elif self.t[-1] < when <= self.t[-1] + tolerance:
            when = self.t[-1]
        if when < self.t[0] or when > self.t[-1]:
            return None
        i = bisect_left(self.t, when)
        if self.t[i] == when or i == 0:
            return self.v[i]
        t0, t1, v0, v1 = self.t[i - 1], self.t[i], self.v[i - 1], self.v[i]
        return v0 + (v1 - v0) * (when - t0) / (t1 - t0)

    def delta(self, start: float, end: float) -> Optional[float]:
        a, b = self.value(start), self.value(end)
        return None if a is None or b is None else max(b - a, 0.0)


def merge(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[List[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def build_series(trace: List[Dict]) -> Dict[str, Series]:
    rows = [r for r in trace if not r.get("meta")]
    gpu_ids = sorted({g for r in rows if r.get("gpu_j") for g in r["gpu_j"]})
    gpu_total = [(r["t"], sum(r["gpu_j"].values()) if r.get("gpu_j") else None) for r in rows]
    series = {"gpu": Series(gpu_total)}
    for g in gpu_ids:
        series[f"gpu:{g}"] = Series([(r["t"], (r.get("gpu_j") or {}).get(g)) for r in rows])
    series["cpu_pkg"] = Series([(r["t"], r.get("cpu_pkg_j")) for r in rows])
    series["job_cpu"] = Series([(r["t"], r.get("job_cpu_s")) for r in rows])
    series["node_cpu"] = Series([(r["t"], r.get("node_busy_cpu_s")) for r in rows])
    return series


def cost_source(calls: List[Dict]) -> str:
    if not any(c.get("cost_usd") for c in calls):
        return "none"
    sources = {c.get("cost_source") for c in calls if c.get("cost_usd")}
    if sources == {"provider_billed"}:
        return "provider_billed"
    if None in sources:
        return "list_price_no_cache"
    return "tokens_x_price" if sources == {"tokens_x_price"} else "mixed"


def join(trace: List[Dict], calls: List[Dict], windows: List[Dict]) -> List[Dict]:
    series = build_series(trace)
    idle_w = None
    for window in windows:  # idle baseline: GPU power with models loaded, nothing running
        if window.get("kind") == "idle":
            energy = series["gpu"].delta(window["started_at"], window["finished_at"])
            if energy is not None:
                idle_w = energy / (window["finished_at"] - window["started_at"])
    records = []
    for window in windows:
        if window.get("kind") != "query":
            continue
        start, end = window["started_at"], window["finished_at"]
        duration = end - start
        inside = [c for c in calls if c.get("started_at", 0) >= start and c.get("finished_at", 0) <= end]
        intervals = merge([(c["started_at"], c["finished_at"]) for c in inside])
        gpu = series["gpu"].delta(start, end)
        gpu_llm = None
        if intervals and series["gpu"].t:
            parts = [series["gpu"].delta(s, e) for s, e in intervals]
            gpu_llm = sum(p for p in parts if p is not None)
        cpu_pkg = series["cpu_pkg"].delta(start, end)
        job_cpu = series["job_cpu"].delta(start, end)
        node_cpu = series["node_cpu"].delta(start, end)
        share = (job_cpu / node_cpu) if job_cpu is not None and node_cpu else None
        records.append({
            **{k: window.get(k) for k in ("system", "query", "repetition", "model", "served_model", "exit_code")},
            "window_s": duration,
            "gpu_energy_j": gpu,
            "gpu_idle_w": idle_w,
            "gpu_dynamic_energy_j": max(gpu - idle_w * duration, 0.0) if gpu is not None and idle_w is not None else None,
            "gpu_energy_in_llm_calls_j": gpu_llm,
            "gpu_energy_outside_llm_calls_j": (gpu - gpu_llm) if gpu is not None and gpu_llm is not None else None,
            "cpu_pkg_node_energy_j": cpu_pkg,
            "cpu_job_share": share,
            "cpu_job_energy_j": cpu_pkg * share if cpu_pkg is not None and share is not None else None,
            "llm_calls": len(inside),
            "llm_errors": sum(1 for c in inside if (c.get("status") or 0) >= 400),
            "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in inside),
            "completion_tokens": sum(c.get("completion_tokens") or 0 for c in inside),
            "llm_active_s": sum(e - s for s, e in intervals),
            "api_cost_usd": sum(c.get("cost_usd") or 0.0 for c in inside),
            # provider_billed (charged amount from the provider) / tokens_x_price (with cache discount when
            # logged) / list_price_no_cache (older logs: no cache data, an upper bound) / none (no API cost)
            "cost_source": cost_source(inside),
            "measurement": "gpu: NVML energy counter; cpu: perf energy-pkg x job CPU share; calls: llm_proxy",
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="folder with trace.jsonl, calls.jsonl, windows.jsonl")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    records = join(read_jsonl(args.run_dir / "trace.jsonl"), read_jsonl(args.run_dir / "calls.jsonl"),
                   read_jsonl(args.run_dir / "windows.jsonl"))
    out = args.out or args.run_dir / "energy_records.json"
    out.write_text(json.dumps(records, indent=2) + "\n")
    print(f"{out}: {len(records)} query windows")
    for r in records:
        gpu = r["gpu_energy_j"]
        print(f"  {r['system']:11} {str(r['query']):6} rep={r['repetition']} {r['window_s']:7.1f}s "
              f"GPU {gpu / 3600 if gpu else float('nan'):.3f} Wh  calls {r['llm_calls']:5}  "
              f"tokens {r['prompt_tokens'] + r['completion_tokens']}")


if __name__ == "__main__":
    main()
