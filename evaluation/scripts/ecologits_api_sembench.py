"""EcoLogits energy and CO2 estimates for the SemBench API-model runs in the dashboard snapshot.

  uv run --no-project --with ecologits==0.11.1 python evaluation/scripts/ecologits_api_sembench.py \
      <dir with sembench-openai-* run folders> --out reports/ecologits_api_sembench.csv

API runs are not measured (inference runs at the provider). For every snapshot row with
energy_source == "api", sum EcoLogits `llm_impacts` over the successful calls inside its query window
(run, system, query, repetition), using each call's completion tokens and latency (EcoLogits' only
inputs). Claude 5.5 is not registered in EcoLogits 0.11.1, so the Claude 4.5 tier is a labelled proxy.
Results are estimates, never measurements: they include EcoLogits' data-centre PUE and its
electricity-mix CO2, and embodied CO2 is reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ecologits import __version__ as ECOLOGITS_VERSION
from ecologits.tracers.utils import llm_impacts

SNAPSHOT = Path(__file__).resolve().parents[2] / "dashboard" / "src" / "data" / "sembench_measured.json"
# served model -> (EcoLogits provider, EcoLogits model); same proxies as config/models.yaml.
ECOLOGITS_MODEL = {
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4.1": ("openai", "gpt-4.1"),
    "gpt-5-mini": ("openai", "gpt-5-mini"),
    "anthropic/claude-haiku-5.5": ("anthropic", "claude-haiku-4-5"),
    "anthropic/claude-sonnet-5.5": ("anthropic", "claude-sonnet-4-5"),
    "anthropic/claude-opus-5.5": ("anthropic", "claude-opus-4-5"),
}


def span(value, scale=1.0):
    raw = getattr(value, "value", value)
    low, high = float(getattr(raw, "min", raw)) * scale, float(getattr(raw, "max", raw)) * scale
    return low, high


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/ecologits_api_sembench.csv"))
    args = parser.parse_args()
    rows = [r for r in json.loads(SNAPSHOT.read_text())["rows"] if r.get("energy_source") == "api"]
    cache: dict[str, tuple[list, list]] = {}
    out = []
    for row in rows:
        run = args.runs_dir / row["run"]
        if row["run"] not in cache:
            cache[row["run"]] = (read_jsonl(run / "windows.jsonl"), read_jsonl(run / "calls.jsonl"))
        windows, calls = cache[row["run"]]
        window = next((w for w in windows if w["kind"] == "query" and w["system"] == row["system"]
                       and w["query"] == row["query"] and int(w.get("repetition", 1)) == int(row["repetition"])), None)
        provider, eco_model = ECOLOGITS_MODEL[row["model"]]
        inside = [c for c in calls if window and window["started_at"] <= c["started_at"] <= window["finished_at"]
                  and (c.get("status") or 0) < 400 and c.get("completion_tokens") is not None]
        energy, usage_co2, embodied_co2, failed = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0
        for call in inside:
            impacts = llm_impacts(provider=provider, model_name=eco_model,
                                  output_token_count=int(call["completion_tokens"]),
                                  request_latency=max(call["finished_at"] - call["started_at"], 0.001))
            if impacts is None or getattr(impacts, "has_errors", False):
                failed += 1
                continue
            for total, (low, high) in ((energy, span(impacts.energy, 1000)),  # kWh -> Wh
                                       (usage_co2, span(impacts.usage.gwp, 1000)),  # kg -> g
                                       (embodied_co2, span(impacts.embodied.gwp, 1000))):
                total[0] += low
                total[1] += high
        ok = bool(inside) and not failed
        mid = lambda pair: round((pair[0] + pair[1]) / 2, 6) if ok else None  # noqa: E731
        rng = lambda pair, i: round(pair[i], 6) if ok else None  # noqa: E731
        out.append({
            "run": row["run"], "system": row["system"], "query": row["query"], "repetition": row["repetition"],
            "model": row["model"], "ecologits_model": f"{provider}/{eco_model}",
            "proxy": eco_model not in row["model"], "status": row["status"], "valid": row["valid"],
            "score": row.get("score"), "runtime_s": row.get("runtime_s"),
            "snapshot_llm_calls": row.get("llm_calls"), "window_calls": len(inside), "calls_without_estimate": failed,
            "input_tokens": sum(int(c.get("prompt_tokens") or 0) for c in inside),
            "output_tokens": sum(int(c["completion_tokens"]) for c in inside),
            "est_energy_wh": mid(energy), "est_energy_wh_min": rng(energy, 0), "est_energy_wh_max": rng(energy, 1),
            "est_usage_co2_g": mid(usage_co2), "est_embodied_co2_g": mid(embodied_co2),
            "api_cost_usd": row.get("api_cost_usd"), "energy_source": f"estimated_by_ecologits_{ECOLOGITS_VERSION}",
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    missing = sum(r["est_energy_wh"] is None for r in out)
    print(f"{len(out)} API runs -> {args.out} (EcoLogits {ECOLOGITS_VERSION}); without estimate: {missing}")


if __name__ == "__main__":
    main()
