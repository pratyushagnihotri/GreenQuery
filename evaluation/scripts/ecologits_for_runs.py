"""EcoLogits energy estimates per query window, from the logged calls (V5).

  python evaluation/scripts/ecologits_for_runs.py tests/results/sembench-openai-* tests/results/sembench-measured-* \
      --out reports/ecologits_estimates.csv

For every call in a query window: EcoLogits `llm_impacts` with the call's completion tokens and latency
(the only inputs EcoLogits uses). API runs: EcoLogits' OpenAI model profile. Local runs: EcoLogits'
energy model with the local model's parameter count and PUE 1, compared with measured GPU energy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from reasondb.reasoning.providers import ecologits_estimate

# Local models are mostly not in EcoLogits' repository (no Qwen 2.5, Gemma 3, Llama 3.2), so for them we
# apply EcoLogits' own model (compute_llm_impacts) with the model's parameter count (billions, dense)
# and PUE = 1 (no data-centre overhead), comparable to our GPU measurement.
PARAMS_B = {
    "qwen2.5:3b-instruct": 3.09, "qwen2.5:7b-instruct": 7.62, "qwen2.5:14b-instruct": 14.7,
    "qwen2.5:32b-instruct": 32.5, "llama3.2:1b": 1.24, "llama3.2:3b": 3.21, "llama3.1:8b": 8.03,
    "gemma3:4b": 4.3, "gemma3:12b": 12.2, "gemma3:27b": 27.4,
}


CLAUDE_PROXY = {  # served model -> EcoLogits stand-in (config/models.yaml ecologits_model)
    "anthropic/claude-haiku-5.5": "claude-haiku-4-5",
    "anthropic/claude-sonnet-5.5": "claude-sonnet-4-5",
    "anthropic/claude-opus-5.5": "claude-opus-4-5",
}


def estimate_kwh(served: str, output_tokens: int, latency: float):
    """kWh for one call: EcoLogits method (parameter count) for local models, EcoLogits API model otherwise."""
    if served in PARAMS_B:
        from ecologits.impacts.llm import compute_llm_impacts
        impacts = compute_llm_impacts(
            model_active_parameter_count=PARAMS_B[served], model_total_parameter_count=PARAMS_B[served],
            output_token_count=output_tokens, request_latency=latency, datacenter_pue=1.0, datacenter_wue=0.0,
            if_electricity_mix_adpe=0.0, if_electricity_mix_pe=0.0, if_electricity_mix_gwp=0.0,
            if_electricity_mix_wue=0.0)
        value = getattr(impacts.energy, "value", impacts.energy)
        low, high = getattr(value, "min", value), getattr(value, "max", value)
        return (float(low) + float(high)) / 2
    if served.startswith("anthropic/"):
        # EcoLogits has no Claude 5.5 profiles; use the registry's labelled stand-in (Claude 4.5 tier).
        return ecologits_estimate("anthropic", CLAUDE_PROXY[served], output_tokens, latency)[0]
    return ecologits_estimate("openai", served, output_tokens, latency)[0]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/ecologits_estimates.csv"))
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        config = dict(line.split("=", 1) for line in (run / "run_config.txt").read_text().splitlines()
                      if "=" in line) if (run / "run_config.txt").exists() else {}
        if config.get("operator_models"):
            continue  # multi-model runs mix models per query; skip
        calls = read_jsonl(run / "calls.jsonl")
        measured = {(r["system"], r["query"]): r for r in json.loads((run / "energy_records.json").read_text())} \
            if (run / "energy_records.json").exists() else {}
        for window in (w for w in read_jsonl(run / "windows.jsonl") if w["kind"] == "query"):
            inside = [c for c in calls if window["started_at"] <= c["started_at"] <= window["finished_at"]
                      and (c.get("status") or 0) < 400 and c.get("completion_tokens") is not None]
            total, unknown, served = 0.0, 0, None
            for call in inside:
                served = call.get("served_model")
                kwh = estimate_kwh(served, int(call["completion_tokens"]),
                                   max(call["finished_at"] - call["started_at"], 0.001))
                if kwh is None:
                    unknown += 1
                else:
                    total += kwh
            record = measured.get((window["system"], window["query"]), {})
            rows.append({
                "run": run.name, "system": window["system"], "query": window["query"],
                "model": served or config.get("served_model") or config.get("local_model"),
                "calls": len(inside), "calls_without_estimate": unknown,
                "output_tokens": sum(int(c["completion_tokens"]) for c in inside),
                "input_tokens": sum(int(c.get("prompt_tokens") or 0) for c in inside),
                "ecologits_wh": round(total * 1000, 4) if inside and unknown < len(inside) else None,
                "measured_gpu_wh": round(record["gpu_energy_j"] / 3600, 4) if record.get("gpu_energy_j") and
                not run.name.startswith("sembench-openai") else None,
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} query windows -> {args.out}")


if __name__ == "__main__":
    main()
