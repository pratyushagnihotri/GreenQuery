"""Combine measured SemBench runs into one table: system x model x query.

  python evaluation/scripts/combine_sembench_measured.py tests/results/sembench-measured-* \
      --out reports/sembench_measured

Each run folder (run_sembench_measured.sbatch) holds run_config.txt, energy_records.json and
metrics/rep_<n>/<system>_Q<q>.json. Writes <out>.csv and <out>.md.

Quality is reported with its own metric: F1 for Q1/Q2/Q5, relative error for Q4 (lower is better).
`score` puts both on 0..1, higher is better (F1, or max(0, 1 - relative error)).

A row is `valid` only if the query finished in time and at most 1% of its LLM calls failed at
the proxy (engines retry single failures); invalid rows stay in the table with the reason, so failures remain data.
When several runs cover the same (system, model, query, repetition), the latest run wins.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

COLUMNS = ["system", "model", "query", "repetition", "status", "valid", "reason", "quality_metric",
           "quality", "score", "runtime_s", "llm_calls", "llm_errors", "prompt_tokens",
           "completion_tokens", "gpu_wh", "gpu_dynamic_wh", "cpu_job_wh", "gpu_idle_w", "api_cost_usd",
           "cost_source", "run"]


def read_config(run: Path) -> Dict[str, str]:
    path = run / "run_config.txt"
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def read_metric(run: Path, system: str, query: str, repetition: int) -> Dict:
    path = run / "metrics" / f"rep_{repetition}" / f"{system}_{query}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return next(iter(data.values()), {}) if data else {}


def quality(metric: Dict) -> tuple:
    if "f1_score" in metric:
        value = metric["f1_score"]
        return "f1", value, value
    if "relative_error" in metric:
        value = metric["relative_error"]
        return "relative_error", value, max(0.0, 1.0 - value)
    return "", None, None


def wh(joules: Optional[float]) -> Optional[float]:
    return None if joules is None else round(joules / 3600, 4)


def rows_for_run(run: Path) -> List[Dict]:
    records_path = run / "energy_records.json"
    if not records_path.exists():
        return []
    config = read_config(run)
    model = config.get("local_model", "")
    operators = [m.replace("ollama/", "") for m in config.get("operator_models", "").split(",") if m]
    if operators:  # multi-model runs are a separate configuration, not the same cell as single-model
        model = f"{model} | operators: {' + '.join(operators)}"
    if config.get("quality_target", "0.8") != "0.8":
        model = f"{model} | target {config['quality_target']}"
    rows = []
    for record in json.loads(records_path.read_text()):
        metric = read_metric(run, record["system"], record["query"], record["repetition"])
        name, value, score = quality(metric)
        reasons = []
        if record.get("exit_code") == 124:
            reasons.append("timeout")
        elif record.get("exit_code"):
            reasons.append(f"exit {record['exit_code']}")
        if record.get("llm_errors"):
            reasons.append(f"{record['llm_errors']}/{record['llm_calls']} LLM calls failed")
        if not metric:
            reasons.append("no metrics")
        elif metric.get("status") != "success":
            reasons.append(f"engine: {metric.get('error', metric.get('status'))}")
        # A few failed calls are retried by the engines; more than 1% means the run did not see the model.
        failed_share = (record.get("llm_errors") or 0) / max(record.get("llm_calls") or 0, 1)
        invalid = failed_share > 0.01 or any(r == "timeout" or r.startswith(("exit", "no metrics")) for r in reasons)
        rows.append({
            "system": record["system"], "model": model, "query": record["query"],
            "repetition": record["repetition"], "status": metric.get("status", ""),
            "valid": not invalid, "reason": "; ".join(reasons), "quality_metric": name,
            "quality": value, "score": score, "runtime_s": round(record["window_s"], 1),
            "llm_calls": record["llm_calls"], "llm_errors": record.get("llm_errors", 0),
            "prompt_tokens": record["prompt_tokens"], "completion_tokens": record["completion_tokens"],
            "gpu_wh": wh(record["gpu_energy_j"]), "gpu_dynamic_wh": wh(record["gpu_dynamic_energy_j"]),
            "cpu_job_wh": wh(record["cpu_job_energy_j"]),
            "gpu_idle_w": None if record["gpu_idle_w"] is None else round(record["gpu_idle_w"], 1),
            "api_cost_usd": None if record.get("api_cost_usd") is None else round(record["api_cost_usd"], 4),
            "cost_source": record.get("cost_source", ""),
            "run": run.name,
        })
    return rows


def combine(runs: List[Path]) -> List[Dict]:
    latest: Dict[tuple, Dict] = {}
    for run in sorted(runs, key=lambda p: p.name):  # job ids increase with time
        for row in rows_for_run(run):
            latest[(row["system"], row["model"], row["query"], row["repetition"])] = row
    return sorted(latest.values(), key=lambda r: (r["model"], r["query"], r["system"], r["repetition"]))


def fmt(value, digits: int = 2) -> str:
    if value is None or value == "":
        return "–"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def markdown(rows: List[Dict]) -> str:
    lines = ["| model | query | system | quality | runtime s | calls | tokens | GPU Wh | CPU Wh | API $ | note |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        q = f"{r['quality_metric'].replace('relative_error', 'rel.err')} {fmt(r['quality'])}" if r["quality_metric"] else "–"
        note = ("" if r["valid"] else "**invalid** ") + r["reason"]
        lines.append(f"| {r['model']} | {r['query']} | {r['system']} | {q} | {fmt(r['runtime_s'], 0)} | "
                     f"{r['llm_calls']} | {r['prompt_tokens'] + r['completion_tokens']} | {fmt(r['gpu_wh'])} | "
                     f"{fmt(r['cpu_job_wh'])} | {fmt(r['api_cost_usd'], 4) if r['api_cost_usd'] else '–'} | {note} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/sembench_measured"))
    args = parser.parse_args()
    rows = combine([run for run in args.runs if run.is_dir()])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix(".csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix(".md").write_text(markdown(rows))
    print(f"{len(rows)} rows ({sum(r['valid'] for r in rows)} valid) -> {args.out}.csv / .md")


if __name__ == "__main__":
    main()
