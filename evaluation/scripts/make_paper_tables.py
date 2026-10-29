"""LaTeX tables (booktabs) for the paper, generated from the combined results.

  python evaluation/scripts/make_paper_tables.py --out paper/edbt2027-demo/tables

tab_cross_system.tex  energy (Wh) / quality per system and query for one local model (default Qwen 2.5 7B)
tab_api_models.tex    API models: correct answers, output tokens per call, total API cost
tab_multi_model.tex   multi-model optimizers: share of calls on the smaller model, energy, quality (Q1, Q2)

Quality = F1 (Q1, Q2, Q5) or 1 - relative error (Q4); "T/O" = 1 h timeout, "err" = engine error,
"n/r" = not run. Include with \\input{tables/<name>}; requires \\usepackage{booktabs}.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SYSTEMS = ["thalamusdb", "palimpzest", "lotus", "stretto", "stretto_abacus", "caesura"]
LABEL = {"stretto": "Stretto", "stretto_abacus": "Abacus (in Stretto)", "lotus": "LOTUS",
         "palimpzest": "Palimpzest", "thalamusdb": "ThalamusDB", "caesura": "CAESURA"}
QUERIES = ["Q1", "Q2", "Q4", "Q5"]
API_MODELS = [("gpt-4o-mini", "gpt-4o-mini"), ("gpt-5-mini", "gpt-5-mini"), ("gpt-4.1", "gpt-4.1"),
              ("anthropic/claude-haiku-5.5", "Claude Haiku 5.5"), ("anthropic/claude-sonnet-5.5", "Claude Sonnet 5.5"),
              ("anthropic/claude-opus-5.5", "Claude Opus 5.5")]


def wh(value: float) -> str:
    return f"{value:.2f}" if value < 1 else f"{value:.1f}" if value < 10 else f"{value:.0f}"


def cell(row) -> str:
    if row is None:
        return "n/r"
    if row.get("runs", 1) > 1:  # median over repetitions, range in brackets
        low, high = row["energy_range"]
        quality = "T/O" if row["timeouts"] == row["runs"] else f"{row['score']:.2f}"
        flag = rf"$^{{{row['timeouts']}\times\text{{T/O}}}}$" if 0 < row["timeouts"] < row["runs"] else ""
        return f"{wh(row['energy_wh'])} [{wh(low)}--{wh(high)}] / {quality}{flag}"
    if "timeout" in str(row["reason"]):
        return f"{wh(row['energy_wh'])} / T/O"
    if row["status"] != "success":
        return f"{wh(row['energy_wh'])} / err"
    return f"{wh(row['energy_wh'])} / {row['score']:.2f}"


def find(frame, model, system, query):
    """One row per cell; with repetitions, energy/quality become medians and `energy_range` the min-max."""
    match = frame[(frame["model"] == model) & (frame["system"] == system) & (frame["query"] == query)]
    if match.empty:
        return None
    row = match.sort_values("repetition").iloc[0].copy()
    if len(match) > 1:
        row["energy_wh"] = match["energy_wh"].median()
        row["score"] = match["score"].median()
        row["energy_range"] = (match["energy_wh"].min(), match["energy_wh"].max())
        row["runs"] = len(match)
        row["timeouts"] = int(match["reason"].astype(str).str.contains("timeout").sum())
    return row


def cross_system(frame: pd.DataFrame, model: str, label: str) -> str:
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             "System & " + " & ".join(QUERIES) + r" \\", r"\midrule"]
    for system in SYSTEMS:
        lines.append(LABEL[system] + " & " + " & ".join(cell(find(frame, model, system, q)) for q in QUERIES) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = (f"Energy per query (Wh, GPU incl.\\ idle + CPU share) / quality with {label} on one A100. "
               r"Quality: F1 (Q1, Q2, Q5), $1-$relative error (Q4); T/O: 1\,h timeout; err: engine error. "
               r"Q1 and Q5: median of 3 runs [min--max].")
    return wrap("\n".join(lines), caption, "tab:cross-system")


def api_models(frame: pd.DataFrame, tokens: dict) -> str:
    lines = [r"\begin{tabular}{@{}lccr@{}}", r"\toprule",
             r"Model & correct (of 24) & output tok./call & API cost \\", r"\midrule"]
    for model, label in API_MODELS:
        part = frame[frame["model"] == model]
        correct = int(((part["status"] == "success") & part["valid"] & (part["score"] >= 0.8)).sum())
        ran = len(part)
        cost = part["api_cost_usd"].fillna(0).sum()
        out = tokens.get(model)
        tok = "--" if out is None else f"{out:.0f}"
        of = "" if ran == 24 else f" (of {ran})"
        lines.append(f"{label} & {correct}{of} & {tok} & \\${cost:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = (r"API models over 6 systems $\times$ 4 queries: answers with quality $\geq 0.8$, average output "
               r"tokens per LLM call, and actual API cost (provider-billed for Claude via OpenRouter, "
               r"tokens $\times$ list price for OpenAI).")
    return wrap("\n".join(lines), caption, "tab:api-models")


def multi_model(choices: list) -> str:
    def name(operators):
        short = (operators.replace("qwen2.5:", "Qwen ").replace("-instruct", "").replace("gemma3:", "Gemma ")
                 .replace("llama3.2:", "Llama ").replace("llama3.1:", "Llama "))
        return re.sub(r"(\d+)b\b", r"\1B", short)

    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"Operators (target) & Optimizer & small model (Q1) & Q1 Wh & F1 Q1 / Q2 \\", r"\midrule"]
    setups = []
    for row in choices:
        key = (row["operators"], row["target"])
        if key not in setups:
            setups.append(key)
    for operators, target in setups:
        for system, label in (("stretto", "Stretto"), ("stretto_abacus", "Abacus")):
            q = {r["query"]: r for r in choices if r["operators"] == operators and r["target"] == target
                 and r["system"] == system}
            q1, q2 = q.get("Q1"), q.get("Q2")
            def f1(r):
                return "err" if r is None or r["status"] != "success" else f"{r['quality']:.2f}"
            energy = (q1["gpu_wh"] or 0) + (q1["cpu_wh"] or 0) if q1 else float("nan")
            first = f"{name(operators)} ({target})" if system == "stretto" else ""
            lines.append(f"{first} & {label} & {q1['small_share']:.0%} & {wh(energy)} & {f1(q1)} / {f1(q2)}"
                         .replace("%", r"\%") + r" \\")
        lines.append(r"\addlinespace")
    lines[-1:] = [r"\bottomrule", r"\end{tabular}"]
    caption = ("Multi-model operators: share of LLM calls served by the smaller model on Q1, Q1 energy "
               "(Wh, GPU + CPU share), and quality. Stretto's optimizer never moves the bulk to the smaller "
               "model; Abacus does for Gemma 4B+27B and at target 0.6.")
    return wrap("\n".join(lines), caption, "tab:multi-model")


def wrap(body: str, caption: str, label: str) -> str:
    return (r"\begin{table}[t]" + "\n" + r"\caption{" + caption + "}\n" + rf"\label{{{label}}}" + "\n" +
            r"\small" + "\n" + r"\resizebox{\columnwidth}{!}{%" + "\n" + body + "}\n" + r"\end{table}" + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, default=Path("reports/sembench_measured.csv"))
    parser.add_argument("--choices", type=Path, default=Path("reports/multi_model_choices.json"))
    parser.add_argument("--tokens", type=Path, default=Path("reports/api_tokens_per_call.json"),
                        help="average output tokens per call per API model (computed from the call logs)")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--model-label", default="Qwen~2.5 7B")
    parser.add_argument("--out", type=Path, default=Path("paper/edbt2027-demo/tables"))
    args = parser.parse_args()
    frame = pd.read_csv(args.table)
    frame["valid"] = frame["valid"].astype(str).str.lower() == "true"
    frame["energy_wh"] = frame["gpu_wh"] + frame["cpu_job_wh"].fillna(0)
    frame["score"] = frame["score"].fillna(0.0)
    tokens = json.loads(args.tokens.read_text()) if args.tokens.exists() else {}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tab_cross_system.tex").write_text(cross_system(frame, args.model, args.model_label))
    (args.out / "tab_api_models.tex").write_text(api_models(frame, tokens))
    (args.out / "tab_multi_model.tex").write_text(multi_model(json.loads(args.choices.read_text())))
    print(f"tables -> {args.out}")


if __name__ == "__main__":
    main()
