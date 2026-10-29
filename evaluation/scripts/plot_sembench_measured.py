"""Paper figures from the combined SemBench table (combine_sembench_measured.py).

  python evaluation/scripts/plot_sembench_measured.py reports/sembench_measured.csv --out paper/edbt2027-demo/figures

fig_quality_energy.pdf  quality score vs total energy (GPU + CPU job share), one panel per query,
                        one marker per system, colour per model; timeouts/invalid rows hollow at
                        their energy so the cost of failing is visible.
fig_energy_by_system.pdf  energy per (system, query), bars per model, log scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SYSTEMS = ["stretto", "stretto_abacus", "lotus", "palimpzest", "thalamusdb", "caesura"]
LABELS = {"stretto": "Stretto", "stretto_abacus": "Abacus (in Stretto)", "lotus": "LOTUS",
          "palimpzest": "Palimpzest", "thalamusdb": "ThalamusDB", "caesura": "CAESURA"}
MARKERS = dict(zip(SYSTEMS, ["o", "s", "^", "D", "v", "P"]))
QUERY_TITLES = {"Q1": "Q1 filter + limit", "Q2": "Q2 filter (one movie)",
                "Q4": "Q4 positivity ratio", "Q5": "Q5 pair join + limit"}


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["gpu_wh"].notna()]  # measured local runs only (API runs have no GPU energy)
    frame = frame[~frame["model"].str.contains("operators:")]  # multi-model runs are a separate experiment
    frame["valid"] = frame["valid"].astype(str).str.lower() == "true"
    frame["energy_wh"] = frame["gpu_wh"].fillna(0) + frame["cpu_job_wh"].fillna(0)
    frame["score"] = frame["score"].fillna(0.0)
    return frame[frame["system"].isin(SYSTEMS)]


def quality_energy(frame: pd.DataFrame, out: Path) -> None:
    models = sorted(frame["model"].unique())
    colours = dict(zip(models, plt.cm.tab10.colors))
    queries = [q for q in QUERY_TITLES if q in set(frame["query"])]
    fig, axes = plt.subplots(1, len(queries), figsize=(3.2 * len(queries), 3.0), sharey=True)
    for ax, query in zip(axes, queries):
        part = frame[frame["query"] == query]
        for _, row in part.iterrows():
            ax.scatter(row["energy_wh"], row["score"], marker=MARKERS[row["system"]], s=42,
                       facecolors=colours[row["model"]] if row["valid"] else "none",
                       edgecolors=colours[row["model"]], linewidths=1.2)
        ax.set_xscale("log")
        ax.set_title(QUERY_TITLES[query], fontsize=9)
        ax.set_xlabel("energy per query [Wh, log]", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("quality (F1 / 1 − rel. error)", fontsize=8)
    handles = [plt.Line2D([], [], marker=MARKERS[s], linestyle="", color="grey", label=LABELS[s])
               for s in SYSTEMS if s in set(frame["system"])]
    handles += [plt.Line2D([], [], marker="o", linestyle="", color=colours[m], label=m) for m in models]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", color="grey",
                              label="timeout / invalid"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(out / "fig_quality_energy.pdf")
    fig.savefig(out / "fig_quality_energy.png", dpi=160)
    plt.close(fig)


def energy_by_system(frame: pd.DataFrame, out: Path) -> None:
    models = sorted(frame["model"].unique())
    colours = dict(zip(models, plt.cm.tab10.colors))
    queries = [q for q in QUERY_TITLES if q in set(frame["query"])]
    systems = [s for s in SYSTEMS if s in set(frame["system"])]
    fig, axes = plt.subplots(1, len(queries), figsize=(3.2 * len(queries), 3.0), sharey=True)
    width = 0.8 / len(models)
    for ax, query in zip(axes, queries):
        part = frame[frame["query"] == query]
        for k, model in enumerate(models):
            for i, system in enumerate(systems):
                row = part[(part["system"] == system) & (part["model"] == model)]
                if row.empty:
                    continue
                row = row.iloc[0]
                ax.bar(i + k * width - 0.4 + width / 2, row["energy_wh"], width, color=colours[model],
                       hatch=None if row["valid"] else "///", edgecolor="white" if row["valid"] else "black",
                       linewidth=0.4)
        ax.set_yscale("log")
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels([LABELS[s] for s in systems], rotation=45, ha="right", fontsize=7)
        ax.set_title(QUERY_TITLES[query], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel("energy per query [Wh, log]", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[m], label=m) for m in models]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch="///", edgecolor="black",
                                 label="timeout / invalid"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig_energy_by_system.pdf")
    fig.savefig(out / "fig_energy_by_system.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", type=Path)
    parser.add_argument("--out", type=Path, default=Path("paper/edbt2027-demo/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = load(args.table)
    quality_energy(frame, args.out)
    energy_by_system(frame, args.out)
    print(f"{len(frame)} rows -> {args.out}/fig_quality_energy.pdf, fig_energy_by_system.pdf")


if __name__ == "__main__":
    main()
