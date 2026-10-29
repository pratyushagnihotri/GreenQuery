"""Paper figures from the combined results (reports/sembench_measured.csv, reports/multi_model_choices.json).

  python evaluation/scripts/make_paper_figures.py --out paper/edbt2027-demo/figures

fig_quality_energy.pdf   quality vs measured energy, per query; three main local models (colour), systems (marker)
fig_model_size.pdf       measured energy vs model size (Qwen 2.5 3B-32B) per system, Q1/Q2/Q4
fig_api_cost.pdf         API cost per query per system for the six API models (log scale)
fig_multi_model.pdf      multi-model optimizers: Stretto vs Abacus energy per setup, small-model share labelled
fig_architecture.pdf     platform architecture

Colours: validated categorical palette (dataviz reference instance); <= 3 colours in scatter plots,
markers/labels as secondary encoding everywhere. Quality = F1 (Q1/Q2/Q5) or 1 - relative error (Q4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SYSTEMS = ["stretto", "stretto_abacus", "lotus", "palimpzest", "thalamusdb", "caesura"]
LABEL = {"stretto": "Stretto", "stretto_abacus": "Abacus", "lotus": "LOTUS", "palimpzest": "Palimpzest",
         "thalamusdb": "ThalamusDB", "caesura": "CAESURA"}
MARKER = dict(zip(SYSTEMS, ["o", "s", "^", "D", "v", "P"]))
QUERY = {"Q1": "Q1 filter + limit", "Q2": "Q2 filter, one movie", "Q4": "Q4 positivity ratio",
         "Q5": "Q5 pair join + limit"}
MAIN_MODELS = ["qwen2.5:7b-instruct", "gemma3:12b", "llama3.1:8b"]
MODEL_LABEL = {"qwen2.5:7b-instruct": "Qwen 2.5 7B", "gemma3:12b": "Gemma 3 12B", "llama3.1:8b": "Llama 3.1 8B",
               "gpt-4o-mini": "gpt-4o-mini", "gpt-5-mini": "gpt-5-mini", "gpt-4.1": "gpt-4.1",
               "anthropic/claude-haiku-5.5": "Claude Haiku 5.5", "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
               "anthropic/claude-opus-5.5": "Claude Opus 5.5"}
API_MODELS = ["gpt-4o-mini", "gpt-5-mini", "gpt-4.1", "anthropic/claude-haiku-5.5",
              "anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5"]
QWEN = [("qwen2.5:3b-instruct", 3), ("qwen2.5:7b-instruct", 7), ("qwen2.5:14b-instruct", 14),
        ("qwen2.5:32b-instruct", 32)]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["valid"] = frame["valid"].astype(str).str.lower() == "true"
    frame["energy_wh"] = frame["gpu_wh"] + frame["cpu_job_wh"].fillna(0)
    frame["score"] = frame["score"].fillna(0.0)
    return frame


def grid(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def legend_below(fig, handles, ncol) -> None:
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=ncol, fontsize=6.5, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.3, columnspacing=1.0)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def quality_energy(frame: pd.DataFrame, out: Path) -> None:
    data = frame[frame["model"].isin(MAIN_MODELS) & frame["gpu_wh"].notna()]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=True)
    for ax, query in zip(axes, QUERY):
        part = data[data["query"] == query]
        for _, row in part.iterrows():
            colour = PALETTE[MAIN_MODELS.index(row["model"])]
            ax.scatter(row["energy_wh"], row["score"], marker=MARKER[row["system"]], s=26, linewidths=1.0,
                       facecolors=colour if row["valid"] else "none", edgecolors=colour, zorder=3)
        ax.set_xscale("log")
        ax.set_title(QUERY[query], fontsize=8, color=INK)
        ax.set_xlabel("energy per query [Wh]")
        grid(ax)
    axes[0].set_ylabel("quality")
    axes[0].set_ylim(-0.05, 1.05)
    handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[i], label=MODEL_LABEL[m])
               for i, m in enumerate(MAIN_MODELS)]
    handles += [plt.Line2D([], [], marker=MARKER[s], ls="", color=MUTED, label=LABEL[s]) for s in SYSTEMS]
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", color=MUTED, label="timeout/invalid"))
    legend_below(fig, handles, 5)
    save(fig, out, "fig_quality_energy")


def model_size(frame: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.0), sharey=True)
    for ax, query in zip(axes, ["Q1", "Q2", "Q4"]):
        for i, system in enumerate(SYSTEMS):
            xs, ys, valid = [], [], []
            for model, size in QWEN:
                row = frame[(frame["model"] == model) & (frame["system"] == system) & (frame["query"] == query)]
                if row.empty or row["energy_wh"].isna().all():
                    continue
                xs.append(size)
                ys.append(row["energy_wh"].median())  # median over repetitions where present
                valid.append(bool(row.iloc[0]["valid"]) and row.iloc[0]["status"] == "success")
            if not xs:
                continue
            ax.plot(xs, ys, color=PALETTE[i], linewidth=1.2, zorder=2)
            for x, y, ok in zip(xs, ys, valid):
                ax.scatter(x, y, marker=MARKER[system], s=20, zorder=3, color=PALETTE[i],
                           facecolors=PALETTE[i] if ok else "white", linewidths=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([3, 7, 14, 32])
        ax.set_xticklabels(["3B", "7B", "14B", "32B"])
        ax.minorticks_off()
        ax.set_title(QUERY[query], fontsize=8, color=INK)
        ax.set_xlabel("Qwen 2.5 model size")
        grid(ax)
    axes[0].set_ylabel("energy per query [Wh]")
    handles = [plt.Line2D([], [], marker=MARKER[s], color=PALETTE[i], label=LABEL[s]) for i, s in enumerate(SYSTEMS)]
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", color=MUTED,
                              label="engine error / invalid"))
    legend_below(fig, handles, 4)
    save(fig, out, "fig_model_size")


def api_cost(frame: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=True)
    width = 0.8 / len(API_MODELS)
    for ax, query in zip(axes, QUERY):
        for k, model in enumerate(API_MODELS):
            for j, system in enumerate(SYSTEMS):
                row = frame[(frame["model"] == model) & (frame["system"] == system) & (frame["query"] == query)]
                if row.empty or not row.iloc[0]["api_cost_usd"] > 0:
                    continue
                ax.bar(j - 0.4 + width * (k + 0.5), row.iloc[0]["api_cost_usd"], width * 0.85,
                       color=PALETTE[k], edgecolor="white", linewidth=0.3)
        ax.set_yscale("log")
        ax.set_xticks(range(len(SYSTEMS)))
        ax.set_xticklabels([LABEL[s] for s in SYSTEMS], rotation=50, ha="right", fontsize=6.5)
        ax.set_title(QUERY[query], fontsize=8, color=INK)
        grid(ax, "y")
    axes[0].set_ylabel("API cost per query [USD]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[k], label=MODEL_LABEL[m]) for k, m in enumerate(API_MODELS)]
    legend_below(fig, handles, 6)
    save(fig, out, "fig_api_cost")


def multi_model(choices_path: Path, out: Path) -> None:
    rows = json.loads(choices_path.read_text())
    setups = []
    for row in rows:
        label = row["operators"].replace("qwen2.5:", "Qwen ").replace("-instruct", "").replace("gemma3:", "Gemma ") \
            .replace("llama3.2:", "Llama ").replace("llama3.1:", "Llama ").replace("b ", "B ")
        label = label[:-1] + "B" if label.endswith("b") else label
        if row["target"] != "0.8":
            label += f"\n(target {row['target']})"
        row["setup"] = label
        if label not in setups:
            setups.append(label)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=False)
    for ax, query in zip(axes, ["Q1", "Q2", "Q4"]):
        for k, system in enumerate(["stretto", "stretto_abacus"]):
            for i, setup in enumerate(setups):
                match = [r for r in rows if r["setup"] == setup and r["system"] == system and r["query"] == query]
                if not match:
                    continue
                r = match[0]
                energy = (r["gpu_wh"] or 0) + (r["cpu_wh"] or 0)
                x = i - 0.19 + 0.38 * k
                ax.bar(x, energy, 0.34, color=PALETTE[k], edgecolor="white", linewidth=0.3,
                       hatch=None if r["status"] == "success" else "////")
                share = r["small_share"] or 0
                ax.text(x, energy, f"{share:.0%}", ha="center", va="bottom", fontsize=4.6, color=MUTED,
                        rotation=90)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)  # room for the share labels above the tallest bars
        ax.set_xticks(range(len(setups)))
        ax.set_xticklabels(setups, rotation=50, ha="right", fontsize=6)
        ax.set_title(QUERY[query], fontsize=8, color=INK)
        grid(ax, "y")
    axes[0].set_ylabel("energy per query [Wh]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[0], label="Stretto optimizer"),
               plt.Rectangle((0, 0), 1, 1, color=PALETTE[1], label="Abacus optimizer"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=MUTED, hatch="////", label="engine error")]
    legend_below(fig, handles, 3)
    fig.text(0.5, 1.0, "label above bar = share of LLM calls served by the smaller operator model",
             ha="center", fontsize=6.5, color=MUTED)
    save(fig, out, "fig_multi_model")


def architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.4)
    ax.axis("off")

    def box(x, y, w, h, title, body, colour):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                                    facecolor=colour + "1f", edgecolor=colour, linewidth=1.0))
        ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="top", fontsize=6.6, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.85, body, ha="center", va="top", fontsize=5.4, color=MUTED, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                     color=MUTED, linewidth=0.8, shrinkA=0, shrinkB=0))

    box(0.3, 10.6, 9.4, 1.6, "Dashboard", "start / stop runs · results · provenance", PALETTE[0])
    box(0.3, 8.3, 9.4, 1.6, "GreenQuery service", "SLURM jobs · live logs · result store", PALETTE[0])
    box(0.3, 5.6, 9.4, 2.0, "Unmodified engines (SemBench runners)",
        "Stretto · Abacus · LOTUS\nPalimpzest · ThalamusDB · CAESURA", PALETTE[1])
    box(0.3, 2.9, 4.4, 2.0, "Logging proxy", "tokens · latency · cost\nmodel routing", PALETTE[2])
    box(5.3, 2.9, 4.4, 2.0, "Energy trace", "GPU counter (NVML)\nCPU (RAPL) · idle", PALETTE[2])
    box(0.3, 0.2, 4.4, 2.0, "Models", "Ollama on 1×A100\nOpenAI · OpenRouter", PALETTE[3])
    box(5.3, 0.2, 4.4, 2.0, "Join per query", "energy · calls · cost\nquality · provenance", PALETTE[3])
    arrow(5, 10.6, 5, 9.9)
    arrow(5, 8.3, 5, 7.6)
    arrow(2.5, 5.6, 2.5, 4.9)
    arrow(2.5, 2.9, 2.5, 2.2)
    arrow(7.5, 2.9, 7.5, 2.2)
    arrow(4.7, 3.9, 5.3, 3.9)
    save(fig, out, "fig_architecture")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, default=Path("reports/sembench_measured.csv"))
    parser.add_argument("--choices", type=Path, default=Path("reports/multi_model_choices.json"))
    parser.add_argument("--out", type=Path, default=Path("paper/edbt2027-demo/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = load(args.table)
    quality_energy(frame, args.out)
    model_size(frame, args.out)
    api_cost(frame, args.out)
    multi_model(args.choices, args.out)
    architecture(args.out)
    print(f"figures -> {args.out}")


if __name__ == "__main__":
    main()
