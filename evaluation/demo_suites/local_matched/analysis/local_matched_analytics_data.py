"""Validated compact inputs for the unified matched-planner presentation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "data"
MODEL_LABELS = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen 2.5 14B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
}
DATASETS = {
    "movie": {
        "label": "Movie",
        "source": DATA_ROOT / "movie_query_metrics.json",
        "demo": "movie",
        "expected_queries": 4,
        "context": "Movie-review queries, including filters, a relational join, and a ratio estimate.",
    },
    "rotowire": {
        "label": "RotoWire",
        "source": DATA_ROOT / "rotowire_query_metrics.json",
        "demo": "rotowire",
        "expected_queries": 2,
        "context": "Basketball-report queries that join records and repeatedly extract facts from text.",
    },
    "medical": {
        "label": "Medical",
        "source": DATA_ROOT / "medical_query_metrics.json",
        "demo": "medical",
        "expected_queries": 4,
        "context": "Patient-record queries, including text and image-QA workloads.",
    },
}
QUERY_GUIDES = {
    "movie": {
        "Q1": "Find positive reviews",
        "Q2": "Find positive reviews for Taken 3",
        "Q3": "Find review pairs with the same sentiment",
        "Q4": "Estimate Taken 3's positive-review ratio",
    },
    "rotowire": {
        "R1": "Extract each player's points and assists",
        "R2": "Extract player points where the home team won",
    },
    "medical": {
        "Q1": "Find patients with allergies",
        "Q3": "Find cancer patients with family and lung conditions",
        "Q4": "Average the age of patients with acne",
        "Q9": "Find sick patients with skin X-ray records",
    },
}


def load_dataset(dataset: str) -> pd.DataFrame:
    """Return one complete, cold-cache matched-planner dataset."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    spec = DATASETS[dataset]
    rows = json.loads(spec["source"].read_text())
    frame = pd.DataFrame(row for row in rows if row["demo"] == spec["demo"]).copy()
    models = set(frame["model"])
    expected = len(models) * 10 * spec["expected_queries"]
    if len(frame) != expected or frame[["model", "run", "query"]].duplicated().any():
        raise ValueError(f"{spec['label']} must have {expected} unique model/run/query measurements.")
    if len(models) < 2 or not models <= set(MODEL_LABELS) or (frame["cache_hits"] != 0).any():
        raise ValueError(f"{spec['label']} is not a complete supported-model cold-cache comparison.")
    if (frame["planner_model"] != frame["model"]).any():
        raise ValueError(f"{spec['label']} does not use the same planner and execution model.")
    if not (frame["energy_source"] == "measured_from_nvml_query_window").all():
        raise ValueError(f"{spec['label']} has non-NVML energy measurements.")
    frame["model_label"] = frame["model"].map(MODEL_LABELS)
    frame["query_id"] = frame["query"].str.split("_", n=1).str[0]
    frame["query_description"] = frame["query"].str.split("_", n=1).str[1].str.replace("_", " ")
    frame["energy_mwh"] = frame["energy_kwh"] * 1_000_000
    frame["run_number"] = frame["run"].str[-2:].astype(int)
    return frame
