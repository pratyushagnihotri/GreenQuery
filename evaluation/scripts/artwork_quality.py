"""Stable Artwork query matrix and deterministic ground-truth scoring."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "reasondb/evaluation/benchmarks/files/paintings_sampled_no_duplicated.csv"
GROUND_TRUTH = ROOT / "reasondb/evaluation/ground_truth/artwork/artwork_no_duplicated.csv"
EXPECTED_QUERIES = (
    "A1_madonna_and_child",
    "A2_more_than_two_people",
    "A3_more_than_three_people",
    "A4_saints_with_halos",
    "A5_death_dominant_theme",
    "A6_madonna_and_child_century",
)
FILTERS = (
    (EXPECTED_QUERIES[0], "m&c", "{image} depicts Madonna and Child"),
    (EXPECTED_QUERIES[1], "more_than_2_people", "{image} depicts more than two people"),
    (EXPECTED_QUERIES[2], "more_than_3_people", "{image} depicts more than three people"),
    (EXPECTED_QUERIES[3], "saints_with_halos", "{image} depicts saints identifiable by their halos"),
    (EXPECTED_QUERIES[4], "death_theme", "{image} depicts a scene in which death is a dominant theme"),
)


def ground_truth() -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with SOURCE.open(newline="") as source, GROUND_TRUTH.open(newline="") as truth:
        source_rows = list(csv.DictReader(source))
        truth_rows = list(csv.DictReader(truth))
    if len(source_rows) != 65 or len(truth_rows) != 65:
        raise RuntimeError("Artwork source and ground truth must each contain exactly 65 rows")
    if [int(row["_index_artworks"]) for row in truth_rows] != list(range(65)):
        raise RuntimeError("Artwork ground-truth indices must be 0 through 64 in source order")
    titles = [row["title"] for row in source_rows]
    if len(set(titles)) != 65:
        raise RuntimeError("Artwork titles must be unique for deterministic scoring")
    for row in truth_rows:
        if any(row[label] not in {"0", "1"} for _, label, _ in FILTERS):
            raise RuntimeError("Artwork selection labels must be binary")
        int(row["century"])
    return source_rows, {title: truth_rows[index] for index, title in enumerate(titles)}


def selection_quality(rows: list[dict], label: str) -> dict:
    source_rows, truth = ground_truth()
    expected = {row["title"] for row in source_rows if truth[row["title"]][label] == "1"}
    actual = {str(row["title"]) for row in rows}
    correct = len(actual & expected)
    precision = correct / len(actual) if actual else 0.0
    recall = correct / len(expected) if expected else 0.0
    return {
        "status": "evaluated",
        "metric": "artwork_title_set_precision_recall",
        "label_column": label,
        "expected_rows": len(expected),
        "returned_rows": len(actual),
        "correct_rows": correct,
        "false_positive_rows": len(actual - expected),
        "false_negative_rows": len(expected - actual),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "ground_truth": str(GROUND_TRUTH.relative_to(ROOT)),
        "ground_truth_sha256": hashlib.sha256(GROUND_TRUTH.read_bytes()).hexdigest(),
    }


def _century(value) -> int | None:
    match = re.search(r"-?\d+", str(value))
    return int(match.group()) if match else None


def century_quality(rows: list[dict]) -> dict:
    selection = selection_quality(rows, "m&c")
    _, truth = ground_truth()
    unique_rows = {str(row["title"]): row for row in rows}
    selected = [row for title, row in unique_rows.items() if title in truth and truth[title]["m&c"] == "1"]
    checked = len(selected)
    correct = sum(
        _century(row.get("century")) == int(truth[str(row["title"])]["century"])
        for row in selected
    )
    return {
        "status": "evaluated",
        "metric": "artwork_selection_and_century",
        "selection": selection,
        "century_extraction": {
            "status": "evaluated",
            "metric": "exact_century_accuracy_on_correctly_selected_rows",
            "label_column": "century",
            "checked_rows": checked,
            "correct_rows": correct,
            "accuracy": correct / checked if checked else 0.0,
        },
        "ground_truth": selection["ground_truth"],
        "ground_truth_sha256": selection["ground_truth_sha256"],
    }
