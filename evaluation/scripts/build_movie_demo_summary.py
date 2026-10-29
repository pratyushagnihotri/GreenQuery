"""Build a small graph-ready summary for a Movie demo measurement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.artifacts import load_measurement

OPENAI_TOKEN_PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-2025-04-14": (2.0, 8.0),
}


def _labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as source:
        return {row["reviewid"]: row for row in csv.DictReader(source)}


def _source_row_count(path: Path) -> int:
    with path.open(newline="") as source:
        return sum(1 for _ in csv.DictReader(source))


def _quality(
    query: dict[str, Any], labels: dict[str, dict[str, str]], q4_labels: dict[str, dict[str, str]]
) -> dict[str, Any]:
    name = query["query_name"]
    ids = query.get("result_summary", {}).get("result_ids", [])
    if name in {"Q1_positive_reviews", "Q2_positive_reviews_for_taken_3"}:
        expected = {
            reviewid
            for reviewid, row in labels.items()
            if row["scoresentiment"] == "POSITIVE"
            and (name == "Q1_positive_reviews" or row["id"] == "taken_3")
        }
        matches = sum(reviewid in expected for reviewid in ids)
        return {
            "status": "evaluated",
            "metric": "precision_at_result_limit",
            "returned_rows": len(ids),
            "correct_rows": matches,
            "precision": matches / len(ids) if ids else None,
            "ground_truth": "reviews_100.scoresentiment == POSITIVE",
        }
    if name == "Q3_reviews_same_sentiment":
        summary = query.get("result_summary", {})
        pairs = summary.get("result_identity_tuples")
        if not summary.get("result_identity_columns") or pairs is None or any(len(pair) != 2 for pair in pairs):
            return {"status": "not_evaluated", "reason": "review-id pairs not recorded"}
        expected = {
            (left, right)
            for left, left_row in labels.items()
            for right, right_row in labels.items()
            if left_row["scoresentiment"] == right_row["scoresentiment"]
        }
        predicted = {tuple(pair) for pair in pairs}
        matches = len(predicted & expected)
        precision = matches / len(predicted) if predicted else 0.0
        recall = matches / len(expected) if expected else 0.0
        return {
            "status": "evaluated",
            "metric": "pair_precision_recall",
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "ground_truth": "pairs with equal reviews_100.scoresentiment",
        }
    if name == "Q4_positive_reviews_for_taken_3_count":
        values = query.get("result_summary", {}).get("result_values") or {}
        observed = next((value for value in values.values() if isinstance(value, (int, float))), None)
        if observed is None:
            return {"status": "not_evaluated", "reason": "aggregate count not recorded"}
        expected = sum(
            row["id"] == "taken_3" and row["scoresentiment"] == "POSITIVE"
            for row in labels.values()
        )
        return {
            "status": "evaluated",
            "metric": "exact_aggregate_count",
            "expected_count": expected,
            "observed_count": observed,
            "correct": observed == expected,
            "ground_truth": "reviews_100.scoresentiment == POSITIVE and id == taken_3",
        }
    if name == "Q4_positive_reviews_for_taken_3_ratio":
        total = query.get("result_summary", {}).get("evaluation_input_rows", 0)
        observed = len(ids) / total if total else None
        taken_3 = [row for row in q4_labels.values() if row["id"] == "taken_3"]
        expected = (
            sum(row["scoresentiment"] == "POSITIVE" for row in taken_3) / len(taken_3)
            if taken_3
            else None
        )
        return {
            "status": "evaluated" if observed is not None and expected is not None else "not_evaluated",
            "metric": "ratio_absolute_error",
            "expected_ratio": expected,
            "observed_ratio": observed,
            "absolute_error": abs(observed - expected) if observed is not None and expected is not None else None,
            "ground_truth": "reviews_1000.scoresentiment == POSITIVE for Taken 3",
        }
    return {"status": "not_evaluated", "reason": "result-specific scorer not yet recorded"}


def _online_cost(query: dict[str, Any]) -> tuple[float | None, str]:
    api = query.get("api_calls") or {}
    recorded = query.get("api_cost_usd")
    if recorded not in (None, 0) or not api.get("total_calls"):
        return recorded, "recorded"
    if query.get("model_provider") != "openai":
        return None, "not_available"
    prices = OPENAI_TOKEN_PRICES.get(query.get("model_name", ""))
    if prices is None:
        return None, "not_available"
    input_price, output_price = prices
    return (
        api.get("total_prompt_tokens", 0) * input_price / 1_000_000
        + api.get("total_completion_tokens", 0) * output_price / 1_000_000,
        "estimated_from_recorded_tokens_and_model_pricing",
    )


def _record(
    query: dict[str, Any], labels: dict[str, dict[str, str]], q4_labels: dict[str, dict[str, str]]
) -> dict[str, Any]:
    api = query.get("api_calls") or {}
    local = query.get("local_inference") or {}
    energy = query.get("energy") or {}
    energy_source = energy.get("metric_source") or (query.get("metric_sources") or {}).get("energy")
    co2_source = (query.get("metric_sources") or {}).get("co2", "not_available")
    api_cost_usd, api_cost_source = _online_cost(query)
    result_summary = query.get("result_summary") or {}
    phases = result_summary.get("api_phase_metrics") or {}
    events = query.get("api_events") or []
    return {
        "query": query.get("query_name"),
        "model": query.get("model_name"),
        "operators": query.get("physical_operators", []),
        "execution_plan": query.get("execution_plan", {}),
        "runtime_seconds": query.get("execution_time_seconds"),
        "energy": {
            "energy_kwh": query.get("energy_kwh") if energy_source != "not_available" else None,
            "energy_kwh_min": result_summary.get("api_energy_kwh_min"),
            "energy_kwh_max": result_summary.get("api_energy_kwh_max"),
            "co2_g": query.get("co2_g") if co2_source != "not_available" else None,
            "co2_g_min": result_summary.get("api_co2_g_min"),
            "co2_g_max": result_summary.get("api_co2_g_max"),
            "energy_source": energy_source,
            "co2_source": co2_source,
        },
        "online": {
            "api_calls": api.get("total_calls", 0),
            "cached_calls": api.get("cached_calls", 0),
            "input_tokens": api.get("total_prompt_tokens", 0),
            "output_tokens": api.get("total_completion_tokens", 0),
            "api_cost_usd": api_cost_usd,
            "api_cost_source": api_cost_source,
            "requested_models": sorted({event.get("requested_model_name") for event in events if event.get("requested_model_name")}),
            "resolved_models": sorted({event.get("resolved_model_name") for event in events if event.get("resolved_model_name")}),
            "planning": phases.get("planning", {}),
            "execution": phases.get("execution", {}),
        },
        "offline": {
            "local_inference_requests": local.get("request_count", 0),
            "input_tokens": local.get("execution_input_tokens", 0),
            "output_tokens": local.get("execution_output_tokens", 0),
        },
        "quality": _quality(query, labels, q4_labels),
        "status": query.get("status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", type=Path)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("reasondb/evaluation/benchmarks/files/reviews_100.csv"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--q4-ground-truth",
        type=Path,
        default=Path("reasondb/evaluation/benchmarks/files/reviews_1000.csv"),
    )
    args = parser.parse_args()
    measurement = load_measurement(args.measurement, hydrate=True)
    labels = _labels(args.ground_truth)
    q4_labels = _labels(args.q4_ground_truth)
    summary = {
        "summary_layout_version": "movie-demo-summary-v1",
        "measurement_id": measurement.get("measurement_id"),
        "cache_policy": measurement.get("config", {}).get("online_cache_policy"),
        "ground_truth": {
            "dataset": str(args.ground_truth),
            "dataset_sha256": hashlib.sha256(args.ground_truth.read_bytes()).hexdigest(),
            "label_column": "scoresentiment",
            "positive_label": "POSITIVE",
            "source_rows": _source_row_count(args.ground_truth),
            "unique_review_ids": len(labels),
        },
        "queries": [
            _record(query, labels, q4_labels)
            for query in measurement.get("queries", [])
        ],
    }
    output = args.output or args.measurement.with_name(f"{args.measurement.stem}_summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
