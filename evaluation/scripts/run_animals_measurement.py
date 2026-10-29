"""Run labelled Animals Q1/Q2/Q3/Q10 with matched local planning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.config import MeasurementConfig
from evaluation.measurement.wrapper import MeasurementWrapper
from reasondb.backends.audio_model import audio_qa_batch_size
from reasondb.backends.text_qa import direct_qa_batch_size
from reasondb.backends.vision_model import vision_qa_batch_size
from reasondb.database.indentifier import RemoteColumn
from reasondb.interface.config import Config as ReasonDBConfig
from reasondb.interface.connect import RaccoonDB
from reasondb.operators.filter.audio_qa_filter import AudioQaFilter
from reasondb.operators.filter.image_qa_filter import ImageQaFilter
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
from reasondb.utils.cache import CACHE_DIR


ROOT = Path(__file__).parents[2]
FILES = ROOT / "reasondb/evaluation/benchmarks/files"
GROUND_TRUTH = ROOT / "reasondb/evaluation/ground_truth/animals"
EXPECTED_QUERIES = ("Q1_zebra_image_count", "Q2_lion_audio_count", "Q3_top_zebra_city", "Q10_top_zebra_city_station")


def _database_name() -> str:
    name = os.environ.get("GREENQUERY_RUN_NAMESPACE")
    if not name or not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("GREENQUERY_RUN_NAMESPACE must use only letters, numbers, and underscores")
    return name


def _clear_cache(database_name: str) -> None:
    path = (CACHE_DIR.resolve() / database_name).resolve()
    if path.parent != CACHE_DIR.resolve():
        raise RuntimeError(f"Unexpected cache path: {path}")
    if path.exists():
        shutil.rmtree(path)


def _require_data() -> None:
    required = [FILES / "animals_500.csv", FILES / "animals_audio.csv", *GROUND_TRUTH.glob("Q*_500.csv")]
    missing = [path for path in required if not path.exists()]
    if missing or not (ROOT / "SemBench/files/animals").exists():
        raise FileNotFoundError(
            "Animals data is not prepared. Run `python evaluation/scripts/prepare_animals_dataset.py` "
            "outside a measurement job. Missing: " + ", ".join(map(str, missing or [ROOT / "SemBench/files/animals"]))
        )


def _number(frame) -> float:
    for value in frame.iloc[0].tolist():
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    raise ValueError("query result contains no numeric aggregate")


def _aggregate_quality(frame, ground_truth: Path) -> dict:
    with ground_truth.open(newline="") as source:
        expected = float(next(csv.DictReader(source))["count_star()"])
    observed = _number(frame)
    return {
        "status": "evaluated", "metric": "exact_aggregate_count",
        "expected_count": expected, "observed_count": observed, "correct": observed == expected,
        "ground_truth": str(ground_truth.relative_to(ROOT)),
        "ground_truth_sha256": hashlib.sha256(ground_truth.read_bytes()).hexdigest(),
    }


def _top_group_quality(frame, ground_truth: Path, columns: tuple[str, ...]) -> dict:
    with ground_truth.open(newline="") as source:
        expected = {tuple(str(row[column]) for column in columns) for row in csv.DictReader(source)}
    count_column = next(column for column in frame.columns if column not in columns)
    top = frame[frame[count_column] == frame[count_column].max()]
    actual = {tuple(str(row[column]) for column in columns) for _, row in top.iterrows()}
    correct = len(actual & expected)
    precision = correct / len(actual) if actual else 0.0
    recall = correct / len(expected) if expected else 0.0
    return {
        "status": "evaluated", "metric": "top_group_set_precision_recall",
        "group_columns": list(columns), "expected_rows": len(expected), "returned_rows": len(actual),
        "correct_rows": correct, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "ground_truth": str(ground_truth.relative_to(ROOT)),
        "ground_truth_sha256": hashlib.sha256(ground_truth.read_bytes()).hexdigest(),
    }


def _dataset_hash() -> str:
    digest = hashlib.sha256()
    for path in (FILES / "animals_500.csv", FILES / "animals_audio.csv"):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--energy-measurement", choices=["none", "nvml"], default="nvml")
    parser.add_argument("--cold-cache", action="store_true")
    parser.add_argument("--matched-planner", action="store_true")
    parser.add_argument("--only-query", action="append", choices=EXPECTED_QUERIES)
    args = parser.parse_args()
    if not args.matched_planner:
        raise ValueError("Animals matched demo requires --matched-planner")
    if not os.environ.get("GREENQUERY_CO2_G_PER_KWH"):
        raise ValueError("GREENQUERY_CO2_G_PER_KWH must be set")
    roles = {role: os.environ.get(f"GREENQUERY_{role.upper()}_GPU_INDEX") for role in ("text", "vision", "audio")}
    if not all(roles.values()) or len(set(roles.values())) != 3:
        raise ValueError("Animals requires distinct text, vision, and audio GPU indices")
    _require_data()
    database_name = _database_name()
    if args.cold_cache:
        _clear_cache(database_name)

    config = MeasurementConfig(
        demo_name="animals", device=args.device, model_name=args.model_name,
        planner_model_name=args.model_name, model_provider="local", model_type="dgx",
        backend_profile="dgx-local-text-vision-audio", energy_measurement=args.energy_measurement,
        co2_g_per_kwh=float(os.environ["GREENQUERY_CO2_G_PER_KWH"]), hardware_provider="dgx", gpu_count=3,
        direct_qa_batch_size=direct_qa_batch_size(), reasondb_database_name=database_name,
        dataset_name="animals_500_images_166_audio_shipped", dataset_source="ReasonDB Animals CSVs plus staged SemBench media",
        dataset_rows=666, dataset_sha256=_dataset_hash(), cache_state="disabled" if args.cold_cache else "not_controlled",
        model_cache_state="disabled" if args.cold_cache else "not_controlled",
        local_cache_policy="disabled" if args.cold_cache else "default",
        planner_cache_state="disabled" if args.cold_cache else "not_controlled", online_cache_policy="disabled",
        query_set="animals_py_q1_q2_q3_q10", operator_policy="semantic-multimodal", baseline_type="animals_py_demo",
    )
    config_dict = config.to_dict()
    config_dict.update({
        "local_text_model_id": args.model_name, "local_planner_model_id": args.model_name,
        "local_vision_model_id": os.environ.get("GREENQUERY_LOCAL_VISION_MODEL", "llava-hf/llama3-llava-next-8b-hf"),
        "local_audio_model_id": os.environ.get("GREENQUERY_LOCAL_AUDIO_MODEL", "Qwen/Qwen2-Audio-7B-Instruct"),
        "vision_qa_batch_size": vision_qa_batch_size(), "audio_qa_batch_size": audio_qa_batch_size(),
        "run_namespace": database_name, "nvml_gpu_roles": roles,
        "q2_ground_truth_note": "Shipped Q2_500.csv is authoritative; its count differs from filename-label inspection.",
    })

    with MeasurementWrapper("animals", args.output_dir, config_dict, split_artifacts=True) as wrapper:
        reason_config = ReasonDBConfig(
            database_name, device=config.device, operator_policy=config.operator_policy,
            backend_profile=config.backend_profile, local_cache_policy=config.local_cache_policy,
            local_text_model_id=args.model_name, local_planner_model_id=args.model_name,
        )
        with RaccoonDB(database_name, config=reason_config, device=config.device) as rc:
            images = rc.add_table(FILES / "animals_500.csv", "animals", image_columns=[RemoteColumn("animals.imagepath", "animals.image")])
            audio = rc.add_table(FILES / "animals_audio.csv", "animals_audio", audio_columns=[RemoteColumn("animals_audio.audiopath", "animals_audio.audio")])
            queries = (
                ("Q1_zebra_image_count", "vision", lambda: images.filter("{image} depicts a zebra").aggregate("Count the rows as [count]"), GROUND_TRUTH / "Q1_500.csv", None),
                ("Q2_lion_audio_count", "audio", lambda: audio.filter("{audio} contains the sound of a lion").aggregate("Count the rows as [count]"), GROUND_TRUTH / "Q2_500.csv", None),
                ("Q3_top_zebra_city", "vision", lambda: images.filter("{image} depicts a zebra").groupby("Group by {city}").aggregate("Count rows as [count] by city"), GROUND_TRUTH / "Q3_500.csv", ("city",)),
                ("Q10_top_zebra_city_station", "vision", lambda: images.filter("{image} depicts a zebra").groupby("Group by {city}, {stationid}").aggregate("Count rows as [count] by city and stationid"), GROUND_TRUTH / "Q10_500.csv", ("city", "stationid")),
            )
            if args.only_query:
                selected = set(args.only_query)
                queries = tuple(query for query in queries if query[0] in selected)
            active_query_ids = [query[0] for query in queries]
            for name, role, build, truth, group_columns in queries:
                toolbox = rc.executor.configurator.physical_operators
                original_filters = toolbox.filter_operators
                selected = AudioQaFilter if role == "audio" else ImageQaFilter
                toolbox.filter_operators = [item for item in original_filters if isinstance(item, selected)]
                with wrapper.measure_query(
                    name,
                    f"semantic_{role}_filter_aggregate",
                    baseline_type="animals_py_demo",
                    planner_required=name != "Q2_lion_audio_count",
                ) as tracker:
                    try:
                        result = build().execute(name, RecallGuarantee(0.6), PrecisionGuarantee(0.6))
                        frame = result.to_df()
                        tracker.set_result(result)
                        tracker.set_result_summary_fields(
                            quality=_top_group_quality(frame, truth, group_columns) if group_columns else _aggregate_quality(frame, truth),
                            expected_query_ids=active_query_ids,
                        )
                    finally:
                        toolbox.filter_operators = original_filters


if __name__ == "__main__":
    main()
