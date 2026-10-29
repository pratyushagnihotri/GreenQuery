"""Run the shipped Medical Q1/Q3/Q4/Q9 queries with matched local planning."""

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
from evaluation.scripts.run_movie_measurement import load_env_file
from reasondb.backends.text_qa import direct_qa_batch_size
from reasondb.backends.vision_model import vision_qa_batch_size
from reasondb.database.indentifier import InPlaceColumn, RemoteColumn
from reasondb.interface.config import Config as ReasonDBConfig
from reasondb.interface.connect import RaccoonDB
from reasondb.operators.filter.image_qa_filter import ImageQaFilter
from reasondb.operators.filter.traditional_filter import TraditionalFilter
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
from reasondb.reasoning.providers import model_requires_ecologits, provider_for, require_model_credentials
from reasondb.utils.cache import CACHE_DIR


ROOT = Path(__file__).parents[2]
FILES = ROOT / "reasondb/evaluation/benchmarks/files"
GROUND_TRUTH = ROOT / "reasondb/evaluation/ground_truth/medical"
EXPECTED_QUERIES = ("Q1_allergy_patients", "Q3_cancer_family_lung_problems_patients", "Q4_average_age_acne_patients", "Q9_sick_patients_skin_xray")
API_BACKEND_PROFILE = "api"


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
    missing = [path for path in (
        FILES / "text_symptoms_data_1000.csv",
        FILES / "image_skin_data_1000.csv",
        FILES / "image_x_ray_data_1000.csv",
        FILES / "patient_data_with_labels_1000.csv",
        ROOT / "SemBench/files/medical/data/raw_data",
    ) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Medical data is not prepared. Run `python evaluation/scripts/prepare_medical_dataset.py` "
            "outside a measurement job. Missing: " + ", ".join(map(str, missing))
        )


def _ids(path: Path) -> set[str]:
    with path.open(newline="") as source:
        return {str(row["patient_id"]) for row in csv.DictReader(source)}


def _set_quality(frame, ground_truth: Path) -> dict:
    expected = _ids(ground_truth)
    actual = {str(value) for value in frame["patient_id"].tolist()}
    correct = len(actual & expected)
    precision = correct / len(actual) if actual else 0.0
    recall = correct / len(expected) if expected else 0.0
    return {
        "status": "evaluated",
        "metric": "patient_id_set_precision_recall",
        "expected_rows": len(expected),
        "returned_rows": len(actual),
        "correct_rows": correct,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "ground_truth": str(ground_truth.relative_to(ROOT)),
        "ground_truth_sha256": hashlib.sha256(ground_truth.read_bytes()).hexdigest(),
    }


def _aggregate_quality(frame) -> dict:
    ground_truth = GROUND_TRUTH / "Q4_1000.csv"
    with ground_truth.open(newline="") as source:
        expected = float(next(csv.DictReader(source))["average_acne_age"])
    observed = next(
        float(value)
        for value in frame.iloc[0].tolist()
        if _as_float(value) is not None
    )
    return {
        "status": "evaluated",
        "metric": "exact_aggregate_average_age",
        "expected_average_age": expected,
        "observed_average_age": observed,
        "absolute_error": abs(observed - expected),
        "correct": observed == expected,
        "ground_truth": str(ground_truth.relative_to(ROOT)),
        "ground_truth_sha256": hashlib.sha256(ground_truth.read_bytes()).hexdigest(),
    }


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dataset_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(FILES.glob("*data_1000.csv")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend-profile", choices=[API_BACKEND_PROFILE, "dgx-local-text-vision"], default="dgx-local-text-vision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--energy-measurement", choices=["none", "nvml"], default="nvml")
    parser.add_argument("--cold-cache", action="store_true")
    parser.add_argument("--matched-planner", action="store_true")
    parser.add_argument("--online-cache-policy", choices=["default", "disabled"], default="default")
    parser.add_argument("--split-artifacts", action="store_true")
    args = parser.parse_args()
    if args.cold_cache:
        args.online_cache_policy = "disabled"
    is_api = args.backend_profile == API_BACKEND_PROFILE
    if not is_api and not args.matched_planner:
        raise ValueError("Medical matched demo requires --matched-planner")
    if is_api:
        load_env_file()
        require_model_credentials(args.model_name)
        os.environ["GREENQUERY_OPENAI_MODEL"] = args.model_name
        args.device = "cpu"
        args.energy_measurement = "none"
    if not is_api and not os.environ.get("GREENQUERY_CO2_G_PER_KWH"):
        raise ValueError("GREENQUERY_CO2_G_PER_KWH must be set")
    text_gpu = os.environ.get("GREENQUERY_TEXT_GPU_INDEX")
    vision_gpu = os.environ.get("GREENQUERY_VISION_GPU_INDEX")
    if not is_api and (not text_gpu or not vision_gpu or text_gpu == vision_gpu):
        raise ValueError("Medical requires distinct GREENQUERY_TEXT_GPU_INDEX and GREENQUERY_VISION_GPU_INDEX")
    _require_data()
    database_name = _database_name()
    if args.cold_cache:
        _clear_cache(database_name)

    config = MeasurementConfig(
        demo_name="medical",
        device=args.device,
        model_name=args.model_name,
        planner_model_name=args.model_name,
        model_provider=provider_for(args.model_name) if is_api else "local",
        model_type="api" if is_api else "dgx",
        backend_profile=args.backend_profile,
        energy_measurement=args.energy_measurement,
        co2_g_per_kwh=None if is_api else float(os.environ["GREENQUERY_CO2_G_PER_KWH"]),
        hardware_provider=None if is_api else "dgx",
        gpu_count=None if is_api else 2,
        direct_qa_batch_size=None if is_api else direct_qa_batch_size(),
        reasondb_database_name=database_name,
        dataset_name="medical_1000_shipped",
        dataset_source="ReasonDB Medical benchmark files plus the shipped SemBench image archive",
        dataset_rows=1000,
        dataset_sha256=_dataset_hash(),
        cache_state="disabled" if args.cold_cache else "not_controlled",
        model_cache_state="disabled" if args.cold_cache else "not_controlled",
        local_cache_policy="disabled" if args.cold_cache else "default",
        planner_cache_state="disabled" if args.cold_cache else "not_controlled",
        online_cache_policy=args.online_cache_policy,
        query_set="medical_py_q1_q3_q4_q9",
        operator_policy="semantic-multimodal",
        baseline_type="medical_py_demo",
    )
    config_dict = config.to_dict()
    config_dict["run_namespace"] = database_name
    if is_api:
        config_dict.update(
            require_ecologits=model_requires_ecologits(args.model_name),
            online_measurement_boundary="provider_estimated_from_ecologits"
            if model_requires_ecologits(args.model_name)
            else "provider_usage_only_no_energy_estimate",
        )
    else:
        config_dict.update({
            "local_text_model_id": args.model_name,
            "local_planner_model_id": args.model_name,
            "local_vision_model_id": os.environ.get("GREENQUERY_LOCAL_VISION_MODEL", "llava-hf/llama3-llava-next-8b-hf"),
            "vision_qa_batch_size": vision_qa_batch_size(),
            "nvml_gpu_roles": {"text": text_gpu, "vision": vision_gpu},
        })

    with MeasurementWrapper("medical", args.output_dir, config_dict, split_artifacts=args.split_artifacts or not is_api) as wrapper:
        reason_config = ReasonDBConfig(
            database_name,
            device=config.device,
            operator_policy=config.operator_policy,
            backend_profile=config.backend_profile,
            local_cache_policy=config.local_cache_policy,
            openai_model_id=args.model_name if is_api else None,
            openai_cache_enabled=config.online_cache_policy != "disabled",
            local_text_model_id=None if is_api else args.model_name,
            local_planner_model_id=None if is_api else args.model_name,
        )
        with RaccoonDB(database_name, config=reason_config, device=config.device) as rc:
            symptoms = rc.add_table(FILES / "text_symptoms_data_1000.csv", "patients_symptoms", text_columns=[InPlaceColumn("patients_symptoms.symptoms")])
            skin = rc.add_table(FILES / "image_skin_data_1000.csv", "skin", image_columns=[RemoteColumn("skin.image_path", "skin.skin_picture")])
            x_ray = rc.add_table(FILES / "image_x_ray_data_1000.csv", "x_ray", image_columns=[RemoteColumn("x_ray.image_path", "x_ray.xray_picture")])
            patient = rc.add_table(FILES / "patient_data_with_labels_1000.csv", "patient")

            queries = (
                ("Q1_allergy_patients", "semantic_text_filter_project", lambda: symptoms.filter("{symptoms} indicates allergies").project("Keep {patient_id}"), "allergy_patients", GROUND_TRUTH / "Q1_1000.csv"),
                ("Q3_cancer_family_lung_problems_patients", "traditional_filter_join_semantic_image_filter_project", lambda: patient.filter("{is_sick} is True").filter("{did_family_have_cancer} is 1").join(x_ray, "Join on {patient_id}").filter("{xray_picture} shows lung problems").project("Keep {patient_id}"), "cancer_family_lung_problems_patients", GROUND_TRUTH / "Q3_1000.csv"),
                ("Q4_average_age_acne_patients", "semantic_text_filter_join_project_aggregate", lambda: symptoms.filter("{symptoms} indicates acne").join(patient, "Join on {patient_id}").project("Keep {age}").aggregate("Average of {age}[average_age_acne_patients]"), "average_age_acne_patients", None),
                ("Q9_sick_patients_skin_xray", "semantic_image_filter_join_semantic_image_filter_project", lambda: skin.filter("{skin_picture} shows sick moles").join(x_ray, "Join on {patient_id}").filter("{xray_picture} shows lung problems").project("Keep {patient_id}"), "sick_patients_skin_xray", GROUND_TRUTH / "Q9_1000.csv"),
            )
            for name, operator, build, result_name, truth in queries:
                toolbox = rc.executor.configurator.physical_operators
                original_filters = toolbox.filter_operators
                if name in {"Q3_cancer_family_lung_problems_patients", "Q9_sick_patients_skin_xray"}:
                    toolbox.filter_operators = [
                        item for item in original_filters
                        if isinstance(item, (TraditionalFilter, ImageQaFilter))
                    ]
                with wrapper.measure_query(name, operator, baseline_type="medical_py_demo") as tracker:
                    try:
                        result = build().execute(result_name, RecallGuarantee(0.6), PrecisionGuarantee(0.6))
                        frame = result.to_df()
                        tracker.set_result(result)
                        tracker.set_result_summary_fields(
                            quality=_aggregate_quality(frame) if truth is None else _set_quality(frame, truth),
                            expected_query_ids=list(EXPECTED_QUERIES),
                        )
                    finally:
                        toolbox.filter_operators = original_filters


if __name__ == "__main__":
    main()
