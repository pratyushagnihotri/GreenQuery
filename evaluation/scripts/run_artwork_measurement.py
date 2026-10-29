"""Run the six labelled Artwork queries with matched local planning."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.config import MeasurementConfig
from evaluation.measurement.wrapper import MeasurementWrapper
from evaluation.scripts.artwork_quality import (
    EXPECTED_QUERIES,
    FILTERS,
    GROUND_TRUTH,
    SOURCE,
    century_quality,
    ground_truth,
    selection_quality,
)
from evaluation.scripts.prepare_artwork_dataset import LOCAL_CSV, validate_prepared_data
from reasondb.backends.text_qa import direct_qa_batch_size
from reasondb.backends.vision_model import vision_qa_batch_size
from reasondb.database.indentifier import RemoteColumn
from reasondb.interface.config import Config as ReasonDBConfig
from reasondb.interface.connect import RaccoonDB
from reasondb.operators.extract.text_qa_extract import TextQaExtract
from reasondb.operators.filter.image_qa_filter import ImageQaFilter
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
from reasondb.utils.cache import CACHE_DIR


SUPPORTED_MODELS = ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "meta-llama/Llama-3.1-8B-Instruct")


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


def _dataset_hash() -> str:
    digest = hashlib.sha256()
    for path in (SOURCE, GROUND_TRUTH, LOCAL_CSV):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--energy-measurement", choices=["none", "nvml"], default="nvml")
    parser.add_argument("--cold-cache", action="store_true")
    parser.add_argument("--matched-planner", action="store_true")
    args = parser.parse_args()
    if not args.matched_planner:
        raise ValueError("Artwork matched demo requires --matched-planner")
    if not os.environ.get("GREENQUERY_CO2_G_PER_KWH"):
        raise ValueError("GREENQUERY_CO2_G_PER_KWH must be set")
    text_gpu = os.environ.get("GREENQUERY_TEXT_GPU_INDEX")
    vision_gpu = os.environ.get("GREENQUERY_VISION_GPU_INDEX")
    if not text_gpu or not vision_gpu or text_gpu == vision_gpu:
        raise ValueError("Artwork requires distinct GREENQUERY_TEXT_GPU_INDEX and GREENQUERY_VISION_GPU_INDEX")
    validate_prepared_data()
    ground_truth()
    database_name = _database_name()
    if args.cold_cache:
        _clear_cache(database_name)

    config = MeasurementConfig(
        demo_name="artwork",
        device=args.device,
        model_name=args.model_name,
        planner_model_name=args.model_name,
        model_provider="local",
        model_type="dgx",
        backend_profile="dgx-local-text-vision",
        energy_measurement=args.energy_measurement,
        co2_g_per_kwh=float(os.environ["GREENQUERY_CO2_G_PER_KWH"]),
        hardware_provider="dgx",
        gpu_count=2,
        direct_qa_batch_size=direct_qa_batch_size(),
        reasondb_database_name=database_name,
        dataset_name="artwork_no_duplicated_local_images",
        dataset_source="ReasonDB Artwork benchmark and labelled ground truth with prepared Wikimedia images",
        dataset_rows=65,
        dataset_sha256=_dataset_hash(),
        cache_state="disabled" if args.cold_cache else "not_controlled",
        model_cache_state="disabled" if args.cold_cache else "not_controlled",
        local_cache_policy="disabled" if args.cold_cache else "default",
        planner_cache_state="disabled" if args.cold_cache else "not_controlled",
        online_cache_policy="disabled",
        query_set="artwork_a1_a6_labelled",
        operator_policy="semantic-multimodal",
        baseline_type="artwork_matched_demo",
    )
    config_dict = config.to_dict()
    config_dict.update({
        "local_text_model_id": args.model_name,
        "local_planner_model_id": args.model_name,
        "local_vision_model_id": os.environ.get("GREENQUERY_LOCAL_VISION_MODEL", "llava-hf/llama3-llava-next-8b-hf"),
        "vision_qa_batch_size": vision_qa_batch_size(),
        "run_namespace": database_name,
        "nvml_gpu_roles": {"text": text_gpu, "vision": vision_gpu},
    })

    with MeasurementWrapper("artwork", args.output_dir, config_dict, split_artifacts=True) as wrapper:
        reason_config = ReasonDBConfig(
            database_name,
            device=config.device,
            operator_policy=config.operator_policy,
            backend_profile=config.backend_profile,
            local_cache_policy=config.local_cache_policy,
            local_text_model_id=args.model_name,
            local_planner_model_id=args.model_name,
        )
        with RaccoonDB(database_name, config=reason_config, device=config.device) as rc:
            artworks = rc.add_table(
                LOCAL_CSV,
                "artworks",
                image_columns=[RemoteColumn("artworks.image_url", "artworks.image")],
            )
            toolbox = rc.executor.configurator.physical_operators
            original_filters = toolbox.filter_operators
            original_extracts = toolbox.extract_operators
            toolbox.filter_operators = [op for op in original_filters if isinstance(op, ImageQaFilter)]
            try:
                for name, label, expression in FILTERS:
                    with wrapper.measure_query(name, "semantic_image_filter_project", baseline_type="artwork_matched_demo") as tracker:
                        result = artworks.filter(expression).project("Keep {title}").execute(
                            name, RecallGuarantee(0.6), PrecisionGuarantee(0.6)
                        )
                        tracker.set_result(result)
                        tracker.set_result_summary_fields(
                            quality=selection_quality(result.to_df().to_dict("records"), label),
                            expected_query_ids=list(EXPECTED_QUERIES),
                        )

                toolbox.extract_operators = [op for op in original_extracts if isinstance(op, TextQaExtract)]
                with wrapper.measure_query(
                    EXPECTED_QUERIES[5],
                    "semantic_image_filter_text_extract_project",
                    baseline_type="artwork_matched_demo",
                ) as tracker:
                    result = (
                        artworks.filter("{image} depicts Madonna and Child")
                        .extract("Extract the [century] from {inception_text}")
                        .project("Keep {title} and {century}")
                        .execute(EXPECTED_QUERIES[5], RecallGuarantee(0.6), PrecisionGuarantee(0.6))
                    )
                    tracker.set_result(result)
                    tracker.set_result_summary_fields(
                        quality=century_quality(result.to_df().to_dict("records")),
                        expected_query_ids=list(EXPECTED_QUERIES),
                    )
            finally:
                toolbox.filter_operators = original_filters
                toolbox.extract_operators = original_extracts


if __name__ == "__main__":
    main()
