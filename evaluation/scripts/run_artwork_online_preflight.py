"""Run the single online Artwork API preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.config import MeasurementConfig
from evaluation.measurement.wrapper import MeasurementWrapper
from evaluation.scripts.artwork_quality import EXPECTED_QUERIES, GROUND_TRUTH, SOURCE, century_quality, ground_truth
from reasondb.database.indentifier import RemoteColumn
from reasondb.interface.config import Config as ReasonDBConfig
from reasondb.interface.connect import RaccoonDB
from reasondb.operators.extract.text_qa_extract import TextQaExtract
from reasondb.operators.filter.image_qa_filter import ImageQaFilter
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
from reasondb.utils.cache import CACHE_DIR

QUERY_ID = EXPECTED_QUERIES[5]
ARTWORK_IMAGE_CACHE = CACHE_DIR / "artworks_sampled" / "artworks_files"


def load_env_file(path: Path | None = None) -> None:
    env_path = path or Path(__file__).parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def count_csv_rows(path: Path) -> int:
    with path.open() as source:
        return max(sum(1 for _ in source) - 1, 0)


def dataset_hash() -> str:
    digest = hashlib.sha256()
    for path in (SOURCE, GROUND_TRUTH):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_cached_local_csv() -> Path:
    target = Path(tempfile.gettempdir()) / f"greenquerylab_artwork_cached_{os.getpid()}.csv"
    with SOURCE.open(newline="") as source, target.open("w", newline="") as output:
        reader = csv.DictReader(source)
        fieldnames = [*(reader.fieldnames or []), "inception_text"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            image_path = ARTWORK_IMAGE_CACHE / urlparse(row["image_url"]).path.rsplit("/", 1)[-1]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing cached Artwork image: {image_path}")
            row["image_url"] = str(image_path)
            row["inception_text"] = f"Creation date: {row['inception']}"
            writer.writerow(row)
    return target


def database_name(run_namespace: str | None) -> str:
    name = run_namespace or f"online_artwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("--run-namespace must use only letters, numbers, and underscores")
    return name


def clear_database_cache(name: str) -> Path:
    cache_root = CACHE_DIR.resolve()
    path = (cache_root / name).resolve()
    if path.parent != cache_root:
        raise RuntimeError(f"Refusing to clear unexpected cache path: {path}")
    if path.exists():
        shutil.rmtree(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gpt-4o-mini")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-namespace")
    args = parser.parse_args()

    load_env_file()
    if args.model_name != "gpt-4o-mini":
        raise ValueError("This preflight is scoped to exactly one gpt-4o-mini run")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    ground_truth()
    source_csv = make_cached_local_csv()
    name = database_name(args.run_namespace)
    cleared = clear_database_cache(name)
    os.environ["GREENQUERY_OPENAI_MODEL"] = args.model_name

    config = MeasurementConfig(
        demo_name="artwork",
        device=args.device,
        precision_guarantee=0.6,
        recall_guarantee=0.6,
        output_dir=args.output_dir,
        model_name=args.model_name,
        planner_model_name=args.model_name,
        model_provider="openai",
        model_type="api",
        backend_profile="api",
        energy_measurement="none",
        reasondb_database_name=name,
        dataset_name="artwork_no_duplicated_cached_images",
        dataset_source="ReasonDB Artwork 65-painting benchmark and labelled ground truth with cached Wikimedia images",
        dataset_rows=count_csv_rows(source_csv),
        dataset_sha256=dataset_hash(),
        cache_state="disabled",
        model_cache_state="disabled",
        local_cache_policy="disabled",
        planner_cache_state="cache_disabled",
        online_cache_policy="disabled",
        query_set="artwork_a6_online_preflight",
        operator_policy="semantic-multimodal",
        baseline_type="artwork_online_api_preflight",
    )
    config_dict = config.to_dict()
    config_dict.update({
        "require_ecologits": True,
        "online_measurement_boundary": "provider_estimated_from_ecologits",
        "run_namespace": name,
        "expected_query_id": QUERY_ID,
        "cleared_reasondb_cache": str(cleared),
    })

    with MeasurementWrapper("artwork", args.output_dir, config_dict, split_artifacts=True) as wrapper:
        reason_config = ReasonDBConfig(
            name,
            device=config.device,
            operator_policy=config.operator_policy,
            backend_profile=config.backend_profile,
            local_cache_policy=config.local_cache_policy,
            openai_model_id=config.model_name,
            openai_cache_enabled=False,
        )
        with RaccoonDB(name, config=reason_config, device=config.device) as rc:
            artworks = rc.add_table(
                source_csv,
                "artworks",
                image_columns=[RemoteColumn("artworks.image_url", "artworks.image")],
            )
            toolbox = rc.executor.configurator.physical_operators
            original_filters = toolbox.filter_operators
            original_extracts = toolbox.extract_operators
            toolbox.filter_operators = [op for op in original_filters if isinstance(op, ImageQaFilter)]
            toolbox.extract_operators = [op for op in original_extracts if isinstance(op, TextQaExtract)]
            try:
                with wrapper.measure_query(
                    QUERY_ID,
                    "semantic_image_filter_text_extract_project",
                    baseline_type=config.baseline_type,
                ) as tracker:
                    result = (
                        artworks.filter("{image} depicts Madonna and Child")
                        .extract("Extract the [century] from {inception_text}")
                        .project("Keep {title} and {century}")
                        .execute(QUERY_ID, RecallGuarantee(config.recall_guarantee), PrecisionGuarantee(config.precision_guarantee))
                    )
                    tracker.set_result(result)
                    tracker.set_result_summary_fields(
                        quality=century_quality(result.to_df().to_dict("records")),
                        expected_query_ids=[QUERY_ID],
                    )
            finally:
                toolbox.filter_operators = original_filters
                toolbox.extract_operators = original_extracts


if __name__ == "__main__":
    main()
