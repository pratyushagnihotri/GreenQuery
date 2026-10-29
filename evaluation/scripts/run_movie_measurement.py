"""Run Movie demo with measurement."""

import argparse
import csv as csv_module
import os
import re
import shutil
import sys
import tempfile
import hashlib
from pathlib import Path

# Add greenQueryLab to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.wrapper import MeasurementWrapper
from evaluation.measurement.config import MeasurementConfig
from reasondb.interface.connect import RaccoonDB
from reasondb.interface.config import Config as ReasonDBConfig
from reasondb.database.indentifier import InPlaceColumn
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
from reasondb.reasoning.providers import model_requires_ecologits, provider_for, require_model_credentials
from reasondb.utils.cache import CACHE_DIR

SEMANTIC_OPERATORS = {"TextQaFilter", "TextQaExtract", "ExtractAndQaFilter"}
SEMANTIC_SAMPLE_ROWS = 12


def load_env_file(path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries from .env without printing secrets."""
    env_path = path or Path(__file__).parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def count_csv_rows(csv_path: Path) -> int:
    """Count data rows in a CSV file without loading the full file."""
    with open(csv_path) as f:
        return max(sum(1 for _ in f) - 1, 0)


def make_semantic_sample_csv(source_csv: Path, row_limit: int = SEMANTIC_SAMPLE_ROWS) -> Path:
    """Create a small runtime CSV for low-cost semantic API measurement."""
    target = Path(tempfile.gettempdir()) / f"greenquerylab_movie_reviews_semantic_{row_limit}.csv"
    with open(source_csv, newline="") as src, open(target, "w", newline="") as dst:
        reader = csv_module.reader(src)
        writer = csv_module.writer(dst)
        header = next(reader)
        writer.writerow(header)
        for row_index, row in enumerate(reader):
            if row_index >= row_limit:
                break
            writer.writerow(row)
    return target


def make_movie_subset_csv(source_csv: Path, movie_id: str) -> tuple[Path, list[tuple[str, str]]]:
    """Create a deterministic movie subset for semantic review classification."""
    target = Path(tempfile.gettempdir()) / (
        f"greenquerylab_{source_csv.stem}_{movie_id}_{os.getuid()}.csv"
    )
    identities = []
    with open(source_csv, newline="") as src, open(target, "w", newline="") as dst:
        reader = csv_module.DictReader(src)
        writer = csv_module.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["id"] == movie_id:
                writer.writerow(row)
                identities.append((row["reviewtext"], row["reviewid"]))
    return target, identities


def movie_database_name(query_set: str, run_namespace: str | None = None) -> str:
    base_name = "movie_sampled_semantic" if query_set == "semantic" else "movie_sampled"
    return f"{base_name}_{run_namespace}" if run_namespace else base_name


def clear_movie_database_cache(query_set: str, run_namespace: str | None = None) -> Path:
    """Remove only this runner's persistent Movie metadata cache."""
    db_name = movie_database_name(query_set, run_namespace)
    cache_root = CACHE_DIR.resolve()
    cache_path = (cache_root / db_name).resolve()
    if cache_path.parent != cache_root:
        raise RuntimeError(f"Refusing to clear unexpected cache path: {cache_path}")
    if cache_path.exists():
        shutil.rmtree(cache_path)
    return cache_path


def extract_physical_operators(result) -> set[str]:
    """Extract physical operator names from a ReasonDB result."""
    pipelines = getattr(result, "tuned_pipelines", []) or []
    operators = set()
    for pipeline in pipelines:
        operators.update(re.findall(r'"operator":\s*"([^"]+)"', str(pipeline)))
    return operators


def require_semantic_operator(result, query_name: str):
    """Fail the run unless a semantic text operator appears in the physical plan."""
    operators = extract_physical_operators(result)
    semantic_hits = {
        operator
        for operator in operators
        if any(operator.startswith(prefix) for prefix in SEMANTIC_OPERATORS)
    }
    if not semantic_hits:
        operator_text = ", ".join(sorted(operators)) or "not_available"
        raise RuntimeError(
            f"{query_name} did not use a semantic text operator. "
            f"Observed operators: {operator_text}"
        )
    print(f"   Semantic operator evidence: {', '.join(sorted(semantic_hits))}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Movie demo with measurement.")
    parser.add_argument(
        "--query-set",
        choices=["baseline", "semantic", "all", "demo"],
        default="baseline",
        help="Movie query set to run. Default preserves Phase 2.5 Q1/Q2 behavior.",
    )
    parser.add_argument(
        "--only-q1",
        action="store_true",
        help="Run only the first query from demos/movie.py.",
    )
    parser.add_argument(
        "--operator-policy",
        choices=["default", "semantic-text-only"],
        default="default",
        help="Physical operator policy to use for the ReasonDB run.",
    )
    parser.add_argument(
        "--backend-profile",
        choices=["api", "dgx-local-text", "cloudlab-local-text"],
        default="api",
        help="Backend profile. Default keeps the existing OpenAI/API setup.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device string passed to ReasonDB, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--model-provider",
        default=None,
        help="Provider metadata written to results. Defaults from backend profile.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model ID used by the backend and written to results. Defaults from backend profile.",
    )
    parser.add_argument(
        "--matched-planner",
        action="store_true",
        help="Use the same local model service for ReasonDB planning and execution.",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        help="Model type metadata written to results. Defaults from backend profile.",
    )
    parser.add_argument(
        "--energy-measurement",
        choices=["none", "nvml"],
        default="none",
        help="Energy source strategy. Use nvml only when NVIDIA GPU telemetry is available.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results/movie",
        help="Directory for measurement artifacts.",
    )
    parser.add_argument(
        "--idle-gpu-power-w",
        type=float,
        default=None,
        help="Measured post-load idle GPU power for supplementary net energy.",
    )
    parser.add_argument(
        "--hardware-provider",
        default=None,
        help="Hardware provider metadata, for example dgx or cloudlab.",
    )
    parser.add_argument(
        "--cloudlab-node-type",
        default=None,
        help="CloudLab node type metadata, for example r7525 or d8545.",
    )
    parser.add_argument(
        "--gpu-model",
        default=None,
        help="GPU model metadata written to results.",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="Number of GPUs visible/used for the run.",
    )
    parser.add_argument(
        "--model-cache-state",
        default=None,
        help="Model execution cache state, for example model_direct_no_kv_cache.",
    )
    parser.add_argument(
        "--local-cache-policy",
        choices=["default", "disabled"],
        default="default",
        help="Disable both client answer-cache reads/writes and use direct server inference.",
    )
    parser.add_argument(
        "--planner-cache-state",
        default="not_controlled",
        help="Expected planner cache state; the observed outcome is recorded per query.",
    )
    parser.add_argument(
        "--online-cache-policy",
        choices=["default", "disabled"],
        default="default",
        help="Disable OpenAI/ReasonDB LLM response cache for measured API runs.",
    )
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="Disable planner and local answer caches for a controlled measurement.",
    )
    parser.add_argument(
        "--run-namespace",
        default=os.getenv("GREENQUERY_RUN_NAMESPACE"),
        help="Optional unique suffix for the ReasonDB database and cold-cache path.",
    )
    parser.add_argument(
        "--split-artifacts",
        action="store_true",
        help="Move large result arrays into checksummed gzip sidecars.",
    )
    parser.add_argument("--demo-query", choices=["Q1", "Q2", "Q3", "Q4"])
    parser.add_argument(
        "--demo-q4-mode",
        choices=["count", "ratio"],
        default="count",
        help="Use the legacy 100-review count or the comparable 1,000-review ratio for Q4.",
    )
    parser.add_argument(
        "--semantic-row-limit",
        type=int,
        default=SEMANTIC_SAMPLE_ROWS,
        help="Number of Movie rows used by the semantic query.",
    )
    return parser.parse_args()


def baseline_type_for(query_set: str, backend_profile: str = "api") -> str:
    if query_set == "demo":
        return "movie_py_demo"
    if query_set == "baseline":
        return "traditional_planner"
    if query_set == "semantic":
        if backend_profile in ("dgx-local-text", "cloudlab-local-text"):
            return f"semantic_{backend_profile.replace('-', '_')}"
        return "semantic_api"
    return "mixed"


def default_model_metadata(backend_profile: str) -> tuple[str, str, str]:
    if backend_profile == "dgx-local-text":
        return "local", "meta-llama/Llama-3.1-8B-Instruct", "dgx"
    if backend_profile == "cloudlab-local-text":
        return "local", "meta-llama/Llama-3.1-8B-Instruct", "cloudlab"
    model = os.getenv("GREENQUERY_OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
    return provider_for(model), model, "api"


def configured_co2_factor() -> float | None:
    raw = os.getenv("GREENQUERY_CO2_G_PER_KWH")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        print(
            f"⚠️  Ignoring invalid GREENQUERY_CO2_G_PER_KWH={raw!r}; "
            "CO2 will remain not_available unless another source provides it."
        )
        return None


def run_movie_with_measurement(
    query_set: str = "baseline",
    operator_policy: str = "default",
    backend_profile: str = "api",
    device: str = "cpu",
    model_provider: str | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
    energy_measurement: str = "none",
    output_dir: str = "evaluation/results/movie",
    idle_gpu_power_w: float | None = None,
    hardware_provider: str | None = None,
    cloudlab_node_type: str | None = None,
    gpu_model: str | None = None,
    gpu_count: int | None = None,
    model_cache_state: str | None = None,
    local_cache_policy: str = "default",
    planner_cache_state: str = "not_controlled",
    online_cache_policy: str = "default",
    semantic_row_limit: int = SEMANTIC_SAMPLE_ROWS,
    only_q1: bool = False,
    split_artifacts: bool = False,
    demo_query: str | None = None,
    demo_q4_mode: str = "count",
    matched_planner: bool = False,
    run_namespace: str | None = None,
):
    """Run Movie demo with full measurement."""
    if semantic_row_limit <= 0:
        raise ValueError("semantic_row_limit must be positive")

    # Dataset path
    source_csv = (
        Path(__file__).parents[2]
        / "reasondb"
        / "evaluation"
        / "benchmarks"
        / "files"
        / "reviews_100.csv"
    )
    csv = (
        make_semantic_sample_csv(source_csv, semantic_row_limit)
        if query_set == "semantic"
        else source_csv
    )
    dataset_name = (
        f"movie_reviews_{semantic_row_limit}_semantic"
        if query_set == "semantic"
        else "movie_reviews_100"
    )
    dataset_source = (
        f"Sandbench movie reviews sample, first {semantic_row_limit} rows for semantic baseline"
        if query_set == "semantic"
        else "Sandbench movie reviews sample"
    )

    default_provider, default_name, default_type = default_model_metadata(backend_profile)
    model_name = model_name or default_name
    if backend_profile == "api":
        default_provider = provider_for(model_name)
    model_provider = model_provider or default_provider
    model_type = model_type or default_type
    if backend_profile == "api":
        os.environ["GREENQUERY_OPENAI_MODEL"] = model_name
    openai_cache_enabled = online_cache_policy != "disabled"
    hardware_provider = hardware_provider or (
        "cloudlab" if model_type == "cloudlab" else ("dgx" if model_type == "dgx" else None)
    )

    # Configuration
    database_name = movie_database_name(query_set, run_namespace)
    config = MeasurementConfig(
        demo_name="movie",
        device=device,
        precision_guarantee=0.8,
        recall_guarantee=0.8,
        output_dir=output_dir,
        model_name=model_name,
        planner_model_name=model_name if (matched_planner or backend_profile == "api") else None,
        model_provider=model_provider,
        model_type=model_type,
        backend_profile=backend_profile,
        energy_measurement=energy_measurement,
        co2_g_per_kwh=configured_co2_factor(),
        hardware_provider=hardware_provider,
        cloudlab_node_type=cloudlab_node_type,
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        idle_gpu_power_w=idle_gpu_power_w,
        reasondb_database_name=database_name,
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        dataset_rows=count_csv_rows(csv),
        dataset_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
        cache_state=os.getenv("MEASUREMENT_CACHE_STATE", "not_controlled"),
        model_cache_state=model_cache_state
        or os.getenv("MEASUREMENT_MODEL_CACHE_STATE", "not_controlled"),
        local_cache_policy=local_cache_policy,
        planner_cache_state=planner_cache_state,
        online_cache_policy=online_cache_policy,
        query_set=query_set,
        operator_policy=operator_policy,
        baseline_type=baseline_type_for(query_set, backend_profile),
    )

    print(f"📊 Starting Movie Demo Measurement")
    print(f"   Dataset: {csv}")
    print(f"   Dataset rows: {config.dataset_rows}")
    print(f"   Model: {config.model_provider}/{config.model_name}")
    print(f"   Model type: {config.model_type}")
    print(f"   Backend profile: {config.backend_profile}")
    print(f"   Hardware provider: {config.hardware_provider}")
    print(f"   CloudLab node type: {config.cloudlab_node_type}")
    print(f"   GPU model: {config.gpu_model}")
    print(f"   GPU count: {config.gpu_count}")
    print(f"   Energy measurement: {config.energy_measurement}")
    print(f"   Cache state: {config.cache_state}")
    print(f"   Model cache state: {config.model_cache_state}")
    print(f"   Planner cache state: {config.planner_cache_state}")
    print(f"   Online cache policy: {config.online_cache_policy}")
    print(f"   Query set: {config.query_set}")
    print(f"   Only Q1: {only_q1}")
    print(f"   Operator policy: {config.operator_policy}")
    print(f"   Device: {config.device}")
    print(f"   Precision: {config.precision_guarantee}")
    print(f"   Recall: {config.recall_guarantee}")
    print()

    # Start measurement
    config_dict = config.to_dict()
    if backend_profile == "api":
        config_dict["require_ecologits"] = model_requires_ecologits(config.model_name)
        config_dict["online_measurement_boundary"] = (
            "provider_estimated_from_ecologits"
            if config_dict["require_ecologits"]
            else "provider_usage_only_no_energy_estimate"
        )
    if run_namespace:
        config_dict["run_namespace"] = run_namespace
    with MeasurementWrapper("movie", config.output_dir, config_dict, split_artifacts=split_artifacts) as wrapper:

        reason_config = ReasonDBConfig(
            database_name,
            device=config.device,
            operator_policy=config.operator_policy,
            backend_profile=config.backend_profile,
            local_cache_policy=config.local_cache_policy,
            openai_model_id=config.model_name if config.backend_profile == "api" else None,
            openai_cache_enabled=openai_cache_enabled,
            local_text_model_id=config.model_name if config.backend_profile != "api" else None,
            local_planner_model_id=config.model_name if matched_planner else None,
        )
        with RaccoonDB(database_name, config=reason_config, device=config.device) as rc:
            reviews = rc.add_table(
                path=csv,
                table_name="reviews",
                text_columns=[InPlaceColumn("reviews.reviewtext")],
            )
            q2_reviews = q2_identities = q4_reviews = q4_identities = None
            if query_set == "demo" and (
                demo_query in (None, "Q2")
                or (demo_query in (None, "Q4") and demo_q4_mode == "count")
            ):
                q2_csv, q2_identities = make_movie_subset_csv(source_csv, "taken_3")
                q2_reviews = rc.add_table(
                    path=q2_csv,
                    table_name="taken_3_reviews",
                    text_columns=[InPlaceColumn("taken_3_reviews.reviewtext")],
                )
            if query_set == "demo" and demo_q4_mode == "ratio" and demo_query in (None, "Q4"):
                q4_source = source_csv.with_name("reviews_1000.csv")
                q4_csv, q4_identities = make_movie_subset_csv(q4_source, "taken_3")
                q4_reviews = rc.add_table(
                    path=q4_csv,
                    table_name="taken_3_ratio_reviews",
                    text_columns=[InPlaceColumn("taken_3_ratio_reviews.reviewtext")],
                )

            if query_set in ("baseline", "all"):
                # Q1: Positive reviews (filter + project + limit)
                print("🔍 Query 1: Finding 5 clearly positive reviews...")
                with wrapper.measure_query(
                    "Q1_positive_reviews",
                    "filter_project_limit",
                    baseline_type="traditional_planner",
                ) as tracker:
                    with open(csv, newline="") as identity_source:
                        tracker.set_identity_map(
                            [
                                (row["reviewtext"], row["reviewid"])
                                for row in csv_module.DictReader(identity_source)
                            ]
                        )
                    df_query = (
                        reviews
                        .filter("{reviewtext} is clearly positive")
                        .project("Keep {reviewid}")
                        .limit("Keep 5 rows")
                    )
                    result = df_query.execute(
                        "positive_reviews",
                        PrecisionGuarantee(config.precision_guarantee),
                        RecallGuarantee(config.recall_guarantee),
                    )
                    tracker.set_result(result)
                    print("   ✅ Query 1 complete")
                    result.pprint()
                    print()

                if not only_q1:
                    # Q2: Join query - reviews with same sentiment
                    print("🔍 Query 2: Finding reviews with same sentiment (join)...")
                    with wrapper.measure_query(
                        "Q2_reviews_same_sentiment",
                        "join",
                        baseline_type="traditional_planner",
                    ) as tracker:
                        df_query = reviews.join(
                            reviews.rename("Rename {reviewtext} to [reviewtext_other]"),
                            "{left.reviewtext} and {right.reviewtext_other} share the same sentiment",
                        )
                        result = df_query.execute(
                            "reviews_with_same_sentiment_v2",
                            PrecisionGuarantee(config.precision_guarantee)
                        )
                        tracker.set_result(result)
                    print("   ✅ Query 2 complete")
                    result.pprint()
                    print()

            if query_set == "demo":
                demo_queries = [
                    (
                        "Q1_positive_reviews",
                        "filter_project_limit",
                        lambda: reviews.filter("{reviewtext} is clearly positive")
                        .project("Keep {reviewid}")
                        .limit("Keep 5 rows"),
                        "positive_reviews",
                    ),
                    (
                        "Q2_positive_reviews_for_taken_3",
                        "filter_filter_project_limit",
                        lambda: q2_reviews.filter("{reviewtext} is clearly positive")
                        .project("Keep {reviewid}")
                        .limit("Keep 5 rows"),
                        "positive_reviews_for_taken_3",
                    ),
                    (
                        "Q3_reviews_same_sentiment",
                        "self_join",
                        lambda: reviews.join(
                            reviews.rename("Rename {reviewtext} to [reviewtext_other]").rename(
                                "Rename {reviewid} to [reviewid_other]"
                            ),
                            "{left.reviewtext} and {right.reviewtext_other} share the same sentiment",
                        ),
                        "reviews_with_same_sentiment",
                    ),
                    (
                        "Q4_positive_reviews_for_taken_3_ratio",
                        "filter_project_ratio",
                        lambda: q4_reviews.filter("{reviewtext} is clearly positive")
                        .project("Keep {reviewid}"),
                        "positive_reviews_for_taken_3_ratio",
                    )
                    if demo_q4_mode == "ratio"
                    else (
                        "Q4_positive_reviews_for_taken_3_count",
                        "filter_aggregate_count",
                        lambda: q2_reviews.filter("{reviewtext} is clearly positive")
                        .aggregate("Count reviews as [positive_reviews_for_taken_3_count]"),
                        "positive_reviews_for_taken_3_count",
                    ),
                ]
                if demo_query:
                    demo_queries = [item for item in demo_queries if item[0].startswith(demo_query + "_")]
                for name, operator, build_query, result_name in demo_queries:
                    print(f"🔍 {name}...")
                    with wrapper.measure_query(name, operator, baseline_type="movie_py_demo") as tracker:
                        if name.startswith("Q2"):
                            tracker.set_identity_map(q2_identities or [])
                            tracker.set_result_summary_fields(
                                deterministic_filter={"column": "id", "equals": "taken_3"},
                                evaluation_input_rows=len(q2_identities or []),
                                evaluation_dataset="reviews_100.csv",
                            )
                        if name.startswith("Q4"):
                            q4_dataset = "reviews_1000.csv" if demo_q4_mode == "ratio" else "reviews_100.csv"
                            tracker.set_identity_map(q4_identities or q2_identities or [])
                            tracker.set_result_summary_fields(
                                deterministic_filter={"column": "id", "equals": "taken_3"},
                                evaluation_input_rows=len(q4_identities or q2_identities or []),
                                evaluation_dataset=q4_dataset,
                            )
                        result = build_query().execute(
                            result_name,
                            PrecisionGuarantee(config.precision_guarantee),
                            RecallGuarantee(config.recall_guarantee),
                        )
                        tracker.set_result(result)
                    print(f"   ✅ {name} complete")

            if query_set in ("semantic", "all") and not only_q1:
                # Q3: Semantic API baseline - force text QA operator evidence.
                print("🔍 Query 3: Finding reviews driven by actor/performance judgment...")
                with wrapper.measure_query(
                    "Q3_actor_performance_semantic_filter",
                    "semantic_filter",
                    baseline_type=baseline_type_for("semantic", backend_profile),
                ) as tracker:
                    with open(csv, newline="") as identity_source:
                        tracker.set_identity_map(
                            [
                                (row["reviewtext"], row["reviewid"])
                                for row in csv_module.DictReader(identity_source)
                            ]
                        )
                    df_query = (
                        reviews
                        .filter(
                            "{reviewtext} says an actor performance is a major reason "
                            "for the reviewer's opinion"
                        )
                    )
                    result = df_query.execute(
                        "actor_performance_semantic_reviews",
                        PrecisionGuarantee(config.precision_guarantee),
                        RecallGuarantee(config.recall_guarantee),
                    )
                    require_semantic_operator(
                        result,
                        "Q3_actor_performance_semantic_filter",
                    )
                    tracker.set_result(result)
                    print("   ✅ Query 3 complete")
                    result.pprint()
                    print()

                if operator_policy != "semantic-text-only":
                    print(
                        "   ⚠️  Semantic Q3 passed operator verification, but "
                        "semantic-text-only policy was not enabled."
                    )

            if query_set == "baseline" and operator_policy != "default":
                print(
                    "   ⚠️  Baseline query set ran with a non-default operator policy. "
                    "Use --operator-policy default for comparable Q1/Q2 baseline runs."
                )

    print("\n✅ Movie measurement complete!")
    print(f"   Results saved to: {config.output_dir}")


if __name__ == "__main__":
    # Verify API key is set
    load_env_file()
    args = parse_args()
    if args.cold_cache:
        args.online_cache_policy = "disabled"
        args.local_cache_policy = "disabled"
        args.planner_cache_state = "cache_disabled"
        cleared_cache = clear_movie_database_cache(args.query_set, args.run_namespace)
        print(f"🧹 Cleared Movie metadata cache: {cleared_cache}")
    if args.backend_profile == "api":
        try:
            require_model_credentials(args.model_name or default_model_metadata("api")[1])
        except RuntimeError as error:
            print(f"❌ Error: {error}")
            sys.exit(1)

    run_movie_with_measurement(
        query_set=args.query_set,
        operator_policy=args.operator_policy,
        backend_profile=args.backend_profile,
        device=args.device,
        model_provider=args.model_provider,
        model_name=args.model_name,
        model_type=args.model_type,
        energy_measurement=args.energy_measurement,
        output_dir=args.output_dir,
        idle_gpu_power_w=args.idle_gpu_power_w,
        hardware_provider=args.hardware_provider,
        cloudlab_node_type=args.cloudlab_node_type,
        gpu_model=args.gpu_model,
        gpu_count=args.gpu_count,
        model_cache_state=args.model_cache_state,
        local_cache_policy=args.local_cache_policy,
        planner_cache_state=args.planner_cache_state,
        online_cache_policy=args.online_cache_policy,
        semantic_row_limit=args.semantic_row_limit,
        only_q1=args.only_q1,
        split_artifacts=args.split_artifacts,
        demo_query=args.demo_query,
        demo_q4_mode=args.demo_q4_mode,
        matched_planner=args.matched_planner,
        run_namespace=args.run_namespace,
    )

# Made with Bob
