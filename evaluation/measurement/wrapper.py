"""Measurement wrapper for demo execution."""

import json
import os
import re
import time
import csv
import subprocess
import hashlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from ecologits import EcoLogits
except Exception:
    EcoLogits = None

from .api_usage import (
    events_since,
    snapshot_index,
    start_collection,
    stop_collection,
    summarize,
)
from .metrics import APIMetrics, CostMetrics, EnergyMetrics, QueryResult, RuntimeMetrics
from .energy_counters import CounterSampler, counters_enabled, summarize_window
from .nvml import NvidiaSmiEnergySampler, build_phase_energy_metrics, derive_phase_windows
from .local_inference import (
    events_since as local_events_since,
    snapshot_index as local_snapshot_index,
    start_collection as start_local_collection,
    stop_collection as stop_local_collection,
    summarize as summarize_local,
)
from .artifacts import save_measurement

DETERMINISTIC_DATABASE_OPERATORS = {
    "Aggregate",
    "GroupBy",
    "Limit",
    "Project",
    "Rename",
    "Sort",
    "TraditionalFilter",
    "TraditionalJoin",
}


def _json_scalar(value: Any) -> Any:
    """Convert dataframe scalar values without expanding result rows."""
    if hasattr(value, "item"):
        value = value.item()
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def build_operator_metrics(
    operator_plan: list[str],
    local_events: list[dict[str, Any]],
    sampler: NvidiaSmiEnergySampler | None,
    query_energy_kwh: float,
    query_co2_g: float,
) -> tuple[list[dict[str, Any]], float]:
    """Summarize measured local inference by physical operator instance."""
    planned = []
    for plan in operator_plan:
        planned.extend(re.findall(r'"operator":\s*"([^"]+)"', plan))
    metrics: dict[str, dict[str, Any]] = {}
    raw_energy: dict[str, float] = {}
    for event in local_events:
        operator = event.get("operator")
        if not operator:
            continue
        instance_id = operator["instance_id"]
        metric = metrics.setdefault(
            instance_id,
            {
                **operator,
                "attribution": (
                    "measured_planner_inference"
                    if operator.get("phase") == "planner_configuration"
                    else "measured_local_inference"
                ),
                "request_count": 0,
                "input_rows": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "inference_seconds": 0.0,
                "answer_cache_hits": 0,
                "kv_cache_hits": 0,
                "energy_kwh": 0.0,
                "co2_g": 0.0,
            },
        )
        metric["request_count"] += int(event.get("request_count", 0))
        metric["input_rows"] += int(event.get("processed_rows", 0))
        metric["input_tokens"] += int(event.get("input_tokens", 0))
        metric["output_tokens"] += int(event.get("output_tokens", 0))
        metric["inference_seconds"] += float(event.get("inference_seconds", 0.0))
        metric["answer_cache_hits"] += int(event.get("answer_cache_hits", 0))
        metric["kv_cache_hits"] += int(event.get("kv_cache_hits", 0))
        if sampler is not None:
            energy_kwh, _ = sampler.energy_between(
                float(event.get("inference_started_at") or 0.0),
                float(event.get("inference_finished_at") or 0.0),
            )
            raw_energy[instance_id] = raw_energy.get(instance_id, 0.0) + energy_kwh

    total_raw_energy = sum(raw_energy.values())
    scale = min(1.0, query_energy_kwh / total_raw_energy) if total_raw_energy else 0.0
    co2_per_kwh = query_co2_g / query_energy_kwh if query_energy_kwh else 0.0
    for instance_id, metric in metrics.items():
        metric["energy_kwh"] = raw_energy.get(instance_id, 0.0) * scale
        metric["co2_g"] = metric["energy_kwh"] * co2_per_kwh
        metric["energy_attribution"] = (
            "nvml_inference_window" if scale == 1.0 else "nvml_overlap_normalized"
        )

    observed_operators = {metric["operator"] for metric in metrics.values()}
    for index, operator in enumerate(planned):
        if operator not in observed_operators:
            metrics[f"plan_{index}:{operator}"] = {
                "instance_id": f"plan_{index}:{operator}",
                "step_index": index,
                "operator": operator,
                "expression": None,
                "input_tables": [],
                "output_table": None,
                "attribution": "plan_only_db_operator",
                "energy_attribution": "not_separately_measured",
                "request_count": None,
                "input_rows": None,
                "input_tokens": None,
                "output_tokens": None,
                "inference_seconds": None,
                "answer_cache_hits": None,
                "kv_cache_hits": None,
                "energy_kwh": None,
                "co2_g": None,
            }
    attributed_energy = sum(metric["energy_kwh"] or 0.0 for metric in metrics.values())
    return list(metrics.values()), max(query_energy_kwh - attributed_energy, 0.0)


def build_execution_plan(
    operator_plan: list[str], logical_plan: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Turn ReasonDB's recorded plan fragments into a stable UI contract."""
    def expression_key(value: Any) -> str:
        text = str(value or "")
        return re.sub(r"\{[^}.]+\.([^}]+)\}", r"{\1}", text)

    logical_raw: dict[tuple[str, str], dict[str, Any]] = {
        (str(step.get("output")), str(step.get("expression"))): step
        for step in (logical_plan or [])
        if isinstance(step, dict)
    }
    physical_raw: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    for fragment in operator_plan:
        try:
            steps = json.loads(fragment)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            logical = step.get("logical_plan_step")
            if isinstance(logical, dict):
                key = (str(logical.get("output")), str(logical.get("expression")))
                logical_raw[key] = logical
                candidates.append({"logical_key": key, "available_operators": step.get("available_operators", [])})
            elif "operator" in step:
                inputs = tuple(str(value) for value in step.get("inputs", []))
                key = (str(step.get("operator")), str(step.get("output")), inputs)
                physical_raw[key] = step

    logical_ids = {key: f"logical_{index}" for index, key in enumerate(logical_raw)}

    def matched_logical_key(step: dict[str, Any]) -> tuple[str, str] | None:
        config = step.get("operator_config") or {}
        physical_expression = expression_key(config.get("__expression__"))
        return next(
            (candidate for candidate, value in logical_raw.items()
             if value.get("expression") == config.get("__expression__")),
            None,
        ) or next(
            (candidate for candidate, value in logical_raw.items()
             if expression_key(value.get("expression")) == physical_expression),
            None,
        )

    # Deterministic database operators can be selected without a tuning event;
    # an unexecuted semantic candidate must not be presented as a physical step.
    selected_logical = {
        key for step in physical_raw.values() if (key := matched_logical_key(step))
    }
    for candidate in candidates:
        logical_key = tuple(candidate["logical_key"])
        options = candidate["available_operators"] or []
        if logical_key in selected_logical or len(options) != 1:
            continue
        logical = logical_raw.get(logical_key)
        option = options[0]
        if not logical or option.get("operator") not in DETERMINISTIC_DATABASE_OPERATORS:
            continue
        key = (
            str(option["operator"]),
            str(logical.get("output")),
            tuple(str(value) for value in logical.get("inputs", [])),
        )
        physical_raw[key] = {
            "operator": option["operator"],
            "operator_config": option.get("operator_config", {}),
            "tuning_parameters": option.get("tuning_parameters", {}),
            "inputs": logical.get("inputs", []),
            "output": logical.get("output"),
            "selection_source": "single_available_operator",
            "_recorded_logical_key": logical_key,
        }

    physical_ids = {key: f"physical_{index}" for index, key in enumerate(physical_raw)}
    logical_by_output = {key[0]: key for key in logical_raw}
    logical_steps = [
        {"id": logical_ids[key], **step}
        for key, step in logical_raw.items()
    ]
    physical_steps = []
    for key, step in physical_raw.items():
        logical_key = step.get("_recorded_logical_key") or matched_logical_key(step)
        physical_steps.append({
            "id": physical_ids[key],
            "logical_step_id": logical_ids.get(logical_key),
            **{name: value for name, value in step.items() if name != "_recorded_logical_key"},
        })

    logical_edges = [
        {"from": logical_ids[logical_by_output[input_table]], "to": logical_ids[key], "kind": "data_flow"}
        for key, step in logical_raw.items()
        for input_table in step.get("inputs", [])
        if str(input_table) in logical_by_output
    ]
    physical_by_logical = {
        step["logical_step_id"]: step["id"]
        for step in physical_steps
        if step["logical_step_id"] is not None
    }
    physical_edges = [
        {"from": physical_by_logical[edge["from"]], "to": physical_by_logical[edge["to"]], "kind": "data_flow"}
        for edge in logical_edges
        if edge["from"] in physical_by_logical
        and edge["to"] in physical_by_logical
        and physical_by_logical[edge["from"]] != physical_by_logical[edge["to"]]
    ]
    implementation_edges = [
        {"from": step["logical_step_id"], "to": step["id"], "kind": "implementation"}
        for step in physical_steps if step["logical_step_id"] is not None
    ]
    return {
        "schema_version": 1,
        "logical": {"steps": logical_steps, "edges": logical_edges},
        "physical": {"steps": physical_steps, "edges": physical_edges},
        "edges": implementation_edges,
        "optimization": {
            "candidate_steps": [
                {"logical_step_id": logical_ids.get(tuple(item["logical_key"])), "available_operators": item["available_operators"]}
                for item in candidates
            ]
        },
    }


class MeasurementWrapper:
    """Wrapper for measuring demo execution."""

    def __init__(
        self,
        demo_name: str,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None,
        split_artifacts: bool = False,
    ):
        self.demo_name = demo_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.split_artifacts = split_artifacts
        # Which optimizer and operator models produced this run (see reasondb/interface/config.py).
        self.config.setdefault("optimizer", os.environ.get("GREENQUERY_OPTIMIZER", "stretto"))
        operator_models = [m.strip() for m in os.environ.get("GREENQUERY_OPERATOR_MODELS", "").split(",") if m.strip()]
        self.config.setdefault("operator_mode", "multi_model" if operator_models else "single_model")
        self.config.setdefault("operator_models", operator_models or None)

        if self.config.get("model_provider") == "openai":
            if EcoLogits is None:
                raise RuntimeError("Measured OpenAI runs require ecologits[openai].")
            # Anthropic and OpenAI-compatible calls compute EcoLogits impacts
            # explicitly in reasondb.reasoning.providers instead of SDK patching.
            EcoLogits.init(providers=["openai"])

        self.start_time = None
        self.queries = []
        self.environment = self._collect_environment()
        self._api_collection_token = None
        self._local_collection_token = None

    def _collect_environment(self) -> Dict[str, Any]:
        """Collect environment information."""
        import platform
        import sys

        import duckdb
        import openai

        try:
            import ecologits

            ecologits_version = ecologits.__version__
        except Exception:
            ecologits_version = "unknown"

        def command_output(command: list[str]) -> Optional[str]:
            try:
                return subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()
            except Exception:
                return None

        gpu_details = command_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ]
        )
        code_hasher = hashlib.sha256()
        code_files = [
            Path("evaluation/measurement/artifacts.py"),
            Path("evaluation/measurement/local_inference.py"),
            Path("evaluation/measurement/wrapper.py"),
            Path("evaluation/scripts/run_movie_measurement.py"),
            Path("evaluation/scripts/run_rotowire_measurement.py"),
            Path("evaluation/scripts/summarize_phase4_real_numbers.py"),
            Path("evaluation/scripts/analyze_native_movie_online_results.py"),
            Path("evaluation/scripts/build_phase4_online_comparison.py"),
            Path("reasondb/reasoning/llm.py"),
            Path("reasondb/backends/text_qa.py"),
            Path("reasondb/backends/kv_cache_text_qa_server.py"),
            Path("reasondb/interface/config.py"),
            Path("reasondb/query_plan/physical_operator.py"),
        ]
        for path in code_files:
            if path.exists():
                code_hasher.update(str(path).encode())
                code_hasher.update(path.read_bytes())
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "duckdb_version": duckdb.__version__,
            "openai_version": openai.__version__,
            "ecologits_version": ecologits_version,
            "device": self.config.get("device", "cpu"),
            "git_commit": command_output(["git", "rev-parse", "HEAD"])
            or os.getenv("PHASE4_SOURCE_COMMIT"),
            "code_sha256": code_hasher.hexdigest(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "slurm_node": os.getenv("SLURMD_NODENAME") or os.getenv("HOSTNAME"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "gpu_details": gpu_details,
            "cuda_version": command_output(["nvcc", "--version"]),
        }

    def __enter__(self):
        """Enter measurement context."""
        self.start_time = time.time()
        self._api_collection_token = start_collection()
        self._local_collection_token = start_local_collection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit measurement context and save results."""
        end_time = time.time()
        if self._api_collection_token is not None:
            stop_collection(self._api_collection_token)
            self._api_collection_token = None
        if self._local_collection_token is not None:
            stop_local_collection(self._local_collection_token)
            self._local_collection_token = None

        result = {
            "measurement_id": f"{self.demo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "demo_name": self.demo_name,
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "config": self.config,
            "metric_schema_version": "phase_2_5_real_measurement_v1",
            "metric_source_policy": {
                "measured": "Collected from runtime execution or OpenAI response usage.",
                "estimated": "Calculated from measured values and documented model prices.",
                "estimated_by_ecologits": "Estimated by EcoLogits/provider impact metadata when exposed for the API call path.",
                "measured_from_nvml": "Integrated from GPU power telemetry sampled through nvidia-smi/NVML during the query window.",
                "estimated_from_configured_emissions_factor": "Calculated from measured energy and GREENQUERY_CO2_G_PER_KWH.",
                "not_applicable_local_model": "Metric applies to remote API calls and is not applicable to this local model run.",
                "not_available": "Not emitted by the underlying tool for this run.",
            },
            "queries": [q.to_dict() for q in self.queries],
            "totals": self._calculate_totals(),
            "total_execution_time_seconds": end_time - self.start_time,
        }

        for query in result["queries"]:
            predictions = (query.get("local_inference") or {}).get("predictions") or []
            predictions_file = self.output_dir / (
                f"{result['measurement_id']}_{query['query_name']}_predictions.csv"
            )
            with open(predictions_file, "w", newline="") as destination:
                fieldnames = (
                    ["reviewid", "answer", "predicted_label", "log_odds", "input_tokens", "output_tokens"]
                    if predictions
                    else ["result_id", "result_value"]
                )
                writer = csv.DictWriter(destination, fieldnames=fieldnames)
                writer.writeheader()
                if predictions:
                    for prediction in predictions:
                        writer.writerow({key: prediction.get(key) for key in fieldnames})
                else:
                    summary = query.get("result_summary") or {}
                    for result_id in summary.get("result_ids") or []:
                        writer.writerow({"result_id": result_id, "result_value": None})
                    for key, value in (summary.get("result_values") or {}).items():
                        writer.writerow({"result_id": key, "result_value": value})

        output_file = self.output_dir / f"{result['measurement_id']}.json"
        if self.split_artifacts:
            save_measurement(result, output_file)
        else:
            with output_file.open("w") as destination:
                json.dump(result, destination, indent=2)

        print(f"\n✅ Measurement saved to: {output_file}")
        return False

    def add_query_result(self, query_result: QueryResult):
        """Add a query result to the measurement."""
        self.queries.append(query_result)

    def _calculate_totals(self) -> Dict[str, Any]:
        """Calculate total metrics across all queries."""
        total_api_calls = sum(q.api.total_calls for q in self.queries)
        total_cached_calls = sum(q.api.cached_calls for q in self.queries)
        total_tokens = sum(
            (
                int(q.local_inference.get("execution_input_tokens", 0))
                + int(q.local_inference.get("execution_output_tokens", 0))
                + int(q.local_inference.get("planner_input_tokens", 0))
                + int(q.local_inference.get("planner_output_tokens", 0))
                if q.local_inference
                else q.api.total_tokens
            )
            for q in self.queries
        )
        total_local_requests = sum(
            int((q.local_inference or {}).get("request_count", 0))
            for q in self.queries
        )
        total_energy = sum(q.energy.total_kwh for q in self.queries)
        total_carbon = sum(q.energy.total_gco2eq for q in self.queries)
        total_cost = sum(q.cost.total_usd for q in self.queries)

        return {
            "total_queries": len(self.queries),
            "total_api_calls": total_api_calls,
            "total_cached_calls": total_cached_calls,
            "total_tokens": total_tokens,
            "total_local_inference_requests": total_local_requests,
            "total_energy_kwh": total_energy,
            "total_carbon_gco2eq": total_carbon,
            "total_co2_g": total_carbon,
            "total_cost_usd": total_cost,
        }

    @contextmanager
    def measure_query(
        self,
        query_name: str,
        query_type: str,
        baseline_type: Optional[str] = None,
        planner_required: bool = True,
    ):
        """Measure one query execution."""
        tracker = QueryTracker(
            query_name, query_type, self.config, baseline_type, planner_required
        )
        try:
            yield tracker
        except Exception:
            tracker.close()
            raise
        self.add_query_result(tracker.get_result())


class QueryTracker:
    """Tracks metrics for a single query execution."""

    def __init__(
        self,
        query_name: str,
        query_type: str,
        config: Dict[str, Any],
        baseline_type: Optional[str] = None,
        planner_required: bool = True,
    ):
        self.query_name = query_name
        self.query_type = query_type
        self.config = config
        self.baseline_type = baseline_type or config.get("baseline_type")
        self.planner_required = planner_required
        self.start_time = time.time()
        self.api_snapshot = snapshot_index()
        self.local_snapshot = local_snapshot_index()
        self.result_data = None
        self.identity_rows: list[tuple[str, str]] = []
        self.result_summary_fields: dict[str, Any] = {}
        self.energy_sampler = None
        self.counter_sampler: Optional[CounterSampler] = None

        if self.config.get("energy_measurement") == "nvml":
            self.energy_sampler = NvidiaSmiEnergySampler()
            self.energy_sampler.start()
        # Energy counters run alongside (not instead of) the nvidia-smi sampler so
        # both methods measure identical windows and can be compared.
        if counters_enabled():
            self.counter_sampler = CounterSampler().start()

    def close(self):
        """Stop any background measurement resources."""
        if self.energy_sampler is not None:
            self.energy_sampler.stop()
            self.energy_sampler = None
        if self.counter_sampler is not None:
            self.counter_sampler.stop()
            self.counter_sampler = None

    def set_result(self, result_data: Any):
        """Set the query result data."""
        self.result_data = result_data

    def set_identity_map(self, identity_rows: list[tuple[str, str]]):
        self.identity_rows = identity_rows

    def set_result_summary_fields(self, **fields: Any):
        self.result_summary_fields.update(fields)

    def _extract_row_count(self) -> tuple[int, str]:
        rows_returned = 0
        extraction_method = "not_available"
        if self.result_data is None:
            return rows_returned, extraction_method

        try:
            if hasattr(self.result_data, "to_df"):
                df = self.result_data.to_df()
                rows_returned = len(df)
                extraction_method = "to_df"
            elif hasattr(self.result_data, "__len__"):
                rows_returned = len(self.result_data)
                extraction_method = "direct"
        except Exception as e:
            print(f"Warning: Could not extract row count: {e}")
            extraction_method = "failed"
        return rows_returned, extraction_method

    def _extract_reasondb_cost(self) -> Optional[Dict[str, Any]]:
        cost_summary = getattr(self.result_data, "cost_summary", None)
        if cost_summary is None:
            return None
        if hasattr(cost_summary, "to_json"):
            return cost_summary.to_json()
        return None

    def _extract_operator_plan(self) -> list[str]:
        pipelines = getattr(self.result_data, "tuned_pipelines", []) or []
        return [str(pipeline) for pipeline in pipelines]

    def _extract_physical_operators(self, operator_plan: list[str]) -> list[str]:
        operators = []
        for plan in operator_plan:
            operators.extend(re.findall(r'"operator":\s*"([^"]+)"', plan))
        return sorted(set(operators))

    def get_result(self) -> QueryResult:
        """Get the complete query result with metrics."""
        end_time = time.time()
        api_summary = summarize(events_since(self.api_snapshot))
        local_summary = summarize_local(local_events_since(self.local_snapshot))
        if self.config.get("require_ecologits") and api_summary["total_calls"]:
            if not api_summary["impact_complete"]:
                raise RuntimeError("EcoLogits did not provide energy/CO2 for every API call.")
            phases = api_summary["phase_metrics"]
            if self.planner_required and not phases["planning"]["total_calls"]:
                raise RuntimeError("Planner API calls were not tagged as planning.")
            if not phases["execution"]["total_calls"]:
                raise RuntimeError("Execution API calls were not tagged as execution.")
        planner_events = [event for event in local_summary["events"] if event.get("event_kind") == "planner"]
        local_summary["planner_input_tokens"] = sum(int(event.get("input_tokens", 0)) for event in planner_events)
        local_summary["planner_output_tokens"] = sum(int(event.get("output_tokens", 0)) for event in planner_events)
        local_summary["planner_inference_seconds"] = sum(float(event.get("inference_seconds", 0.0)) for event in planner_events)
        local_summary["planner_request_count"] = sum(int(event.get("request_count", 0)) for event in planner_events)
        execution_event = next(
            (
                event
                for event in reversed(local_summary["events"])
                if int(event.get("processed_rows", 0))
                == int(self.config.get("dataset_rows") or 0)
            ),
            local_summary["events"][-1] if local_summary["events"] else {},
        )
        local_summary["execution_input_tokens"] = int(
            execution_event.get("input_tokens", 0)
        )
        local_summary["execution_output_tokens"] = int(
            execution_event.get("output_tokens", 0)
        )
        local_summary["execution_inference_seconds"] = float(
            execution_event.get("inference_seconds", 0.0)
        )
        local_summary["execution_started_at"] = execution_event.get(
            "inference_started_at"
        )
        local_summary["execution_finished_at"] = execution_event.get(
            "inference_finished_at"
        )
        nvml_summary = None
        if self.energy_sampler is not None:
            sampler = self.energy_sampler
            nvml_summary = sampler.stop()
            self.energy_sampler = None
        counter_summary = None
        if self.counter_sampler is not None:
            counter_sampler = self.counter_sampler
            counter_sampler.stop()
            self.counter_sampler = None
            factor = self.config.get("co2_g_per_kwh")
            counter_summary = summarize_window(
                counter_sampler,
                self.start_time,
                end_time,
                derive_phase_windows(local_summary["events"])["windows"],
                float(factor) if factor is not None else None,
            )

        runtime = RuntimeMetrics(
            start_time=self.start_time,
            end_time=end_time,
            duration_seconds=end_time - self.start_time,
            query_name=self.query_name,
        )

        rows_returned, extraction_method = self._extract_row_count()
        result_ids = []
        result_identity_columns = []
        result_identity_tuples = []
        result_values = {}
        try:
            frame = self.result_data.to_df()
            identity_columns = [
                column
                for column in frame.columns
                if any(token in str(column).lower() for token in ("reviewid", "patient_id"))
            ]
            if identity_columns:
                result_ids = sorted(str(value) for value in frame[identity_columns[0]].tolist())
            if len(identity_columns) >= 2:
                result_identity_columns = [str(column) for column in identity_columns]
                result_identity_tuples = sorted(
                    [str(value) for value in row]
                    for row in frame[identity_columns].itertuples(index=False, name=None)
                )
            if len(frame) == 1:
                result_values = {
                    str(column): _json_scalar(value)
                    for column, value in frame.iloc[0].items()
                }
        except AttributeError:
            pass
        row_token_details = {
            int(row["row_offset"]): row for row in execution_event.get("rows") or []
        }
        predictions = []
        identity_queues: dict[str, list[str]] = {}
        for context, reviewid in self.identity_rows:
            identity_queues.setdefault(context, []).append(reviewid)
        for prediction_index, prediction in enumerate(
            execution_event.get("predictions") or []
        ):
                item = dict(prediction)
                context = item.pop("context", "")
                matching_ids = identity_queues.get(context) or []
                item["reviewid"] = matching_ids.pop(0) if matching_ids else None
                token_detail = row_token_details.get(prediction_index, {})
                item["input_tokens"] = item.get("input_tokens") or token_detail.get(
                    "input_tokens"
                )
                item["output_tokens"] = item.get("output_tokens") or token_detail.get(
                    "output_tokens"
                )
                predictions.append(item)
        local_summary["predictions"] = predictions
        local_summary["inference_rows"] = len(predictions)
        local_summary["predicted_positive_ids"] = sorted(
            str(item["reviewid"])
            for item in predictions
            if item.get("predicted_label") == 1 and item.get("reviewid") is not None
        )
        local_summary["result_ids"] = result_ids
        local_summary["prediction_result_ids_match"] = (
            bool(predictions)
            and local_summary["predicted_positive_ids"] == result_ids
        )
        reasondb_cost = self._extract_reasondb_cost()
        operator_plan = self._extract_operator_plan()
        physical_operators = self._extract_physical_operators(operator_plan)
        execution_plan = build_execution_plan(
            operator_plan, getattr(self.result_data, "logical_plan", None)
        )
        if api_summary["total_calls"]:
            api_metric_source = "measured_from_" + "_".join(api_summary["providers"]) + "_usage"
            planner_cache_state = "cache_miss_api_call"
        elif api_summary["cached_calls"]:
            api_metric_source = "measured_cache_hit_no_api_call"
            planner_cache_state = "cache_hit_no_api_call"
        elif self.config.get("model_provider") == "local":
            api_metric_source = "not_applicable_local_model"
            planner_cache_state = self.config.get(
                "planner_cache_state", "not_observed"
            )
        else:
            api_metric_source = "not_available"
            planner_cache_state = self.config.get(
                "planner_cache_state", "not_observed"
            )

        api = APIMetrics(
            total_calls=api_summary["total_calls"],
            cached_calls=api_summary["cached_calls"],
            models_used=api_summary["models_used"],
            providers=api_summary["providers"],
            total_prompt_tokens=api_summary["total_prompt_tokens"],
            total_completion_tokens=api_summary["total_completion_tokens"],
            total_tokens=api_summary["total_tokens"],
            metric_source=api_metric_source,
        )

        energy_kwh = api_summary["energy_kwh"]
        co2_g = api_summary["co2_g"]
        energy_source = api_summary["energy_source"]
        co2_source = api_summary["co2_source"]
        energy_details = None

        if nvml_summary is not None:
            # The primary local metric is the complete query window: it includes
            # planner and executor work performed by the same GPU model.
            energy_kwh = nvml_summary.energy_kwh
            factor = self.config.get("co2_g_per_kwh")
            co2_g = energy_kwh * float(factor) if factor is not None else 0.0
            energy_source = (
                "measured_from_nvml_query_window"
                if nvml_summary.sample_count >= 2 and energy_kwh > 0
                else "not_available"
            )
            co2_source = nvml_summary.co2_source
            energy_details = {
                "method": "nvidia-smi_power_draw_sampling_query_window",
                "gpu_index": os.getenv("GREENQUERY_GPU_INDEX"),
                "sample_count": nvml_summary.sample_count,
                "error": nvml_summary.error,
                "sample_started_at": nvml_summary.sample_started_at,
                "sample_finished_at": nvml_summary.sample_finished_at,
                "raw_query_window_energy_kwh": nvml_summary.energy_kwh,
                "query_energy_kwh_by_gpu": nvml_summary.per_gpu_energy_kwh,
                "power_samples": sampler.serialized_samples(),
            }

        if nvml_summary is None and counter_summary and counter_summary["gpu_energy_kwh"] is not None:
            # Counter-only runs (sampler disabled, e.g. to quantify its overhead)
            # still need a primary local energy figure.
            energy_kwh = counter_summary["gpu_energy_kwh"]
            factor = self.config.get("co2_g_per_kwh")
            co2_g = energy_kwh * float(factor) if factor is not None else 0.0
            energy_source = "measured_from_nvml_energy_counter_query_window"
            co2_source = (
                "estimated_from_configured_emissions_factor" if factor is not None else "not_available"
            )

        divisor = api_summary["total_calls"] or 1
        per_call_kwh = energy_kwh / divisor if api_summary["total_calls"] else 0.0
        per_call_gco2eq = co2_g / divisor if api_summary["total_calls"] else 0.0
        energy = EnergyMetrics(
            total_kwh=energy_kwh,
            per_call_kwh=per_call_kwh,
            total_gco2eq=co2_g,
            per_call_gco2eq=per_call_gco2eq,
            metric_source=energy_source,
        )

        cost = CostMetrics(
            total_usd=api_summary["total_cost_usd"],
            prompt_cost_usd=api_summary["prompt_cost_usd"],
            completion_cost_usd=api_summary["completion_cost_usd"],
            metric_source=api_metric_source,
        )

        row_source = (
            "measured" if extraction_method not in ("failed", "not_available") else extraction_method
        )
        result_summary = {
            "rows_returned": rows_returned,
            "extraction_method": extraction_method,
            "row_count_source": row_source,
            "result_ids": result_ids,
        }
        if result_identity_columns:
            result_summary["result_identity_columns"] = result_identity_columns
            result_summary["result_identity_tuples"] = result_identity_tuples
        if result_values:
            result_summary["result_values"] = result_values
        result_summary.update(self.result_summary_fields)
        result_summary["planner_phase_required"] = self.planner_required
        result_summary["api_phase_metrics"] = api_summary["phase_metrics"]
        result_summary["api_energy_kwh_min"] = api_summary["energy_kwh_min"]
        result_summary["api_energy_kwh_max"] = api_summary["energy_kwh_max"]
        result_summary["api_co2_g_min"] = api_summary["co2_g_min"]
        result_summary["api_co2_g_max"] = api_summary["co2_g_max"]
        metric_sources = {
            "runtime": "measured",
            "rows": row_source,
            "api_calls": api_metric_source,
            "tokens": (
                "measured_from_local_inference"
                if local_summary["processed_rows"]
                else api_metric_source
            ),
            "api_cost": api_metric_source,
            "energy": energy_source,
            "co2": co2_source,
            "reasondb_cost": "measured" if reasondb_cost else "not_available",
            "physical_operators": "measured" if physical_operators else "not_available",
            "execution_plan": "recorded_from_reasondb_plan_serialization",
        }
        if energy_details is not None:
            result_summary["energy_measurement"] = energy_details
        if counter_summary is not None:
            result_summary["energy_counters"] = counter_summary
            metric_sources["energy_counters"] = (
                "measured_from_hardware_counters"
                if counter_summary["gpu_energy_kwh"] is not None or counter_summary["cpu_package_energy_kwh"] is not None
                else "not_available"
            )
        operator_metrics, _ = build_operator_metrics(
            operator_plan,
            local_summary["events"],
            sampler if nvml_summary is not None else None,
            energy_kwh,
            co2_g,
        )
        if nvml_summary is not None:
            phase_energy = build_phase_energy_metrics(
                local_summary["events"],
                sampler,
                energy_kwh,
                co2_g,
                co2_source,
                self.config.get("nvml_gpu_roles"),
                self.planner_required,
            )
            result_summary["phase_energy"] = phase_energy
            result_summary["unattributed_query_energy_kwh"] = phase_energy[
                "unattributed_query_energy_kwh"
            ]
            result_summary["unattributed_query_co2_g"] = phase_energy[
                "unattributed_query_co2_g"
            ]

        return QueryResult(
            query_name=self.query_name,
            query_type=self.query_type,
            runtime=runtime,
            api=api,
            energy=energy,
            cost=cost,
            result_summary=result_summary,
            dataset_name=self.config.get("dataset_name"),
            dataset_source=self.config.get("dataset_source"),
            dataset_rows=self.config.get("dataset_rows"),
            model_provider=self.config.get("model_provider"),
            model_name=self.config.get("model_name"),
            model_type=self.config.get("model_type"),
            backend_profile=self.config.get("backend_profile"),
            energy_measurement=self.config.get("energy_measurement"),
            hardware_provider=self.config.get("hardware_provider"),
            cloudlab_node_type=self.config.get("cloudlab_node_type"),
            gpu_model=self.config.get("gpu_model"),
            gpu_count=self.config.get("gpu_count"),
            cache_state=self.config.get("cache_state"),
            model_cache_state=self.config.get("model_cache_state"),
            planner_cache_state=planner_cache_state,
            query_set=self.config.get("query_set"),
            operator_policy=self.config.get("operator_policy"),
            baseline_type=self.baseline_type,
            physical_operators=physical_operators,
            operator_plan=operator_plan,
            reasondb_cost=reasondb_cost,
            api_events=api_summary["events"],
            local_inference=local_summary,
            operator_metrics=operator_metrics,
            execution_plan=execution_plan,
            metric_sources=metric_sources,
        )
