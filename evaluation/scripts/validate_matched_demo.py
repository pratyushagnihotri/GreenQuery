"""Fail unless one matched-planner measurement has the required evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.artifacts import load_measurement


MOVIE_QUERIES = {
    "Q1_positive_reviews",
    "Q2_positive_reviews_for_taken_3",
    "Q3_reviews_same_sentiment",
    "Q4_positive_reviews_for_taken_3_ratio",
}
ROTOWIRE_QUERIES = {
    "R1_player_points_assists",
    "R2_player_points_home_team_won",
}
MEDICAL_QUERIES = {
    "Q1_allergy_patients",
    "Q3_cancer_family_lung_problems_patients",
    "Q4_average_age_acne_patients",
    "Q9_sick_patients_skin_xray",
}
MEDICAL_ROLE_PHASE_PROVENANCE = "nvml_per_gpu_role_local_inference_windows_v2"
ANIMALS_QUERIES = {"Q1_zebra_image_count", "Q2_lion_audio_count", "Q3_top_zebra_city", "Q10_top_zebra_city_station"}
ANIMALS_VISION_MODEL = "llava-hf/llama3-llava-next-8b-hf"
ANIMALS_AUDIO_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
ANIMALS_GROUND_TRUTH = {
    "Q1_zebra_image_count": ("reasondb/evaluation/ground_truth/animals/Q1_500.csv", 27.0),
    "Q2_lion_audio_count": ("reasondb/evaluation/ground_truth/animals/Q2_500.csv", 5.0),
    "Q3_top_zebra_city": ("reasondb/evaluation/ground_truth/animals/Q3_500.csv", 1),
    "Q10_top_zebra_city_station": ("reasondb/evaluation/ground_truth/animals/Q10_500.csv", 2),
}
ARTWORK_QUERIES = {
    "A1_madonna_and_child",
    "A2_more_than_two_people",
    "A3_more_than_three_people",
    "A4_saints_with_halos",
    "A5_death_dominant_theme",
    "A6_madonna_and_child_century",
}
ARTWORK_LABELS = {
    "A1_madonna_and_child": "m&c",
    "A2_more_than_two_people": "more_than_2_people",
    "A3_more_than_three_people": "more_than_3_people",
    "A4_saints_with_halos": "saints_with_halos",
    "A5_death_dominant_theme": "death_theme",
}
ARTWORK_EXPECTED_ROWS = {
    "A1_madonna_and_child": 9,
    "A2_more_than_two_people": 38,
    "A3_more_than_three_people": 33,
    "A4_saints_with_halos": 33,
    "A5_death_dominant_theme": 16,
}
ARTWORK_VISION_MODEL = "llava-hf/llama3-llava-next-8b-hf"
ARTWORK_TEXT_MODELS = {"Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"}


def rotowire_failures(data: dict, model: str) -> list[str]:
    failures = []
    config = data.get("config", {})
    queries = data.get("queries", [])
    if len(queries) != 2 or {query.get("query_name") for query in queries} != ROTOWIRE_QUERIES:
        failures.append("RotoWire must contain exactly R1 and R2")
    namespace = config.get("reasondb_database_name")
    if not isinstance(namespace, str) or not re.fullmatch(r"rotowire_matched_demo_(qwen|llama)_\d+", namespace):
        failures.append("missing unique RotoWire ReasonDB namespace evidence")
    if config.get("model_name") != model or config.get("planner_model_name") != model:
        failures.append("stored RotoWire planner/executor models do not match")
    for field in ("cache_state", "model_cache_state", "local_cache_policy", "planner_cache_state", "online_cache_policy"):
        if config.get(field) != "disabled":
            failures.append(f"RotoWire {field} is not disabled")

    for query in queries:
        name = query.get("query_name")
        local = query.get("local_inference") or {}
        if query.get("model_name") != model:
            failures.append(f"{name}: executor model does not match")
        if (query.get("energy") or {}).get("metric_source") != "measured_from_nvml_query_window":
            failures.append(f"{name}: total energy is not measured from NVML")
        for field in ("answer_cache_hits", "kv_cache_hits"):
            if field not in local or local[field] != 0:
                failures.append(f"{name}: missing explicit zero {field}")

        phase = (query.get("result_summary") or {}).get("phase_energy") or {}
        planner = phase.get("planner") or {}
        execution = phase.get("execution") or {}
        if not phase.get("valid") or phase.get("provenance") != "nvml_non_overlapping_local_inference_windows_v1":
            failures.append(f"{name}: invalid phase energy provenance")
        if phase.get("energy_source") != "measured_from_nvml_phase_windows" or phase.get("co2_source") != "estimated_from_configured_emissions_factor":
            failures.append(f"{name}: invalid phase energy/CO2 source")
        if planner.get("sample_count", 0) <= 0 or execution.get("sample_count", 0) <= 0:
            failures.append(f"{name}: missing positive planner/execution phase samples")

        values = (
            query.get("energy_kwh"), query.get("co2_g"),
            planner.get("energy_kwh"), planner.get("co2_g"),
            execution.get("energy_kwh"), execution.get("co2_g"),
            phase.get("unattributed_query_energy_kwh"), phase.get("unattributed_query_co2_g"),
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in values):
            failures.append(f"{name}: invalid phase energy/CO2 value")
        else:
            total_energy, total_co2, planner_energy, planner_co2, execution_energy, execution_co2, unattributed_energy, unattributed_co2 = values
            if planner_energy > total_energy or execution_energy > total_energy:
                failures.append(f"{name}: phase energy exceeds query energy")
            if not math.isclose(planner_energy + execution_energy + unattributed_energy, total_energy, rel_tol=1e-6, abs_tol=1e-12):
                failures.append(f"{name}: phase energy does not conserve query energy")
            if not math.isclose(planner_co2 + execution_co2 + unattributed_co2, total_co2, rel_tol=1e-6, abs_tol=1e-12):
                failures.append(f"{name}: phase CO2 does not conserve query CO2")
        if name == "R1_player_points_assists" and ((query.get("result_summary") or {}).get("quality") or {}).get("checked_fields", 0) <= 0:
            failures.append(f"{name}: checked_fields is not positive")
    return failures


def execution_plan_failures(query: dict) -> list[str]:
    name = query.get("query_name")
    plan = query.get("execution_plan") or {}
    logical_ids = {
        step.get("id")
        for step in (plan.get("logical") or {}).get("steps") or []
        if step.get("id")
    }
    physical_steps = (plan.get("physical") or {}).get("steps") or []
    implemented_ids = {
        edge.get("from")
        for edge in plan.get("edges") or []
        if edge.get("kind") == "implementation"
    }
    if not logical_ids:
        return [f"{name}: missing execution-plan logical steps"]
    if not physical_steps:
        return [f"{name}: missing execution-plan physical steps"]
    if not logical_ids <= implemented_ids:
        return [f"{name}: incomplete logical-to-physical execution-plan coverage"]
    return []


def medical_phase_failures(data: dict) -> list[str]:
    failures = []
    config_roles = data.get("config", {}).get("nvml_gpu_roles") or {}
    if data.get("config", {}).get("gpu_count") != 2:
        failures.append("Medical requires exactly two GPUs")
    if set(config_roles) != {"text", "vision"} or not all(config_roles.values()) or config_roles.get("text") == config_roles.get("vision"):
        failures.append("Medical requires distinct physical text/vision GPU roles")

    for query in data.get("queries", []):
        name = query.get("query_name")
        summary = query.get("result_summary") or {}
        phase = summary.get("phase_energy") or {}
        planner = phase.get("planner") or {}
        execution = phase.get("execution") or {}
        if (
            not phase.get("valid")
            or phase.get("provenance") != MEDICAL_ROLE_PHASE_PROVENANCE
            or phase.get("energy_source") != "measured_from_nvml_per_gpu_role_windows"
            or phase.get("co2_source") != "estimated_from_configured_emissions_factor"
            or phase.get("gpu_roles") != config_roles
        ):
            failures.append(f"{name}: invalid per-GPU role phase evidence")

        samples = (summary.get("energy_measurement") or {}).get("power_samples") or []
        if len(samples) < 2 or any(
            set((sample.get("power_watts_by_gpu") or {})) != set(config_roles.values())
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                for value in (sample.get("power_watts_by_gpu") or {}).values()
            )
            for sample in samples
        ):
            failures.append(f"{name}: missing separate physical-GPU power samples")

        planner_roles = planner.get("energy_kwh_by_gpu_role") or {}
        execution_roles = execution.get("energy_kwh_by_gpu_role") or {}
        expected_execution_role = "vision" if name in {
            "Q3_cancer_family_lung_problems_patients",
            "Q9_sick_patients_skin_xray",
        } else "text"
        if planner_roles.get("text", 0) <= 0 or set(planner_roles) != {"text"}:
            failures.append(f"{name}: planner energy is not text-GPU-only")
        if execution_roles.get(expected_execution_role, 0) <= 0 or any(
            isinstance(value, (int, float)) and value > 0
            for role, value in execution_roles.items()
            if role != expected_execution_role
        ):
            failures.append(f"{name}: execution energy crosses Medical GPU roles")

        per_gpu = phase.get("query_energy_kwh_by_gpu") or {}
        if set(per_gpu) != set(config_roles.values()) or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in per_gpu.values()
        ):
            failures.append(f"{name}: invalid per-GPU query energy")

        values = (
            query.get("energy_kwh"), query.get("co2_g"),
            planner.get("energy_kwh"), planner.get("co2_g"),
            execution.get("energy_kwh"), execution.get("co2_g"),
            phase.get("unattributed_query_energy_kwh"), phase.get("unattributed_query_co2_g"),
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in values):
            failures.append(f"{name}: invalid Medical phase energy/CO2 value")
            continue
        total_energy, total_co2, planner_energy, planner_co2, execution_energy, execution_co2, residual_energy, residual_co2 = values
        if not math.isclose(planner_energy + execution_energy + residual_energy, total_energy, rel_tol=1e-6, abs_tol=1e-12):
            failures.append(f"{name}: Medical phase energy does not conserve query energy")
        if per_gpu and not math.isclose(sum(per_gpu.values()), total_energy, rel_tol=1e-6, abs_tol=1e-12):
            failures.append(f"{name}: per-GPU energy does not conserve query energy")
        if not math.isclose(planner_co2 + execution_co2 + residual_co2, total_co2, rel_tol=1e-6, abs_tol=1e-12):
            failures.append(f"{name}: Medical phase CO2 does not conserve query CO2")
    return failures


def artwork_failures(data: dict, model: str) -> list[str]:
    failures = []
    config = data.get("config", {})
    queries = data.get("queries", [])
    roles = config.get("nvml_gpu_roles") or {}
    if model not in ARTWORK_TEXT_MODELS:
        failures.append("Artwork model is not Qwen 2.5 7B or Llama 3.1 8B")
    if len(queries) != 6 or {query.get("query_name") for query in queries} != ARTWORK_QUERIES:
        failures.append("Artwork must contain exactly A1 through A6")
    if config.get("gpu_count") != 2 or set(roles) != {"text", "vision"} or not all(roles.values()) or roles["text"] == roles["vision"]:
        failures.append("Artwork requires distinct physical text/vision GPU roles")
    namespace = config.get("run_namespace")
    if not isinstance(namespace, str) or not re.fullmatch(r"artwork_matched_demo_(qwen|llama)_\d+", namespace):
        failures.append("missing unique Artwork ReasonDB namespace evidence")
    if (
        config.get("model_name") != model
        or config.get("planner_model_name") != model
        or config.get("local_text_model_id") != model
        or config.get("local_planner_model_id") != model
    ):
        failures.append("stored Artwork planner/executor models do not match")
    if config.get("local_vision_model_id") != ARTWORK_VISION_MODEL:
        failures.append("stored Artwork vision model is not the required LLaVA model")
    if config.get("backend_profile") != "dgx-local-text-vision":
        failures.append("Artwork backend profile is not local text+vision")
    if config.get("direct_qa_batch_size") != 8 or config.get("vision_qa_batch_size") != 8:
        failures.append("Artwork direct text/vision batch sizes are not 8")
    for field in ("cache_state", "model_cache_state", "local_cache_policy", "planner_cache_state", "online_cache_policy"):
        if config.get(field) != "disabled":
            failures.append(f"Artwork {field} is not disabled")

    for query in queries:
        name = query.get("query_name")
        summary = query.get("result_summary") or {}
        local = query.get("local_inference") or {}
        events = local.get("events") or []
        quality = summary.get("quality") or {}
        if summary.get("expected_query_ids") != sorted(ARTWORK_QUERIES):
            failures.append(f"{name}: wrong expected query IDs")
        if query.get("model_name") != model:
            failures.append(f"{name}: executor model does not match")
        if (query.get("energy") or {}).get("metric_source") != "measured_from_nvml_query_window":
            failures.append(f"{name}: total energy is not measured from NVML")
        if query.get("energy_kwh", 0) <= 0:
            failures.append(f"{name}: no positive NVML energy")
        for field in ("answer_cache_hits", "kv_cache_hits"):
            if field not in local or local[field] != 0:
                failures.append(f"{name}: missing explicit zero {field}")
        if local.get("planner_request_count", 0) <= 0:
            failures.append(f"{name}: no local planner request")
        if not any(
            event.get("event_kind") == "planner"
            and event.get("execution_mode") == "local_planner_generation"
            and event.get("model_name") == model
            for event in events
        ):
            failures.append(f"{name}: no matched text planner event")
        if not any(
            event.get("event_kind") == "vision"
            and event.get("execution_mode") == "direct_no_cache"
            and event.get("model_name") == ARTWORK_VISION_MODEL
            and event.get("inference_batch_size") == 8
            for event in events
        ):
            failures.append(f"{name}: no direct LLaVA vision event")
        if name == "A6_madonna_and_child_century" and not any(
            event.get("event_kind") != "vision"
            and event.get("execution_mode") == "direct_no_cache"
            and event.get("model_name") == model
            and event.get("inference_batch_size") == 8
            for event in events
        ):
            failures.append(f"{name}: no direct matched text extraction event")

        if quality.get("status") != "evaluated":
            failures.append(f"{name}: missing ground-truth quality")
        elif (
            quality.get("ground_truth") != "reasondb/evaluation/ground_truth/artwork/artwork_no_duplicated.csv"
            or not re.fullmatch(r"[0-9a-f]{64}", str(quality.get("ground_truth_sha256", "")))
        ):
            failures.append(f"{name}: missing Artwork ground-truth provenance")
        elif name in ARTWORK_LABELS and (
            quality.get("metric") != "artwork_title_set_precision_recall"
            or quality.get("label_column") != ARTWORK_LABELS[name]
            or quality.get("expected_rows") != ARTWORK_EXPECTED_ROWS[name]
            or any(
                not isinstance(quality.get(field), (int, float))
                or not math.isfinite(quality[field])
                or not 0 <= quality[field] <= 1
                for field in ("precision", "recall", "f1")
            )
        ):
            failures.append(f"{name}: wrong selection quality metadata")
        elif name == "A6_madonna_and_child_century":
            selection = quality.get("selection") or {}
            century = quality.get("century_extraction") or {}
            if (
                quality.get("metric") != "artwork_selection_and_century"
                or selection.get("status") != "evaluated"
                or selection.get("metric") != "artwork_title_set_precision_recall"
                or selection.get("label_column") != "m&c"
                or selection.get("expected_rows") != 9
                or century.get("status") != "evaluated"
                or century.get("label_column") != "century"
                or century.get("metric") != "exact_century_accuracy_on_correctly_selected_rows"
                or century.get("checked_rows") != selection.get("correct_rows")
                or not isinstance(century.get("accuracy"), (int, float))
                or not math.isfinite(century.get("accuracy", math.nan))
                or not 0 <= century.get("accuracy", -1) <= 1
                or not isinstance(century.get("correct_rows"), int)
                or not 0 <= century.get("correct_rows", -1) <= century.get("checked_rows", -1)
                or (
                    century.get("checked_rows", 0) > 0
                    and not math.isclose(
                        century.get("accuracy", -1),
                        century.get("correct_rows", -1) / century["checked_rows"],
                    )
                )
                or (century.get("checked_rows") == 0 and century.get("accuracy") != 0.0)
            ):
                failures.append(f"{name}: wrong selection/century quality metadata")

        phase = summary.get("phase_energy") or {}
        planner = phase.get("planner") or {}
        execution = phase.get("execution") or {}
        if (
            not phase.get("valid")
            or phase.get("provenance") != MEDICAL_ROLE_PHASE_PROVENANCE
            or phase.get("energy_source") != "measured_from_nvml_per_gpu_role_windows"
            or phase.get("co2_source") != "estimated_from_configured_emissions_factor"
            or phase.get("gpu_roles") != roles
        ):
            failures.append(f"{name}: invalid per-GPU role phase evidence")
        samples = (summary.get("energy_measurement") or {}).get("power_samples") or []
        if len(samples) < 2 or any(
            set((sample.get("power_watts_by_gpu") or {})) != set(roles.values())
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                for value in (sample.get("power_watts_by_gpu") or {}).values()
            )
            for sample in samples
        ):
            failures.append(f"{name}: missing separate physical-GPU power samples")
        planner_roles = planner.get("energy_kwh_by_gpu_role") or {}
        execution_roles = execution.get("energy_kwh_by_gpu_role") or {}
        if set(planner_roles) != {"text"} or planner_roles.get("text", 0) <= 0:
            failures.append(f"{name}: planner energy is not text-GPU-only")
        expected_execution_roles = {"vision", "text"} if name == "A6_madonna_and_child_century" else {"vision"}
        if any(execution_roles.get(role, 0) <= 0 for role in expected_execution_roles) or any(
            isinstance(value, (int, float)) and value > 0
            for role, value in execution_roles.items()
            if role not in expected_execution_roles
        ):
            failures.append(f"{name}: execution energy crosses Artwork GPU roles")
        per_gpu = phase.get("query_energy_kwh_by_gpu") or {}
        values = (
            query.get("energy_kwh"), query.get("co2_g"),
            planner.get("energy_kwh"), planner.get("co2_g"),
            execution.get("energy_kwh"), execution.get("co2_g"),
            phase.get("unattributed_query_energy_kwh"), phase.get("unattributed_query_co2_g"),
        )
        if set(per_gpu) != set(roles.values()) or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in (*per_gpu.values(), *values)
        ):
            failures.append(f"{name}: invalid Artwork phase energy/CO2 value")
        else:
            total_energy, total_co2, planner_energy, planner_co2, execution_energy, execution_co2, residual_energy, residual_co2 = values
            if not math.isclose(planner_energy + execution_energy + residual_energy, total_energy, rel_tol=1e-6, abs_tol=1e-12):
                failures.append(f"{name}: Artwork phase energy does not conserve query energy")
            if not math.isclose(sum(per_gpu.values()), total_energy, rel_tol=1e-6, abs_tol=1e-12):
                failures.append(f"{name}: per-GPU energy does not conserve query energy")
            if not math.isclose(planner_co2 + execution_co2 + residual_co2, total_co2, rel_tol=1e-6, abs_tol=1e-12):
                failures.append(f"{name}: Artwork phase CO2 does not conserve query CO2")
    return failures


def animals_failures(data: dict, model: str) -> list[str]:
    failures = []
    config, queries = data.get("config", {}), data.get("queries", [])
    roles = config.get("nvml_gpu_roles") or {}
    if len(queries) != 4 or {query.get("query_name") for query in queries} != ANIMALS_QUERIES:
        failures.append("Animals must contain exactly Q1/Q2/Q3/Q10")
    if config.get("gpu_count") != 3 or set(roles) != {"text", "vision", "audio"} or not all(roles.values()) or len(set(roles.values())) != 3:
        failures.append("Animals requires distinct physical text/vision/audio GPU roles")
    if config.get("backend_profile") != "dgx-local-text-vision-audio":
        failures.append("Animals backend profile is not local text+vision+audio")
    if any(config.get(field) != model for field in ("model_name", "planner_model_name", "local_text_model_id", "local_planner_model_id")):
        failures.append("stored Animals planner/executor models do not match")
    if config.get("local_vision_model_id") != ANIMALS_VISION_MODEL or config.get("local_audio_model_id") != ANIMALS_AUDIO_MODEL:
        failures.append("stored Animals execution models do not match")
    if any(config.get(field) != 8 for field in ("direct_qa_batch_size", "vision_qa_batch_size", "audio_qa_batch_size")):
        failures.append("Animals direct inference batch sizes are not 8")
    namespace = config.get("run_namespace")
    if not isinstance(namespace, str) or not re.fullmatch(r"animals_matched_demo_(qwen|llama)_\d+", namespace):
        failures.append("missing unique Animals ReasonDB namespace evidence")
    for field in ("cache_state", "model_cache_state", "local_cache_policy", "planner_cache_state", "online_cache_policy"):
        if config.get(field) != "disabled":
            failures.append(f"Animals {field} is not disabled")

    for query in queries:
        name = query.get("query_name")
        summary, local = query.get("result_summary") or {}, query.get("local_inference") or {}
        events, quality = local.get("events") or [], summary.get("quality") or {}
        execution_role = "audio" if name == "Q2_lion_audio_count" else "vision"
        execution_model = ANIMALS_AUDIO_MODEL if execution_role == "audio" else ANIMALS_VISION_MODEL
        if query.get("model_name") != model:
            failures.append(f"{name}: executor model does not match")
        if (query.get("energy") or {}).get("metric_source") != "measured_from_nvml_query_window":
            failures.append(f"{name}: total energy is not measured from NVML")
        if any(field not in local or local[field] != 0 for field in ("answer_cache_hits", "kv_cache_hits")):
            failures.append(f"{name}: missing explicit zero cache-hit evidence")
        if summary.get("expected_query_ids") != list(("Q1_zebra_image_count", "Q2_lion_audio_count", "Q3_top_zebra_city", "Q10_top_zebra_city_station")):
            failures.append(f"{name}: wrong expected query IDs")
        planner_required = name != "Q2_lion_audio_count"
        if summary.get("planner_phase_required") is not planner_required:
            failures.append(f"{name}: wrong deterministic-planner phase marker")
        planner_events = [
            event for event in events
            if event.get("event_kind") == "planner"
            and event.get("execution_mode") == "local_planner_generation"
            and event.get("model_name") == model
        ]
        if planner_required and not planner_events:
            failures.append(f"{name}: missing matched planner event")
        if not planner_required and planner_events:
            failures.append(f"{name}: unexpected planner event for deterministic configuration")
        if not any(
            event.get("event_kind") == execution_role
            and event.get("execution_mode") == "direct_no_cache"
            and event.get("model_name") == execution_model
            and event.get("inference_batch_size") == 8
            and isinstance(event.get("inference_started_at"), (int, float))
            and isinstance(event.get("inference_finished_at"), (int, float))
            and event["inference_finished_at"] > event["inference_started_at"]
            for event in events
        ):
            failures.append(f"{name}: missing timestamped direct {execution_role} event")
        expected_ground_truth, expected_value = ANIMALS_GROUND_TRUTH.get(name, (None, None))
        if quality.get("status") != "evaluated" or quality.get("ground_truth") != expected_ground_truth or not re.fullmatch(r"[0-9a-f]{64}", str(quality.get("ground_truth_sha256", ""))):
            failures.append(f"{name}: missing ground-truth provenance")
        elif name in {"Q1_zebra_image_count", "Q2_lion_audio_count"} and (
            quality.get("metric") != "exact_aggregate_count" or quality.get("expected_count") != expected_value
        ):
            failures.append(f"{name}: wrong aggregate quality metadata")
        elif name in {"Q3_top_zebra_city", "Q10_top_zebra_city_station"} and (
            quality.get("metric") != "top_group_set_precision_recall"
            or quality.get("expected_rows") != expected_value
            or any(not isinstance(quality.get(field), (int, float)) or not math.isfinite(quality[field]) or not 0 <= quality[field] <= 1 for field in ("precision", "recall", "f1"))
        ):
            failures.append(f"{name}: wrong grouped quality metadata")

        phase = summary.get("phase_energy") or {}
        planner, execution = phase.get("planner") or {}, phase.get("execution") or {}
        samples = (summary.get("energy_measurement") or {}).get("power_samples") or []
        if not phase.get("valid") or phase.get("planner_required") is not planner_required or phase.get("provenance") != MEDICAL_ROLE_PHASE_PROVENANCE or phase.get("gpu_roles") != roles or phase.get("energy_source") != "measured_from_nvml_per_gpu_role_windows" or phase.get("co2_source") != "estimated_from_configured_emissions_factor":
            failures.append(f"{name}: invalid three-GPU phase provenance")
        if len(samples) < 2 or any(
            set((sample.get("power_watts_by_gpu") or {})) != set(roles.values())
            or not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in (sample.get("power_watts_by_gpu") or {}).values())
            for sample in samples
        ):
            failures.append(f"{name}: missing three-GPU NVML samples")
        planner_roles, execution_roles = planner.get("energy_kwh_by_gpu_role") or {}, execution.get("energy_kwh_by_gpu_role") or {}
        if set(planner_roles) != {"text"} or (
            planner_roles.get("text", 0) <= 0 if planner_required else planner_roles.get("text", 0) != 0
        ):
            failures.append(f"{name}: planner energy is not text-GPU-only")
        if execution_roles.get(execution_role, 0) <= 0 or any(isinstance(value, (int, float)) and value > 0 for role, value in execution_roles.items() if role != execution_role):
            failures.append(f"{name}: execution energy crosses Animals GPU roles")
        per_gpu = phase.get("query_energy_kwh_by_gpu") or {}
        values = (query.get("energy_kwh"), query.get("co2_g"), planner.get("energy_kwh"), planner.get("co2_g"), execution.get("energy_kwh"), execution.get("co2_g"), phase.get("unattributed_query_energy_kwh"), phase.get("unattributed_query_co2_g"))
        if set(per_gpu) != set(roles.values()) or not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in (*per_gpu.values(), *values)):
            failures.append(f"{name}: invalid Animals phase energy/CO2 values")
        else:
            total_energy, total_co2, planner_energy, planner_co2, execution_energy, execution_co2, residual_energy, residual_co2 = values
            if not math.isclose(planner_energy + execution_energy + residual_energy, total_energy, rel_tol=1e-6, abs_tol=1e-12) or not math.isclose(sum(per_gpu.values()), total_energy, rel_tol=1e-6, abs_tol=1e-12):
                failures.append(f"{name}: Animals phase energy does not conserve query energy")
            if not math.isclose(planner_co2 + execution_co2 + residual_co2, total_co2, rel_tol=1e-6, abs_tol=1e-12):
                failures.append(f"{name}: Animals phase CO2 does not conserve query CO2")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--demo", choices=["movie", "rotowire", "medical", "artwork", "animals"], required=True)
    args = parser.parse_args()
    # Validation inspects per-inference events, which may live in a split sidecar.
    data = load_measurement(args.measurement, hydrate=True)
    failures = []
    if data.get("config", {}).get("planner_model_name") != args.model:
        failures.append("planner model does not match execution model")
    if args.demo == "movie":
        namespace = data.get("config", {}).get("run_namespace")
        database_name = data.get("config", {}).get("reasondb_database_name")
        if not namespace or not database_name or not database_name.endswith(f"_{namespace}"):
            failures.append("missing per-job database namespace")
    if args.demo == "medical":
        config = data.get("config", {})
        if config.get("local_text_model_id") != args.model or config.get("local_planner_model_id") != args.model:
            failures.append("stored local planner/executor IDs do not match")
        if not config.get("run_namespace"):
            failures.append("missing per-job database namespace")
        if config.get("vision_qa_batch_size") != 8:
            failures.append("vision QA batch size is not 8")
    if args.demo == "artwork":
        failures.extend(artwork_failures(data, args.model))
    if args.demo == "animals":
        failures.extend(animals_failures(data, args.model))
    queries = data.get("queries", [])
    expected = {
        "movie": MOVIE_QUERIES,
        "rotowire": ROTOWIRE_QUERIES,
        "medical": MEDICAL_QUERIES,
        "artwork": ARTWORK_QUERIES,
        "animals": ANIMALS_QUERIES,
    }[args.demo]
    if {query.get("query_name") for query in queries} != expected:
        failures.append("unexpected query set")
    if args.demo == "rotowire":
        failures.extend(rotowire_failures(data, args.model))
    if args.demo == "medical":
        failures.extend(medical_phase_failures(data))
    movie_quality = {}
    if args.demo == "movie":
        summary_path = args.measurement.with_name(f"{args.measurement.stem}_summary.json")
        if not summary_path.exists():
            failures.append("missing Movie quality summary")
        else:
            movie_quality = {
                item.get("query"): item.get("quality") or {}
                for item in json.loads(summary_path.read_text()).get("queries", [])
            }
    if (
        args.demo in {"rotowire", "medical", "artwork", "animals"}
        and data.get("config", {}).get("direct_qa_batch_size") != 8
    ):
        failures.append("direct QA batch size is not 8")
    for query in queries:
        local = query.get("local_inference") or {}
        if query.get("status") != "success":
            failures.append(f"{query.get('query_name')}: status is not success")
        if local.get("answer_cache_hits", 0) or local.get("kv_cache_hits", 0):
            failures.append(f"{query.get('query_name')}: cache hit")
        planner_required = not (
            args.demo == "animals"
            and query.get("query_name") == "Q2_lion_audio_count"
        )
        if planner_required and local.get("planner_request_count", 0) <= 0:
            failures.append(f"{query.get('query_name')}: no local planner request")
        if query.get("energy_kwh", 0) <= 0:
            failures.append(f"{query.get('query_name')}: no positive NVML energy")
        if args.demo not in {"medical", "artwork", "animals"}:
            phase_energy = (query.get("result_summary") or {}).get("phase_energy") or {}
            planner = phase_energy.get("planner") or {}
            execution = phase_energy.get("execution") or {}
            if phase_energy.get("provenance") != "nvml_non_overlapping_local_inference_windows_v1" or not phase_energy.get("valid"):
                failures.append(f"{query.get('query_name')}: invalid phase energy provenance")
            elif not planner.get("window_count") or not execution.get("window_count"):
                failures.append(f"{query.get('query_name')}: missing phase window")
            elif planner.get("energy_kwh", 0) + execution.get("energy_kwh", 0) > query.get("energy_kwh", 0) + 1e-12:
                failures.append(f"{query.get('query_name')}: phase energy exceeds query energy")
        if args.demo == "movie":
            failures.extend(execution_plan_failures(query))
            quality = movie_quality.get(query.get("query_name"), {})
            if quality.get("status") != "evaluated":
                failures.append(f"{query.get('query_name')}: missing ground-truth quality")
            if query.get("query_name") == "Q4_positive_reviews_for_taken_3_ratio" and quality.get("metric") != "ratio_absolute_error":
                failures.append("Q4_positive_reviews_for_taken_3_ratio: wrong quality metric")
        if args.demo == "medical" and (query.get("result_summary") or {}).get("quality", {}).get("status") != "evaluated":
            failures.append(f"{query.get('query_name')}: missing ground-truth quality")
        if args.demo == "medical" and query.get("query_name") in {
            "Q3_cancer_family_lung_problems_patients",
            "Q9_sick_patients_skin_xray",
        } and not any(event.get("event_kind") == "vision" for event in local.get("events", [])):
            failures.append(f"{query.get('query_name')}: no LLaVA vision telemetry")
        for event in local.get("events", []):
            if args.demo in {"rotowire", "medical", "artwork", "animals"} and (
                event.get("execution_mode") == "direct_no_cache"
                and event.get("inference_batch_size") != 8
            ):
                failures.append(f"{query.get('query_name')}: direct QA batch telemetry is not 8")
    if failures:
        raise SystemExit("validation failed: " + "; ".join(failures))
    print(f"validated {args.demo} {args.model}: {len(queries)} queries")


if __name__ == "__main__":
    main()
