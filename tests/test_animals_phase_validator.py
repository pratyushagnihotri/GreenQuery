import copy
import unittest

from evaluation.scripts.validate_matched_demo import animals_failures


MODEL = "Qwen/Qwen2.5-7B-Instruct"
ROLES = {"text": "2", "vision": "5", "audio": "7"}


def query(name):
    role = "audio" if name == "Q2_lion_audio_count" else "vision"
    execution_model = "Qwen/Qwen2-Audio-7B-Instruct" if role == "audio" else "llava-hf/llama3-llava-next-8b-hf"
    grouped = name in {"Q3_top_zebra_city", "Q10_top_zebra_city_station"}
    quality = {
        "status": "evaluated", "metric": "top_group_set_precision_recall" if grouped else "exact_aggregate_count",
        "ground_truth_sha256": "a" * 64,
        "ground_truth": f"reasondb/evaluation/ground_truth/animals/{'Q10' if name.startswith('Q10') else name[:2]}_500.csv",
    }
    if grouped:
        quality.update({"expected_rows": 1 if name == "Q3_top_zebra_city" else 2, "precision": 1.0, "recall": 1.0, "f1": 1.0})
    else:
        quality["expected_count"] = 27.0 if name == "Q1_zebra_image_count" else 5.0
    planner_required = name != "Q2_lion_audio_count"
    events = [
        {"event_kind": role, "execution_mode": "direct_no_cache", "model_name": execution_model, "inference_batch_size": 8, "inference_started_at": 1.0, "inference_finished_at": 2.0},
    ]
    if planner_required:
        events.insert(0, {"event_kind": "planner", "execution_mode": "local_planner_generation", "model_name": MODEL})
    return {
        "query_name": name, "model_name": MODEL, "energy_kwh": 1.0, "co2_g": 10.0,
        "energy": {"metric_source": "measured_from_nvml_query_window"},
        "local_inference": {"answer_cache_hits": 0, "kv_cache_hits": 0, "events": events},
        "result_summary": {
            "expected_query_ids": ["Q1_zebra_image_count", "Q2_lion_audio_count", "Q3_top_zebra_city", "Q10_top_zebra_city_station"], "planner_phase_required": planner_required,
            "quality": quality,
            "energy_measurement": {"power_samples": [
                {"power_watts_by_gpu": {"2": 100.0, "5": 100.0, "7": 100.0}},
                {"power_watts_by_gpu": {"2": 100.0, "5": 100.0, "7": 100.0}},
            ]},
            "phase_energy": {
                "valid": True, "planner_required": planner_required, "provenance": "nvml_per_gpu_role_local_inference_windows_v2",
                "energy_source": "measured_from_nvml_per_gpu_role_windows", "co2_source": "estimated_from_configured_emissions_factor", "gpu_roles": ROLES,
                "query_energy_kwh_by_gpu": {"2": 0.4, "5": 0.3, "7": 0.3},
                "planner": {"energy_kwh": 0.2 if planner_required else 0.0, "co2_g": 2.0 if planner_required else 0.0, "energy_kwh_by_gpu_role": {"text": 0.2 if planner_required else 0.0}},
                "execution": {"energy_kwh": 0.3, "co2_g": 3.0, "energy_kwh_by_gpu_role": {
                    "text": 0.0, "vision": 0.3 if role == "vision" else 0.0, "audio": 0.3 if role == "audio" else 0.0,
                }},
                "unattributed_query_energy_kwh": 0.5 if planner_required else 0.7, "unattributed_query_co2_g": 5.0 if planner_required else 7.0,
            },
        },
    }


class AnimalsPhaseValidatorTests(unittest.TestCase):
    def test_valid_three_gpu_evidence_and_audio_timestamps(self):
        data = {
            "config": {
                "gpu_count": 3, "nvml_gpu_roles": ROLES, "backend_profile": "dgx-local-text-vision-audio",
                "model_name": MODEL, "planner_model_name": MODEL, "local_text_model_id": MODEL, "local_planner_model_id": MODEL,
                "local_vision_model_id": "llava-hf/llama3-llava-next-8b-hf", "local_audio_model_id": "Qwen/Qwen2-Audio-7B-Instruct",
                "direct_qa_batch_size": 8, "vision_qa_batch_size": 8, "audio_qa_batch_size": 8,
                "run_namespace": "animals_matched_demo_qwen_123",
                "cache_state": "disabled", "model_cache_state": "disabled", "local_cache_policy": "disabled", "planner_cache_state": "disabled", "online_cache_policy": "disabled",
            },
            "queries": [query(name) for name in ("Q1_zebra_image_count", "Q2_lion_audio_count", "Q3_top_zebra_city", "Q10_top_zebra_city_station")],
        }
        self.assertEqual(animals_failures(data, MODEL), [])
        invalid = copy.deepcopy(data)
        del invalid["queries"][1]["local_inference"]["events"][-1]["inference_started_at"]
        self.assertTrue(animals_failures(invalid, MODEL))


if __name__ == "__main__":
    unittest.main()
