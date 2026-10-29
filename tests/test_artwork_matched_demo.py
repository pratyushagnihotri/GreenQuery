import copy
import unittest

from evaluation.scripts.artwork_quality import (
    EXPECTED_QUERIES,
    FILTERS,
    century_quality,
    ground_truth,
)
from evaluation.scripts.validate_matched_demo import (
    ARTWORK_LABELS,
    ARTWORK_VISION_MODEL,
    artwork_failures,
)
from evaluation.scripts.prepare_artwork_dataset import bounded_wikimedia_url


MODEL = "Qwen/Qwen2.5-7B-Instruct"
ROLES = {"text": "2", "vision": "5"}


def _query(name):
    execution_roles = {"vision": 0.4}
    events = [
        {"event_kind": "planner", "execution_mode": "local_planner_generation", "model_name": MODEL},
        {"event_kind": "vision", "execution_mode": "direct_no_cache", "model_name": ARTWORK_VISION_MODEL, "inference_batch_size": 8},
    ]
    quality = {
        "status": "evaluated",
        "ground_truth": "reasondb/evaluation/ground_truth/artwork/artwork_no_duplicated.csv",
        "ground_truth_sha256": "a" * 64,
        "metric": "artwork_title_set_precision_recall",
        "label_column": ARTWORK_LABELS.get(name),
        "expected_rows": {
            "A1_madonna_and_child": 9,
            "A2_more_than_two_people": 38,
            "A3_more_than_three_people": 33,
            "A4_saints_with_halos": 33,
            "A5_death_dominant_theme": 16,
        }.get(name),
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    if name == EXPECTED_QUERIES[5]:
        execution_roles = {"vision": 0.3, "text": 0.1}
        events.append({"execution_mode": "direct_no_cache", "model_name": MODEL, "inference_batch_size": 8})
        quality = {
            "status": "evaluated",
            "ground_truth": "reasondb/evaluation/ground_truth/artwork/artwork_no_duplicated.csv",
            "ground_truth_sha256": "a" * 64,
            "metric": "artwork_selection_and_century",
            "selection": {
                "status": "evaluated",
                "metric": "artwork_title_set_precision_recall",
                "label_column": "m&c",
                "expected_rows": 9,
                "correct_rows": 2,
            },
            "century_extraction": {
                "metric": "exact_century_accuracy_on_correctly_selected_rows",
                "label_column": "century",
                "checked_rows": 2,
                "correct_rows": 1,
                "accuracy": 0.5,
                "status": "evaluated",
            },
        }
    return {
        "query_name": name,
        "model_name": MODEL,
        "status": "success",
        "energy_kwh": 1.0,
        "co2_g": 10.0,
        "energy": {"metric_source": "measured_from_nvml_query_window"},
        "local_inference": {
            "answer_cache_hits": 0,
            "kv_cache_hits": 0,
            "planner_request_count": 1,
            "events": events,
        },
        "result_summary": {
            "expected_query_ids": list(EXPECTED_QUERIES),
            "quality": quality,
            "energy_measurement": {"power_samples": [
                {"power_watts_by_gpu": {"2": 100.0, "5": 200.0}},
                {"power_watts_by_gpu": {"2": 100.0, "5": 200.0}},
            ]},
            "phase_energy": {
                "valid": True,
                "provenance": "nvml_per_gpu_role_local_inference_windows_v2",
                "energy_source": "measured_from_nvml_per_gpu_role_windows",
                "co2_source": "estimated_from_configured_emissions_factor",
                "gpu_roles": ROLES,
                "query_energy_kwh_by_gpu": {"2": 0.4, "5": 0.6},
                "planner": {"energy_kwh": 0.2, "co2_g": 2.0, "energy_kwh_by_gpu_role": {"text": 0.2}},
                "execution": {"energy_kwh": 0.4, "co2_g": 4.0, "energy_kwh_by_gpu_role": execution_roles},
                "unattributed_query_energy_kwh": 0.4,
                "unattributed_query_co2_g": 4.0,
            },
        },
    }


def _measurement():
    return {
        "config": {
            "model_name": MODEL,
            "planner_model_name": MODEL,
            "local_text_model_id": MODEL,
            "local_planner_model_id": MODEL,
            "local_vision_model_id": ARTWORK_VISION_MODEL,
            "backend_profile": "dgx-local-text-vision",
            "gpu_count": 2,
            "nvml_gpu_roles": ROLES,
            "run_namespace": "artwork_matched_demo_qwen_123",
            "direct_qa_batch_size": 8,
            "vision_qa_batch_size": 8,
            "cache_state": "disabled",
            "model_cache_state": "disabled",
            "local_cache_policy": "disabled",
            "planner_cache_state": "disabled",
            "online_cache_policy": "disabled",
        },
        "queries": [_query(name) for name in EXPECTED_QUERIES],
    }


class ArtworkMatchedDemoTests(unittest.TestCase):
    def test_wikimedia_urls_skip_redirect_chain(self):
        self.assertEqual(
            bounded_wikimedia_url("http://commons.wikimedia.org/wiki/Special:FilePath/Meister%20von%20Me%C3%9Fkirch.jpg"),
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/Meister_von_Me%C3%9Fkirch.jpg?width=1024",
        )

    def test_query_matrix_and_a6_scoring(self):
        source, truth = ground_truth()
        self.assertEqual(len(EXPECTED_QUERIES), 6)
        self.assertEqual(
            [sum(truth[row["title"]][label] == "1" for row in source) for _, label, _ in FILTERS],
            [9, 38, 33, 33, 16],
        )
        positives = [row for row in source if truth[row["title"]]["m&c"] == "1"]
        negative = next(row for row in source if truth[row["title"]]["m&c"] == "0")
        rows = [
            {"title": positives[0]["title"], "century": truth[positives[0]["title"]]["century"]},
            {"title": positives[1]["title"], "century": "99"},
            {"title": negative["title"], "century": truth[negative["title"]]["century"]},
        ]
        quality = century_quality(rows)
        self.assertEqual(quality["selection"]["correct_rows"], 2)
        self.assertEqual(quality["century_extraction"]["checked_rows"], 2)
        self.assertEqual(quality["century_extraction"]["correct_rows"], 1)
        self.assertEqual(quality["century_extraction"]["accuracy"], 0.5)

    def test_validator_requires_a6_text_and_both_gpu_roles(self):
        data = _measurement()
        self.assertEqual(artwork_failures(data, MODEL), [])

        invalid = copy.deepcopy(data)
        invalid["queries"][-1]["local_inference"]["events"].pop()
        self.assertTrue(any("text extraction" in failure for failure in artwork_failures(invalid, MODEL)))

        invalid = copy.deepcopy(data)
        invalid["queries"][-1]["result_summary"]["phase_energy"]["execution"]["energy_kwh_by_gpu_role"] = {"vision": 0.4}
        self.assertTrue(any("GPU roles" in failure for failure in artwork_failures(invalid, MODEL)))

        invalid = copy.deepcopy(data)
        invalid["queries"][-1]["result_summary"]["quality"]["century_extraction"]["checked_rows"] = 1
        self.assertTrue(any("quality metadata" in failure for failure in artwork_failures(invalid, MODEL)))


if __name__ == "__main__":
    unittest.main()
