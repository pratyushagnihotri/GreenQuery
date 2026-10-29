import copy
import unittest

from evaluation.scripts.validate_matched_demo import rotowire_failures


MODEL = "Qwen/Qwen2.5-7B-Instruct"


def artifact():
    phase = {
        "valid": True,
        "provenance": "nvml_non_overlapping_local_inference_windows_v1",
        "energy_source": "measured_from_nvml_phase_windows",
        "co2_source": "estimated_from_configured_emissions_factor",
        "planner": {"energy_kwh": 0.2, "co2_g": 2.0, "sample_count": 2},
        "execution": {"energy_kwh": 0.3, "co2_g": 3.0, "sample_count": 3},
        "unattributed_query_energy_kwh": 0.5,
        "unattributed_query_co2_g": 5.0,
    }
    query = lambda name: {
        "query_name": name,
        "model_name": MODEL,
        "energy_kwh": 1.0,
        "co2_g": 10.0,
        "energy": {"metric_source": "measured_from_nvml_query_window"},
        "local_inference": {"answer_cache_hits": 0, "kv_cache_hits": 0},
        "result_summary": {"phase_energy": copy.deepcopy(phase), "quality": {"checked_fields": 1}},
    }
    return {
        "config": {
            "model_name": MODEL,
            "planner_model_name": MODEL,
            "reasondb_database_name": "rotowire_matched_demo_qwen_123",
            "cache_state": "disabled",
            "model_cache_state": "disabled",
            "local_cache_policy": "disabled",
            "planner_cache_state": "disabled",
            "online_cache_policy": "disabled",
        },
        "queries": [query("R1_player_points_assists"), query("R2_player_points_home_team_won")],
    }


class RotoWireValidatorTests(unittest.TestCase):
    def test_required_evidence(self):
        valid = artifact()
        self.assertEqual(rotowire_failures(valid, MODEL), [])

        cases = {
            "exact queries": lambda data: data["queries"].append(copy.deepcopy(data["queries"][0])),
            "namespace": lambda data: data["config"].update(reasondb_database_name="rotowire_matched_demo"),
            "cache": lambda data: data["queries"][0]["local_inference"].pop("answer_cache_hits"),
            "provenance": lambda data: data["queries"][0]["result_summary"]["phase_energy"].update(valid=False),
            "samples": lambda data: data["queries"][0]["result_summary"]["phase_energy"]["planner"].update(sample_count=0),
            "energy conservation": lambda data: data["queries"][0]["result_summary"]["phase_energy"].update(unattributed_query_energy_kwh=0.4),
            "CO2 conservation": lambda data: data["queries"][0]["result_summary"]["phase_energy"].update(unattributed_query_co2_g=4.0),
            "quality": lambda data: data["queries"][0]["result_summary"]["quality"].update(checked_fields=0),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                invalid = copy.deepcopy(valid)
                mutate(invalid)
                self.assertTrue(rotowire_failures(invalid, MODEL))


if __name__ == "__main__":
    unittest.main()
