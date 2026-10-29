import copy
import unittest

from evaluation.scripts.validate_matched_demo import medical_phase_failures


ROLES = {"text": "2", "vision": "5"}


def query(name, execution_role):
    total_energy = 1.0
    phase = {
        "valid": True,
        "provenance": "nvml_per_gpu_role_local_inference_windows_v2",
        "energy_source": "measured_from_nvml_per_gpu_role_windows",
        "co2_source": "estimated_from_configured_emissions_factor",
        "gpu_roles": ROLES,
        "query_energy_kwh_by_gpu": {"2": 0.4, "5": 0.6},
        "planner": {
            "energy_kwh": 0.2,
            "co2_g": 2.0,
            "energy_kwh_by_gpu_role": {"text": 0.2},
        },
        "execution": {
            "energy_kwh": 0.3,
            "co2_g": 3.0,
            "energy_kwh_by_gpu_role": {
                "text": 0.3 if execution_role == "text" else 0.0,
                "vision": 0.3 if execution_role == "vision" else 0.0,
            },
        },
        "unattributed_query_energy_kwh": 0.5,
        "unattributed_query_co2_g": 5.0,
    }
    return {
        "query_name": name,
        "energy_kwh": total_energy,
        "co2_g": 10.0,
        "result_summary": {
            "phase_energy": phase,
            "energy_measurement": {
                "power_samples": [
                    {"power_watts_by_gpu": {"2": 100.0, "5": 200.0}},
                    {"power_watts_by_gpu": {"2": 100.0, "5": 200.0}},
                ]
            },
        },
    }


class MedicalPhaseValidatorTests(unittest.TestCase):
    def test_role_evidence_and_conservation(self):
        data = {
            "config": {"nvml_gpu_roles": ROLES, "gpu_count": 2},
            "queries": [
                query("Q1_allergy_patients", "text"),
                query("Q9_sick_patients_skin_xray", "vision"),
            ],
        }
        self.assertEqual(medical_phase_failures(data), [])

        invalid = copy.deepcopy(data)
        invalid["queries"][1]["result_summary"]["phase_energy"]["execution"]["energy_kwh_by_gpu_role"]["text"] = 0.1
        self.assertTrue(medical_phase_failures(invalid))

        invalid = copy.deepcopy(data)
        invalid["queries"][0]["result_summary"]["phase_energy"]["unattributed_query_energy_kwh"] = 0.4
        self.assertTrue(medical_phase_failures(invalid))


if __name__ == "__main__":
    unittest.main()
