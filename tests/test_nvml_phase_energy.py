import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.measurement.nvml import (
    NvidiaSmiEnergySampler,
    build_phase_energy_metrics,
    derive_phase_windows,
)


class PhaseEnergyTests(unittest.TestCase):
    def test_sampler_retains_power_by_physical_gpu(self):
        sampler = NvidiaSmiEnergySampler()

        def sampled(*_args, **_kwargs):
            sampler._stop.set()
            return SimpleNamespace(stdout="2, GPU-text, 100.0\n5, GPU-vision, 200.0\n")

        with patch.dict("os.environ", {"GREENQUERY_GPU_INDEX": "2,5"}), patch(
            "evaluation.measurement.nvml.subprocess.run", side_effect=sampled
        ):
            sampler._sample_loop()

        self.assertEqual(sampler.samples[0][1], {"2": 100.0, "5": 200.0})

    def test_phase_windows_do_not_double_count_or_exceed_query_energy(self):
        events = [
            {"event_kind": "planner", "inference_started_at": 1, "inference_finished_at": 4},
            {"event_kind": "planner", "inference_started_at": 2, "inference_finished_at": 5},
            {"event_kind": "vision", "inference_started_at": 5, "inference_finished_at": 7},
        ]
        windows = derive_phase_windows(events)
        self.assertTrue(windows["valid"])
        self.assertEqual(windows["windows"], {"planner": [(1, 5)], "execution": [(5, 7)]})

        sampler = NvidiaSmiEnergySampler()
        sampler.samples = [(second, 100.0) for second in range(9)]
        total_energy, _ = sampler.energy_over_windows([(0, 8)])
        metrics = build_phase_energy_metrics(events, sampler, total_energy, 0.08, "test_factor")
        phase_energy = metrics["planner"]["energy_kwh"] + metrics["execution"]["energy_kwh"]

        self.assertLessEqual(phase_energy, total_energy)
        self.assertAlmostEqual(
            phase_energy + metrics["unattributed_query_energy_kwh"], total_energy
        )

    def test_medical_roles_conserve_energy_without_cross_role_attribution(self):
        sampler = NvidiaSmiEnergySampler()
        sampler.samples = [
            (second, {"2": 100.0, "5": 200.0}) for second in range(4)
        ]
        total_energy, _ = sampler.energy_over_windows([(0, 3)])
        metrics = build_phase_energy_metrics(
            [
                {
                    "event_kind": "planner",
                    "execution_mode": "local_planner_generation",
                    "inference_started_at": 0,
                    "inference_finished_at": 1,
                },
                {
                    "event_kind": "vision",
                    "execution_mode": "direct_no_cache",
                    "inference_started_at": 1,
                    "inference_finished_at": 2,
                },
            ],
            sampler,
            total_energy,
            total_energy * 344,
            "test_factor",
            {"text": "2", "vision": "5"},
        )

        self.assertTrue(metrics["valid"])
        self.assertEqual(metrics["planner"]["energy_kwh_by_gpu_role"], {"text": 100 / 3_600_000})
        self.assertEqual(
            metrics["execution"]["energy_kwh_by_gpu_role"],
            {"text": 0.0, "vision": 200 / 3_600_000},
        )
        self.assertAlmostEqual(
            metrics["planner"]["energy_kwh"]
            + metrics["execution"]["energy_kwh"]
            + metrics["unattributed_query_energy_kwh"],
            total_energy,
        )

    def test_medical_roles_must_be_distinct_and_present_in_samples(self):
        sampler = NvidiaSmiEnergySampler()
        sampler.samples = [(0, {"2": 100.0}), (1, {"2": 100.0})]
        metrics = build_phase_energy_metrics(
            [
                {"event_kind": "planner", "execution_mode": "local_planner_generation", "inference_started_at": 0, "inference_finished_at": 0.4},
                {"execution_mode": "direct_no_cache", "inference_started_at": 0.5, "inference_finished_at": 1},
            ],
            sampler,
            100 / 3_600_000,
            0.01,
            "test_factor",
            {"text": "2", "vision": "5"},
        )
        self.assertFalse(metrics["valid"])
        self.assertIn("gpu_role_missing_from_nvml_samples", metrics["invalid_reasons"])

    def test_three_gpu_audio_role_is_attributed_without_double_counting(self):
        sampler = NvidiaSmiEnergySampler()
        sampler.samples = [(second, {"2": 100.0, "5": 200.0, "7": 300.0}) for second in range(4)]
        total_energy, _ = sampler.energy_over_windows([(0, 3)])
        metrics = build_phase_energy_metrics(
            [
                {"event_kind": "planner", "execution_mode": "local_planner_generation", "inference_started_at": 0, "inference_finished_at": 1},
                {"event_kind": "audio", "execution_mode": "direct_no_cache", "inference_started_at": 1, "inference_finished_at": 2},
            ],
            sampler,
            total_energy,
            total_energy * 344,
            "test_factor",
            {"text": "2", "vision": "5", "audio": "7"},
        )
        self.assertTrue(metrics["valid"])
        self.assertEqual(metrics["planner"]["energy_kwh_by_gpu_role"], {"text": 100 / 3_600_000})
        self.assertEqual(metrics["execution"]["energy_kwh_by_gpu_role"]["audio"], 300 / 3_600_000)
        self.assertEqual(metrics["execution"]["energy_kwh_by_gpu_role"]["vision"], 0.0)

    def test_deterministic_configuration_needs_no_planner_window(self):
        sampler = NvidiaSmiEnergySampler()
        sampler.samples = [(second, {"2": 100.0, "5": 200.0, "7": 300.0}) for second in range(3)]
        total_energy, _ = sampler.energy_over_windows([(0, 2)])
        metrics = build_phase_energy_metrics(
            [{"event_kind": "audio", "execution_mode": "direct_no_cache", "inference_started_at": 0, "inference_finished_at": 1}],
            sampler, total_energy, total_energy * 344, "test_factor",
            {"text": "2", "vision": "5", "audio": "7"}, planner_required=False,
        )
        self.assertTrue(metrics["valid"])
        self.assertFalse(metrics["planner_required"])
        self.assertEqual(metrics["planner"]["energy_kwh"], 0.0)


if __name__ == "__main__":
    unittest.main()
