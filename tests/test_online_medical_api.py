import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from evaluation.scripts.build_online_medical_query_metrics import main as build_metrics
from evaluation.scripts.run_medical_measurement import _database_name
from evaluation.scripts.validate_online_medical_run import validate
from reasondb.reasoning.llm import (
    ConfigurableOpenAIModel,
    Message,
    Prompt,
)
from reasondb.backends.vision_model import LlmVisionModel


MODEL = "gpt-4o-mini"
QUERIES = (
    "Q1_allergy_patients",
    "Q3_cancer_family_lung_problems_patients",
    "Q4_average_age_acne_patients",
    "Q9_sick_patients_skin_xray",
)


def _query(name):
    phase = {
        key: {
            "total_calls": 1,
            "cached_calls": 0,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 2,
            "total_tokens": 12,
            "total_cost_usd": 0.00001,
            "total_runtime_seconds": 0.1,
            "energy_kwh": 0.000001,
            "co2_g": 0.001,
            "impact_complete": True,
        }
        for key in ("planning", "execution")
    }
    phase["unattributed"] = {"total_calls": 0}
    metric = (
        "exact_aggregate_average_age"
        if name == "Q4_average_age_acne_patients"
        else "patient_id_set_precision_recall"
    )
    return {
        "query_name": name,
        "model_name": MODEL,
        "execution_time_seconds": 1.0,
        "total_tokens": 24,
        "api_cost_usd": 0.00002,
        "energy_kwh": 0.000002,
        "co2_g": 0.002,
        "api_calls": {
            "total_calls": 2,
            "cached_calls": 0,
            "total_prompt_tokens": 20,
            "total_completion_tokens": 4,
            "total_tokens": 24,
        },
        "energy": {"metric_source": "estimated_by_ecologits"},
        "metric_sources": {"co2": "estimated_by_ecologits"},
        "physical_operators": ["TextQaFilter-LLMTextQABackend-gpt-4o-mini"],
        "operator_plan": ['{"operator": "TextQaFilter-LLMTextQABackend-gpt-4o-mini"}'],
        "result_summary": {"quality": {"status": "evaluated", "metric": metric}, "api_phase_metrics": phase},
        "api_events": [
            {
                "phase": key,
                "resolved_model_name": MODEL,
                "requested_model_name": MODEL,
                "energy_kwh": 0.000001,
                "co2_g": 0.001,
            }
            for key in ("planning", "execution")
        ],
        "status": "success",
    }


def _measurement():
    return {
        "config": {
            "backend_profile": "api",
            "model_name": MODEL,
            "planner_model_name": MODEL,
            "online_cache_policy": "disabled",
            "cache_state": "disabled",
            "model_cache_state": "disabled",
            "require_ecologits": True,
            "reasondb_database_name": "online_medical_gpt_4o_mini_01",
        },
        "queries": [_query(name) for name in QUERIES],
    }


class OnlineMedicalApiTests(unittest.TestCase):
    def test_api_vision_prompt_embeds_image_and_cache_policy_is_honored(self):
        import os

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "one.jpg"
            Image.new("RGB", (1, 1), "white").save(image)
            prompt = Prompt([Message("Is this visible?", "user", image=image)])
            content = prompt.prompt_messages[0]["content"]
            self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test"
        try:
            self.assertFalse(LlmVisionModel(ConfigurableOpenAIModel(MODEL, cache_enabled=False)).cache_enabled)
            self.assertTrue(LlmVisionModel(ConfigurableOpenAIModel(MODEL, cache_enabled=True)).cache_enabled)
        finally:
            if old_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_medical_namespace_rejects_missing_or_unsafe_env(self):
        import os

        old = os.environ.pop("GREENQUERY_RUN_NAMESPACE", None)
        try:
            with self.assertRaises(ValueError):
                _database_name()
            os.environ["GREENQUERY_RUN_NAMESPACE"] = "bad-name"
            with self.assertRaises(ValueError):
                _database_name()
            os.environ["GREENQUERY_RUN_NAMESPACE"] = "online_medical_gpt_4o_mini_01"
            self.assertEqual(_database_name(), "online_medical_gpt_4o_mini_01")
        finally:
            if old is None:
                os.environ.pop("GREENQUERY_RUN_NAMESPACE", None)
            else:
                os.environ["GREENQUERY_RUN_NAMESPACE"] = old

    def test_validator_and_compact_metrics_accept_strict_medical_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / MODEL / "run_01"
            run_dir.mkdir(parents=True)
            measurement = run_dir / "medical_20260902_000000.json"
            measurement.write_text(json.dumps(_measurement()))
            validate(measurement, MODEL)

            output = root / "medical_query_metrics.json"
            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["build", str(root), str(output)]
                build_metrics()
            finally:
                sys.argv = old_argv
            records = json.loads(output.read_text())
            self.assertEqual(len(records), 4)
            self.assertEqual(records[0]["demo"], "medical")
            self.assertEqual(records[0]["planning"]["energy_kwh"], 0.000001)


if __name__ == "__main__":
    unittest.main()
