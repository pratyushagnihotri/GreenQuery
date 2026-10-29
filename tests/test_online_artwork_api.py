import json
import tempfile
import unittest
import uuid
from pathlib import Path

from evaluation.measurement.wrapper import build_execution_plan
from evaluation.scripts.artwork_quality import EXPECTED_QUERIES
from evaluation.scripts.validate_online_artwork_run import QUERY_ID, validate
from reasondb.reasoning.llm import LLM_CACHE_DIR, LLMCharacteristics, LargeLanguageModel, Message, Prompt


class _NoCacheLlm(LargeLanguageModel):
    def __init__(self):
        super().__init__(
            "no-cache-test",
            LLMCharacteristics(rpm=1, tpm=1, out_len=1, in_len=1, in_cost=0, out_cost=0),
            cache_enabled=False,
        )

    async def _invoke(self, prompt, stop=None):
        return "ok", 0.1, 0.0


class _NullLogger:
    def debug(self, *_args):
        pass


def _query():
    event = {
        "provider": "openai",
        "model_name": "gpt-4o-mini-2024-07-18",
        "requested_model_name": "gpt-4o-mini",
        "resolved_model_name": "gpt-4o-mini-2024-07-18",
        "runtime_seconds": 1.0,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "prompt_cost_usd": 0.0000015,
        "completion_cost_usd": 0.0000012,
        "total_cost_usd": 0.0000027,
        "cached": False,
        "energy_kwh": 0.001,
        "energy_kwh_min": 0.0005,
        "energy_kwh_max": 0.0015,
        "co2_g": 0.4,
        "co2_g_min": 0.2,
        "co2_g_max": 0.6,
        "energy_source": "estimated_by_ecologits",
        "co2_source": "estimated_by_ecologits",
    }
    planning = {**event, "phase": "planning"}
    execution = {**event, "phase": "execution", "prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}
    phases = {
        "planning": {
            "total_calls": 1, "cached_calls": 0, "total_prompt_tokens": 10, "total_completion_tokens": 2,
            "total_tokens": 12, "total_runtime_seconds": 1.0, "total_cost_usd": 0.0000027,
            "energy_kwh": 0.001, "energy_kwh_min": 0.0005, "energy_kwh_max": 0.0015,
            "co2_g": 0.4, "co2_g_min": 0.2, "co2_g_max": 0.6, "impact_complete": True,
        },
        "execution": {
            "total_calls": 1, "cached_calls": 0, "total_prompt_tokens": 20, "total_completion_tokens": 3,
            "total_tokens": 23, "total_runtime_seconds": 1.0, "total_cost_usd": 0.000003,
            "energy_kwh": 0.002, "energy_kwh_min": 0.001, "energy_kwh_max": 0.003,
            "co2_g": 0.8, "co2_g_min": 0.4, "co2_g_max": 1.2, "impact_complete": True,
        },
        "unattributed": {"total_calls": 0, "total_tokens": 0},
    }
    plan = build_execution_plan(
        [
            '[{"operator":"ImageQaFilter","operator_config":{"__expression__":"{image} depicts Madonna and Child"},"inputs":["artworks"],"output":"filtered","tuning_parameters":{}}]',
            '[{"operator":"TextQaExtract","operator_config":{"__expression__":"Extract the [century] from {inception_text}"},"inputs":["filtered"],"output":"extracted","tuning_parameters":{}}]',
            '[{"logical_plan_step":{"type":"LogicalProject","inputs":["extracted"],"output":"projected","expression":"Keep {title} and {century}"},"available_operators":[{"operator":"Project","operator_config":{"__expression__":"Keep {title} and {century}"},"tuning_parameters":{}}]}]',
        ],
        [
            {"type": "LogicalFilter", "inputs": ["artworks"], "output": "filtered", "expression": "{image} depicts Madonna and Child"},
            {"type": "LogicalExtract", "inputs": ["filtered"], "output": "extracted", "expression": "Extract the [century] from {inception_text}"},
            {"type": "LogicalProject", "inputs": ["extracted"], "output": "projected", "expression": "Keep {title} and {century}"},
        ],
    )
    return {
        "query_name": QUERY_ID,
        "status": "success",
        "model_name": "gpt-4o-mini",
        "energy_kwh": 0.003,
        "co2_g": 1.2,
        "api_cost_usd": 0.0000057,
        "api_calls": {"total_calls": 2, "cached_calls": 0, "total_prompt_tokens": 30, "total_completion_tokens": 5, "total_tokens": 35},
        "energy": {"metric_source": "estimated_by_ecologits"},
        "physical_operators": ["ImageQaFilter", "Project", "TextQaExtract"],
        "result_summary": {
            "expected_query_ids": [QUERY_ID],
            "api_phase_metrics": phases,
            "api_energy_kwh_min": 0.0015,
            "api_energy_kwh_max": 0.0045,
            "api_co2_g_min": 0.6,
            "api_co2_g_max": 1.8,
            "quality": {"status": "evaluated", "metric": "artwork_selection_and_century"},
        },
        "api_events": [planning, execution],
        "execution_plan": plan,
    }


def _measurement():
    return {
        "measurement_id": "artwork_test",
        "config": {
            "backend_profile": "api",
            "model_name": "gpt-4o-mini",
            "planner_model_name": "gpt-4o-mini",
            "online_cache_policy": "disabled",
            "cache_state": "disabled",
            "model_cache_state": "disabled",
            "require_ecologits": True,
            "reasondb_database_name": "online_artwork_test",
        },
        "queries": [_query()],
    }


class OnlineArtworkApiTests(unittest.TestCase):
    def test_disabled_llm_cache_does_not_write(self):
        import asyncio

        marker = uuid.uuid4().hex
        prompt = Prompt(messages=[Message(text=f"cache disabled {marker}", role="user")])
        cache_path = LLM_CACHE_DIR / prompt.sha256()
        if cache_path.exists():
            cache_path.unlink()
        llm = _NoCacheLlm()

        async def run():
            await llm.prepare()
            return await llm.invoke_with_runtime_and_cost(prompt, logger=_NullLogger())

        self.assertEqual(asyncio.run(run())[0], "ok")
        self.assertFalse(cache_path.exists())

    def test_artwork_structured_plan_exports_filter_extract_project(self):
        plan = _query()["execution_plan"]
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(
            [step["operator"] for step in plan["physical"]["steps"]],
            ["ImageQaFilter", "TextQaExtract", "Project"],
        )
        self.assertEqual(len(plan["logical"]["edges"]), 2)
        self.assertEqual(len(plan["edges"]), 3)

    def test_online_artwork_validator_requires_phase_impact_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurement.json"
            path.write_text(json.dumps(_measurement()))
            artifact = validate(path, "gpt-4o-mini")
            self.assertEqual(artifact["query"], EXPECTED_QUERIES[5])
            self.assertEqual(artifact["cached_calls"], 0)
            self.assertGreater(artifact["energy_kwh_min"], 0)
            self.assertTrue(artifact["execution_plan"]["physical"]["steps"])

            invalid = _measurement()
            invalid["queries"][0]["api_events"][0]["co2_g_min"] = None
            path.write_text(json.dumps(invalid))
            with self.assertRaises(AssertionError):
                validate(path, "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
