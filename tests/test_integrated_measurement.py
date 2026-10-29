"""Guard the local-plan and API-phase behavior combined by the review merge."""

from evaluation.measurement.api_usage import phase_context, record_openai_call, start_collection, stop_collection
from evaluation.measurement.wrapper import QueryTracker, build_execution_plan


def test_api_phases_are_recorded_without_promoting_unexecuted_semantic_steps():
    token = start_collection()
    try:
        tracker = QueryTracker("Q", "demo", {"require_ecologits": True})
        for phase in ("planning", "execution"):
            with phase_context(phase):
                record_openai_call(
                    model_name="gpt-4.1",
                    runtime_seconds=1.0,
                    prompt_tokens=10,
                    completion_tokens=2,
                    prompt_cost_usd=0.00002,
                    completion_cost_usd=0.000016,
                    energy_kwh=0.001,
                    co2_g=0.4,
                )
        result = tracker.get_result()
    finally:
        stop_collection(token)

    phases = result.result_summary["api_phase_metrics"]
    assert phases["planning"]["total_calls"] == 1
    assert phases["execution"]["total_calls"] == 1
    assert result.result_summary["api_energy_kwh_min"] == 0.002

    plan = build_execution_plan(
        ['[{"logical_plan_step":{"type":"LogicalFilter","inputs":["reviews"],"output":"positive","expression":"review is positive"},"available_operators":[{"operator":"TextQaFilter"}]}]'],
        [{"type": "LogicalFilter", "inputs": ["reviews"], "output": "positive", "expression": "review is positive"}],
    )
    assert plan["physical"]["steps"] == []


if __name__ == "__main__":
    test_api_phases_are_recorded_without_promoting_unexecuted_semantic_steps()
