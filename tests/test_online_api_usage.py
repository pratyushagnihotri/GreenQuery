from evaluation.measurement.api_usage import (
    events_since,
    phase_context,
    record_openai_call,
    snapshot_index,
    start_collection,
    stop_collection,
    summarize,
)
from evaluation.measurement.wrapper import build_execution_plan
from reasondb.reasoning.llm import get_openai_model_characteristics
from reasondb.database.indentifier import DataType

import pandas as pd


def test_api_phase_context_and_ranges_do_not_leak():
    token = start_collection()
    try:
        start = snapshot_index()
        with phase_context("planning"):
            record_openai_call(
                model_name="gpt-4.1-2025-04-14",
                requested_model_name="gpt-4.1",
                resolved_model_name="gpt-4.1-2025-04-14",
                runtime_seconds=1.0,
                prompt_tokens=10,
                completion_tokens=2,
                prompt_cost_usd=0.00002,
                completion_cost_usd=0.000016,
                energy_kwh=2.0,
                energy_kwh_min=1.0,
                energy_kwh_max=3.0,
                co2_g=4.0,
                co2_g_min=3.0,
                co2_g_max=5.0,
            )
        with phase_context("execution"):
            record_openai_call(
                model_name="gpt-4.1-2025-04-14",
                runtime_seconds=2.0,
                prompt_tokens=20,
                completion_tokens=4,
                prompt_cost_usd=0.00004,
                completion_cost_usd=0.000032,
                energy_kwh=5.0,
                co2_g=8.0,
            )
        summary = summarize(events_since(start))
    finally:
        stop_collection(token)
    assert summary["phase_metrics"]["planning"]["total_tokens"] == 12
    assert summary["phase_metrics"]["execution"]["total_tokens"] == 24
    assert summary["energy_kwh_min"] == 6.0
    assert summary["energy_kwh_max"] == 8.0
    assert summary["phase_metrics"]["unattributed"]["total_calls"] == 0


def test_gpt_41_has_versioned_pricing():
    characteristics = get_openai_model_characteristics("gpt-4.1-2025-04-14")
    assert characteristics is not None
    assert characteristics.in_cost == 2.0
    assert characteristics.out_cost == 8.0


def test_pandas_string_dtype_is_a_reasondb_string():
    dtype = pd.Series(["review"], dtype="str").dtype
    assert DataType.from_pandas(dtype, False, False, False, ["review"]) == DataType.STRING


def test_execution_plan_preserves_logical_physical_steps_and_edges():
    plan = build_execution_plan([
        '[{"operator":"TextQaFilter","operator_config":{"__expression__":"{reviews.reviewtext} is positive"},"inputs":["reviews"],"output":"positive","tuning_parameters":{}}]',
    ], [{"type":"LogicalFilter","inputs":["reviews"],"output":"positive","expression":"{reviews.reviewtext} is positive","explanation":"filter"}])
    assert plan["schema_version"] == 1
    assert plan["logical"]["steps"][0]["expression"] == "{reviews.reviewtext} is positive"
    assert plan["physical"]["steps"][0]["operator"] == "TextQaFilter"
    assert plan["edges"] == [{"from": "logical_0", "to": "physical_0", "kind": "implementation"}]


def test_execution_plan_records_one_option_database_steps():
    plan = build_execution_plan([
        '[{"logical_plan_step":{"type":"LogicalRename","inputs":["reviews"],"output":"other","expression":"Rename {reviews.text} to [other]"},"available_operators":[{"operator":"Rename","operator_config":{"__expression__":"Rename {reviews.text} to [other]"},"tuning_parameters":{}}]}]',
    ], [{"type":"LogicalRename","inputs":["reviews"],"output":"other","expression":"Rename {reviews.text} to [other]","explanation":"rename"}])
    step = plan["physical"]["steps"][0]
    assert step["operator"] == "Rename"
    assert step["selection_source"] == "single_available_operator"
    assert plan["edges"] == [{"from": "logical_0", "to": "physical_0", "kind": "implementation"}]
