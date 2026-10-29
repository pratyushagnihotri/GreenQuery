from evaluation.measurement.metrics import APIMetrics, CostMetrics, EnergyMetrics, QueryResult, RuntimeMetrics
from evaluation.measurement.wrapper import build_execution_plan


def test_execution_plan_records_one_option_database_steps():
    plan = build_execution_plan(
        [
            '[{"logical_plan_step":{"type":"LogicalRename","inputs":["reviews"],"output":"renamed","expression":"Rename {reviews.reviewtext} to [reviewtext_other]"},"available_operators":[{"operator":"Rename","operator_config":{"__expression__":"Rename {reviews.reviewtext} to [reviewtext_other]"},"tuning_parameters":{}}]}]',
            '[{"logical_plan_step":{"type":"LogicalJoin","inputs":["reviews","renamed"],"output":"joined","expression":"We need to do a cartesian product - i.e. use a join condition that is always true"},"available_operators":[{"operator":"TraditionalJoin","operator_config":{"join_condition":"1 = 1","join_type":"inner","__expression__":"We need to do a cartesian product - i.e. use a join condition that is always true"},"tuning_parameters":{}}]}]',
        ],
        [
            {"type": "LogicalRename", "inputs": ["reviews"], "output": "renamed", "expression": "Rename {reviews.reviewtext} to [reviewtext_other]", "explanation": "rename"},
            {"type": "LogicalJoin", "inputs": ["reviews", "renamed"], "output": "joined", "expression": "We need to do a cartesian product - i.e. use a join condition that is always true", "explanation": "join"},
        ],
    )

    physical = plan["physical"]["steps"]
    assert [step["operator"] for step in physical] == ["Rename", "TraditionalJoin"]
    assert {step["selection_source"] for step in physical} == {"single_available_operator"}
    assert {edge["from"] for edge in plan["edges"]} == {step["id"] for step in plan["logical"]["steps"]}


def test_query_result_serializes_execution_plan():
    result = QueryResult(
        query_name="Q",
        query_type="demo",
        runtime=RuntimeMetrics(0, 1, 1, "Q"),
        api=APIMetrics(0, [], 0, 0, 0),
        energy=EnergyMetrics(0, 0, 0, 0),
        cost=CostMetrics(0, 0, 0),
        result_summary={"rows_returned": 0},
        execution_plan={"schema_version": 1},
    )

    assert result.to_dict()["execution_plan"] == {"schema_version": 1}


def test_execution_plan_does_not_promote_one_option_semantic_steps():
    plan = build_execution_plan(
        [
            '[{"logical_plan_step":{"type":"LogicalFilter","inputs":["reviews"],"output":"positive","expression":"{reviews.reviewtext} is positive"},"available_operators":[{"operator":"TextQaFilter","operator_config":{"__expression__":"{reviews.reviewtext} is positive"},"tuning_parameters":{}}]}]',
        ],
        [
            {"type": "LogicalFilter", "inputs": ["reviews"], "output": "positive", "expression": "{reviews.reviewtext} is positive", "explanation": "filter"},
        ],
    )

    assert plan["physical"]["steps"] == []
    assert plan["edges"] == []
