from contextlib import contextmanager
from pathlib import Path
from reasondb.backends.text_qa import LLMTextQABackend
from reasondb.database.indentifier import RemoteColumn, VirtualTableIdentifier
from reasondb.evaluation.benchmarks.email import EnronEmail
from reasondb.interface.connect import RaccoonDB
from reasondb.operators.aggregate.aggregate import Aggregate
from reasondb.operators.aggregate.groupby import GroupBy
from reasondb.operators.extract.text_qa_extract import TextQaExtract
from reasondb.operators.filter.traditional_filter import TraditionalFilter
from reasondb.operators.join.traditional_join import TraditionalJoin
from reasondb.operators.project.project import Project
from reasondb.optimizer.dependency_graph import DependencyGraph
from reasondb.query_plan.logical_plan import (
    LogicalAggregate,
    LogicalExtract,
    LogicalFilter,
    LogicalGroupBy,
    LogicalJoin,
    LogicalProject,
    LogicalRename,
)
from reasondb.query_plan.physical_operator import PhysicalOperatorsWithPseudos
from reasondb.query_plan.unoptimized_physical_plan import (
    UnoptimizedPhysicalPlan,
    UnoptimizedPhysicalPlanStep,
)
from reasondb.reasoning.llm import GPT4o

PLAN_CONSECUTIVE_EXTRACTS = UnoptimizedPhysicalPlan(
    _plan_steps=[
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("o1"),
                expression="Extract [sender] from {emails.text}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("o1")],
                output=VirtualTableIdentifier("o2"),
                expression="Extract [name] from {o1.sender}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("o2")],
                output=VirtualTableIdentifier("o3"),
                expression="Extract [first_letter] from {o2.name}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
    ]
)

PLAN_MIXED_TRADITIONAL_AND_MM = UnoptimizedPhysicalPlan(
    _plan_steps=[
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("o1"),
                expression="Extract [sender] from {emails.text}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalProject(
                explanation="",
                inputs=[VirtualTableIdentifier("o1")],
                output=VirtualTableIdentifier("o2"),
                expression="Keep {o1.sender, o1.id, o1.text}",
            ),
            operators=PhysicalOperatorsWithPseudos([Project()]),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalFilter(
                explanation="",
                inputs=[VirtualTableIdentifier("o2")],
                output=VirtualTableIdentifier("o3"),
                expression="Keep all {o2.id} > 10",
            ),
            operators=PhysicalOperatorsWithPseudos([TraditionalFilter()]),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
    ]
)

PLAN_INTERLEAVED = UnoptimizedPhysicalPlan(
    _plan_steps=[
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("o1"),
                expression="Extract [sender] from {emails.text}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalFilter(
                explanation="",
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("o2"),
                expression="Only keep all {emails.id} > 10",
            ),
            operators=PhysicalOperatorsWithPseudos([TraditionalFilter()]),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalFilter(
                explanation="",
                inputs=[VirtualTableIdentifier("o1")],
                output=VirtualTableIdentifier("o3"),
                expression="Only keep all {o1.sender} that use gmail",
            ),
            operators=PhysicalOperatorsWithPseudos([TraditionalFilter()]),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
    ]
)


PLAN_PARALLEL_EXTRACTS1 = UnoptimizedPhysicalPlan(
    _plan_steps=[
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("o1"),
                expression="Extract [sender] from {emails.text}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("o1")],
                output=VirtualTableIdentifier("o2"),
                expression="Extract [name] from {o1.sender}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("o2")],
                output=VirtualTableIdentifier("o3"),
                expression="Extract [domain] from {o2.sender}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
    ]
)

PLAN_PARALLEL_EXTRACTS2 = UnoptimizedPhysicalPlan(
    _plan_steps=[
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("o1"),
                expression="Extract [sender] from {emails.text}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("o1")],
                output=VirtualTableIdentifier("o2"),
                expression="Extract [name] from {o1.sender}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("o2")],
                output=VirtualTableIdentifier("o3"),
                expression="Extract [recipient] from {o2.text}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
    ]
)

PLAN_JOIN = UnoptimizedPhysicalPlan(
    _plan_steps=[
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("o1"),
                expression="Extract [sender] from {emails.text}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("o2"),
                expression="Extract [recipient] from {emails.text}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalJoin(
                explanation="",
                inputs=[VirtualTableIdentifier("o1"), VirtualTableIdentifier("o2")],
                output=VirtualTableIdentifier("o3"),
                expression="Match {o1.sender} == {o2.recipient}",
            ),
            operators=PhysicalOperatorsWithPseudos([TraditionalJoin()]),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
    ]
)

PLAN_RENAME = UnoptimizedPhysicalPlan(
    _plan_steps=[
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("o1"),
                expression="Extract [sender] from {emails.text}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalRename(
                explanation="",
                inputs=[VirtualTableIdentifier("o1")],
                output=VirtualTableIdentifier("o2"),
                expression="Rename {o1.text} to [email]",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("o2")],
                output=VirtualTableIdentifier("o3"),
                expression="Extract [recipient] from {o2.email}",
            ),
            operators=PhysicalOperatorsWithPseudos([TraditionalJoin()]),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
    ]
)

PLAN_AGGREGATE = UnoptimizedPhysicalPlan(
    _plan_steps=[
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="",
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("o1"),
                expression="Extract [sender] from {emails.text}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalGroupBy(
                explanation="Group by sender",
                inputs=[VirtualTableIdentifier("o1")],
                output=VirtualTableIdentifier("o2"),
                expression="Group by {o1.sender}",
            ),
            operators=PhysicalOperatorsWithPseudos([GroupBy()]),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalAggregate(
                explanation="Count emails per sender",
                inputs=[VirtualTableIdentifier("o2")],
                output=VirtualTableIdentifier("o3"),
                expression="Count number of emails",
            ),
            operators=PhysicalOperatorsWithPseudos([Aggregate()]),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="Extract sender from emails",
                inputs=[VirtualTableIdentifier("o3")],
                output=VirtualTableIdentifier("o4"),
                expression="Extract [recipient] from {o3.text}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
        UnoptimizedPhysicalPlanStep(
            logical_plan_step=LogicalExtract(
                explanation="Extract subject from emails",
                inputs=[VirtualTableIdentifier("o4")],
                output=VirtualTableIdentifier("o5"),
                expression="Extract [subject] from {o4.text}",
            ),
            operators=PhysicalOperatorsWithPseudos(
                [TextQaExtract(LLMTextQABackend(GPT4o()))]
            ),
            llm_configurations={},
            estimated_best_operator_idx=0,
        ),
    ]
)


@contextmanager
def setup():
    path = Path("palimpzest/testdata/enron-eval")
    csv_path = EnronEmail.load_email_table(path)
    with RaccoonDB("email") as rc:
        rc.add_table(
            path=csv_path,
            table_name="emails",
            text_columns=[RemoteColumn("emails.text_path", "emails.text")],
        )
        rc.prepare()
        yield rc


def test_from_unoptimized_physical_plan():
    with setup() as rc:
        result1 = DependencyGraph.from_unoptimized_physical_plan(
            unoptimized_physical_plan=PLAN_CONSECUTIVE_EXTRACTS,
            database=rc.database,
        )
        assert result1.dependencies == [set(), {0}, {1}]
        assert result1.nodes == PLAN_CONSECUTIVE_EXTRACTS._plan_steps
        assert result1.database is rc.database
        assert result1.unoptimized_physical_plan is PLAN_CONSECUTIVE_EXTRACTS

        result2 = DependencyGraph.from_unoptimized_physical_plan(
            unoptimized_physical_plan=PLAN_PARALLEL_EXTRACTS1,
            database=rc.database,
        )
        assert result2.dependencies == [set(), {0}, {0}]

        result3 = DependencyGraph.from_unoptimized_physical_plan(
            unoptimized_physical_plan=PLAN_PARALLEL_EXTRACTS2,
            database=rc.database,
        )
        assert result3.dependencies == [set(), {0}, set()]

        result4 = DependencyGraph.from_unoptimized_physical_plan(
            unoptimized_physical_plan=PLAN_JOIN,
            database=rc.database,
        )
        assert result4.dependencies == [set(), set(), {0, 1}]

        result5 = DependencyGraph.from_unoptimized_physical_plan(
            unoptimized_physical_plan=PLAN_RENAME,
            database=rc.database,
        )
        assert result5.dependencies == [set(), {0}, {1}]

        result6 = DependencyGraph.from_unoptimized_physical_plan(
            unoptimized_physical_plan=PLAN_AGGREGATE,
            database=rc.database,
        )
        assert result6.dependencies == [set(), {0}, {1}, {2}, {2}]


def test_pullup_required():
    with setup() as rc:
        graph = DependencyGraph.from_unoptimized_physical_plan(
            unoptimized_physical_plan=PLAN_CONSECUTIVE_EXTRACTS,
            database=rc.database,
        )
        result1 = graph.get_pullup_required()
        assert result1 == [True, True, True]

        graph = DependencyGraph.from_unoptimized_physical_plan(
            unoptimized_physical_plan=PLAN_MIXED_TRADITIONAL_AND_MM,
            database=rc.database,
        )
        result2 = graph.get_pullup_required()
        assert result2 == [True, True, False]

        graph = DependencyGraph.from_unoptimized_physical_plan(
            unoptimized_physical_plan=PLAN_INTERLEAVED,
            database=rc.database,
        )
        result2 = graph.get_pullup_required()
        assert result2 == [True, False, True]


def test_pullup():
    with setup() as rc:
        graph = DependencyGraph.from_unoptimized_physical_plan(
            unoptimized_physical_plan=PLAN_CONSECUTIVE_EXTRACTS,
            database=rc.database,
        )
        result1 = graph.pullup(rc.database)
        assert [type(x.logical_plan_step) for x in result1.plan_steps] == [
            LogicalExtract,
            LogicalExtract,
            LogicalExtract,
        ]
        assert [x.inputs for x in result1.plan_steps] == [
            [VirtualTableIdentifier("emails")],
            [VirtualTableIdentifier("o1")],
            [VirtualTableIdentifier("o2")],
        ]
        assert [x.output for x in result1.plan_steps] == [
            VirtualTableIdentifier("o1"),
            VirtualTableIdentifier("o2"),
            VirtualTableIdentifier("o3"),
        ]



def run_all():
    test_from_unoptimized_physical_plan()
    test_pullup_required()
    test_pullup()


if __name__ == "__main__":
    run_all()
