import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock

from reasondb.backends.audio_qa import AudioQaBackend
from reasondb.database.indentifier import DataType, VirtualColumnIdentifier, VirtualTableIdentifier
from reasondb.database.metadata import DatabaseMetadata
from reasondb.operators.aggregate.aggregate import Aggregate
from reasondb.operators.aggregate.groupby import GroupBy
from reasondb.operators.filter.audio_qa_filter import AudioQaFilter
from reasondb.operators.filter.image_qa_filter import ImageQaFilter
from reasondb.operators.filter.text_qa_filter import TextQaFilter
from reasondb.operators.project.project import Project
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.query_plan.logical_plan import LogicalAggregate, LogicalFilter, LogicalGroupBy, LogicalProject
from reasondb.query_plan.physical_operator import PhysicalOperatorToolbox
from reasondb.reasoning.exceptions import Mistake
from reasondb.reasoning.llm import Message, Prompt


class ImageShortcutTests(unittest.TestCase):
    def test_fix_prompt_keeps_model_template_text_literal(self):
        configurator = PlanConfigurator(MagicMock(), MagicMock())
        fixed = configurator.generate_fix_prompt(
            Mistake("bad configuration"),
            '{"question_template": "Is {{taken_3_reviews.reviewtext}} positive?"}',
            Prompt([Message(role="user", text="Configure the step")]),
        )

        self.assertIn("{{taken_3_reviews.reviewtext}}", fixed.messages[1].only_text)
        self.assertIn("bad configuration", fixed.messages[2].only_text)

    def test_single_table_template_shorthand_is_normalized(self):
        interface = TextQaFilter(MagicMock(), quality=1, fake_cost=0).get_llm_parameters()
        parsed = interface.parse_config(
            [VirtualTableIdentifier("reviews")],
            {
                "question_template": "Is {reviewtext} clearly positive?",
                "context": "reviews.reviewtext",
                "keep_answer": "yes",
            },
        )
        self.assertEqual(str(parsed["question_template"]), "Is {reviews.reviewtext} clearly positive?")

    def test_aggregate_rejects_mismatched_parameter_lists_for_planner_retry(self):
        aggregate = Aggregate(quality=1, fake_cost=0)

        with self.assertRaisesRegex(Mistake, "must have the same length"):
            asyncio.run(aggregate.get_observation(
                MagicMock(), [], MagicMock(), [], {
                    "aggregate_columns": [
                        VirtualColumnIdentifier("animals.city"),
                        VirtualColumnIdentifier("animals.stationid"),
                    ],
                    "aggregation_functions": ["count"],
                    "output_column_names": ["count"],
                }, [], MagicMock(), MagicMock(),
            ))

    def test_animals_q2_routes_attached_audio_metadata_to_audio_filter(self):
        database = MagicMock()
        metadata = DatabaseMetadata(database, AsyncMock(), AsyncMock())
        asyncio.run(metadata.prepare(["animals_audio"], first_time=False))
        audio_view_sql = next(
            call.args[0]
            for call in database.sql.call_args_list
            if "VIEW __audio_columns__" in call.args[0]
        )
        self.assertIn("animals_audio.__audio_columns__", audio_view_sql)

        backend = MagicMock(spec=AudioQaBackend, returns_log_odds=True)
        audio_filter = AudioQaFilter(backend, quality=10, fake_cost=0)
        toolbox = PhysicalOperatorToolbox(
            join_operators=[], join_predicates=[], filter_operators=[audio_filter],
            extract_operators=[], transform_operators=[], limit_operators=[],
            project_operators=[], sorting_operators=[], groupby_operators=[],
            aggregate_operators=[], rename_operators=[],
        )
        table = VirtualTableIdentifier("animals_audio")
        step = LogicalFilter(
            "Filter lion sounds", [table], VirtualTableIdentifier("lion_audio"),
            "{animals_audio.audio} contains the sound of a lion",
        )
        state = MagicMock()
        state.get_data_type.return_value = DataType.AUDIO
        self.assertEqual(toolbox.get_options(step, state, MagicMock()).operators, [audio_filter])

    def test_image_shortcut_has_a_defined_prompt_for_the_common_return_path(self):
        column = SimpleNamespace(column_name="xray_picture")
        step = SimpleNamespace(
            expression="{xray_picture} shows lung problems.",
            get_input_columns=lambda: [column],
            inputs=[],
        )
        toolbox = MagicMock()
        toolbox.get_options.return_value = SimpleNamespace(
            operators=[object.__new__(ImageQaFilter)]
        )
        configurator = PlanConfigurator(MagicMock(), toolbox)
        physical_step = MagicMock(chosen_operator_idx=0)
        physical_step.get_observation_for_operator = AsyncMock(return_value=MagicMock())
        configurator.map_logical_to_physical = MagicMock(return_value=physical_step)
        database_state = MagicMock()
        database_state.get_data_type.return_value = DataType.IMAGE
        state = MagicMock()
        logger = MagicMock()
        previous_prompt = MagicMock()

        prompt, response, mistake = asyncio.run(
            configurator.llm_configure_single_step(
                0, state, database_state, None, previous_prompt, None, None, step, logger
            )
        )

        self.assertIs(prompt, previous_prompt)
        self.assertIn('"ImageQaFilter"', response)
        self.assertIsNone(mistake)

    def test_audio_shortcut_configures_a_contains_filter_without_the_planner(self):
        column = SimpleNamespace(column_name="audio")
        step = SimpleNamespace(
            expression="{animals_audio.audio} contains the sound of a lion.",
            get_input_columns=lambda: [column],
            inputs=[],
            output=MagicMock(),
        )
        toolbox = MagicMock()
        toolbox.get_options.return_value = SimpleNamespace(
            operators=[object.__new__(AudioQaFilter)]
        )
        configurator = PlanConfigurator(MagicMock(), toolbox)
        physical_step = MagicMock(chosen_operator_idx=0)
        physical_step.get_observation_for_operator = AsyncMock(return_value=MagicMock())
        configurator.map_logical_to_physical = MagicMock(return_value=physical_step)
        database_state = MagicMock()
        database_state.get_data_type.return_value = DataType.AUDIO

        _, response, mistake = asyncio.run(
            configurator.llm_configure_single_step(
                0, MagicMock(), database_state, None, None, None, None, step, MagicMock()
            )
        )

        self.assertIn('"AudioQaFilter"', response)
        self.assertIn("Does the audio contain the sound of a lion?", response)
        self.assertIsNone(mistake)

    def test_malformed_planner_configuration_is_retryable(self):
        step = SimpleNamespace(
            expression="{age} is greater than 50.",
            inputs=[],
            output=MagicMock(),
            get_input_columns=lambda: [],
        )
        toolbox = MagicMock()
        toolbox.get_options.return_value = SimpleNamespace(operators=[])
        configurator = PlanConfigurator(MagicMock(), toolbox)
        configurator.generate_prompt = AsyncMock(return_value=MagicMock())
        configurator.llm.invoke = AsyncMock(return_value="bad configuration")
        configurator.map_logical_to_physical = MagicMock(side_effect=AssertionError("bad column"))

        _, _, mistake = asyncio.run(
            configurator.llm_configure_single_step(
                0, MagicMock(), MagicMock(), None, None, None, None, step, MagicMock()
            )
        )

        self.assertIsInstance(mistake, Mistake)

    def test_project_uses_columns_from_the_logical_plan(self):
        table = VirtualTableIdentifier("patient")
        step = LogicalProject(
            "", [table], VirtualTableIdentifier("result"), "Keep {patient.patient_id}"
        )
        toolbox = MagicMock()
        toolbox.get_options.return_value = SimpleNamespace(operators=[object.__new__(Project)])
        configurator = PlanConfigurator(MagicMock(), toolbox)
        physical_step = MagicMock(chosen_operator_idx=0)
        physical_step.get_observation_for_operator = AsyncMock(return_value=MagicMock())
        configurator.map_logical_to_physical = MagicMock(return_value=physical_step)

        _, response, mistake = asyncio.run(
            configurator.llm_configure_single_step(
                0, MagicMock(), MagicMock(), None, None, None, None, step, MagicMock()
            )
        )

        self.assertIn('"keep_columns": ["patient.patient_id"]', response)
        self.assertIsNone(mistake)

    def test_groupby_uses_columns_from_the_logical_plan(self):
        table = VirtualTableIdentifier("animals")
        step = LogicalGroupBy(
            "", [table], VirtualTableIdentifier("grouped"), "Group by {animals.city}"
        )
        toolbox = MagicMock()
        toolbox.get_options.return_value = SimpleNamespace(operators=[object.__new__(GroupBy)])
        configurator = PlanConfigurator(MagicMock(), toolbox)
        physical_step = MagicMock(chosen_operator_idx=0)
        physical_step.get_observation_for_operator = AsyncMock(return_value=MagicMock())
        configurator.map_logical_to_physical = MagicMock(return_value=physical_step)

        _, response, mistake = asyncio.run(
            configurator.llm_configure_single_step(
                0, MagicMock(), MagicMock(), None, None, None, None, step, MagicMock()
            )
        )

        self.assertIn('"group_columns": ["animals.city"]', response)
        self.assertIsNone(mistake)

    def test_average_aggregate_uses_the_logical_plan(self):
        table = VirtualTableIdentifier("patient")
        step = LogicalAggregate(
            "", [table], VirtualTableIdentifier("result"),
            "Average of {patient.age}[average_age_acne_patients]",
        )
        toolbox = MagicMock()
        toolbox.get_options.return_value = SimpleNamespace(operators=[object.__new__(Aggregate)])
        configurator = PlanConfigurator(MagicMock(), toolbox)
        physical_step = MagicMock(chosen_operator_idx=0)
        physical_step.get_observation_for_operator = AsyncMock(return_value=MagicMock())
        configurator.map_logical_to_physical = MagicMock(return_value=physical_step)

        _, response, mistake = asyncio.run(
            configurator.llm_configure_single_step(
                0, MagicMock(), MagicMock(), None, None, None, None, step, MagicMock()
            )
        )

        self.assertIn('"aggregation_functions": ["avg"]', response)
        self.assertIn('"output_column_names": ["average_age_acne_patients"]', response)
        self.assertIsNone(mistake)

    def test_count_aggregate_uses_count_star(self):
        table = VirtualTableIdentifier("animals")
        step = LogicalAggregate(
            "", [table], VirtualTableIdentifier("result"), "Count rows as [count]"
        )
        toolbox = MagicMock()
        toolbox.get_options.return_value = SimpleNamespace(operators=[Aggregate(quality=1, fake_cost=0)])
        configurator = PlanConfigurator(MagicMock(), toolbox)
        physical_step = MagicMock(chosen_operator_idx=0)
        physical_step.get_observation_for_operator = AsyncMock(return_value=MagicMock())
        configurator.map_logical_to_physical = MagicMock(return_value=physical_step)

        _, response, mistake = asyncio.run(
            configurator.llm_configure_single_step(
                0, MagicMock(), MagicMock(), None, None, None, None, step, MagicMock()
            )
        )

        self.assertIn('"aggregate_columns": ["*"]', response)
        self.assertIn('"aggregation_functions": ["count"]', response)
        self.assertIn('"output_column_names": ["count"]', response)
        self.assertIsNone(mistake)


if __name__ == "__main__":
    unittest.main()
