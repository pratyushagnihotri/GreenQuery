"""ApeTextQaFilter — join predicate using APE (Adaptive Parallel Encoding).

Both the left and right contexts are provided as pre-built KV caches.
The question is a static string that references neither document by value.
The operator calls /text_qa_ape_join which computes the full N×M cartesian product.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Type
import pandas as pd
import torch
from torch._prims_common import Tensor
from reasondb.backends.text_qa import TextQaBackend
from reasondb.database.indentifier import (
    DataType,
    DataTypes,
    HiddenColumnType,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.database import Database
from reasondb.database.intermediate_state import IntermediateState
from reasondb.evaluation.benchmark import LabelsDefinition
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import (
    FROM_LOGICAL_PLAN,
    LLMParameter,
    LLMParameterColumnDtype,
    LLMParameterTemplateDtype,
    LLMParameterValueDtype,
    LlmParameterTemplate,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import LogicalFilter, LogicalPlanStep
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    ProfilingCost,
    RunOutsideResult,
)
from reasondb.query_plan.tuning_parameters import (
    TuningParameter,
    TuningParameterContinuous,
)
from reasondb.reasoning.observation import (
    FilterOnComputedDataObservation,
    Observation,
    ThresholdFilterOnComputedDataObservation,
)
from reasondb.reasoning.llm import Message, Prompt, PromptTemplate
from reasondb.utils.logging import FileLogger

DEFAULT_THRESHOLD_LOWER = 0.0
DEFAULT_THRESHOLD_UPPER = 0.0
INITIAL_THRESHOLD_LOWER = -1.0
INITIAL_THRESHOLD_UPPER = 1.0
THRESHOLD_RANGE = (-10.0, 10.0)


class ApeTextQaFilter(PhysicalOperator):
    """Join predicate that uses APE: both left and right contexts are in KV caches.

    The model attends to both cached documents non-causally (temperature-scaled) and
    then answers a static question about the pair. This operator always runs in join
    mode — it is only useful as a join_predicate in the configurator.
    """

    def __init__(
        self,
        text_qa_backend: TextQaBackend,
        quality: float,
        fake_cost: float,
        temperature: float = 0.9,
        scale: float = 0.9,
        max_batch_size: int = 8,
    ):
        self.text_qa_backend = text_qa_backend
        self.temperature = temperature
        self.scale = scale
        self.max_batch_size = max_batch_size
        super().__init__(quality=quality, fake_cost=fake_cost)

    async def get_observation(
        self,
        database_state: IntermediateState,
        inputs: Sequence[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        output_columns: Sequence[VirtualColumnIdentifier],
        llm_parameters: Dict[str, Any],
        data_sample: Sequence[Optional[pd.DataFrame]],
        logical_plan_step: LogicalPlanStep,
        logger: FileLogger,
    ) -> Observation:
        question, context_a, context_b, keep_answer = self._get_params(llm_parameters)
        output_hidden_cols = await database_state.get_output_hidden_cols(
            operation=self,
            llm_configuration=llm_parameters,
            dependent_columns=[context_a, context_b],
            data_type=DataType.FLOAT,
            database_state=database_state,
            logger=logger / "get-skip-columns",
        )
        return ThresholdFilterOnComputedDataObservation(
            hidden_columns=output_hidden_cols,
            logical_plan_step=logical_plan_step,
            quality=self.quality,
        )

    async def profile(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        database_state: IntermediateState,
        observation: Observation,
        llm_parameters: Dict[str, str],
        sample: "ProfilingSampleSpecification",
        data_sample: Sequence[pd.DataFrame],
        logger: FileLogger,
    ):
        run_result = await self.run_outside_db(
            inputs=inputs,
            input_data=data_sample,
            llm_parameters=llm_parameters,
            database_state=database_state,
            observation=observation,
            labels=None,
            logger=logger,
        )
        mask = run_result.output_data
        index_names = list(run_result.input_data.index.names)
        distance_df = pd.DataFrame([m[1] for m in mask], columns=["__decision__"])
        distance_df.index = pd.MultiIndex.from_tuples(
            [tuple(m[0]) for m in mask], names=index_names
        )
        ordered_df = run_result.input_data[[]].merge(
            distance_df, left_on=index_names, right_on=index_names
        )
        full_matrix_or_log_odds = (
            Tensor(ordered_df["__decision__"].tolist()).float().reshape(-1, 1)
        )
        return run_result.input_data, full_matrix_or_log_odds, run_result.cost

    def profile_get_decision_matrix(
        self, parameters: Callable[[str], Tensor], profile_output: Any
    ) -> Tensor:
        logodds = profile_output
        threshold_upper = parameters("logodds_threshold_upper")
        threshold_lower = parameters("logodds_threshold_lower")
        keep_matrix = logodds - threshold_upper.reshape((1, -1))
        discard_matrix = threshold_lower.reshape((1, -1)) - logodds
        unsure_matrix = torch.zeros_like(keep_matrix)
        return torch.stack([keep_matrix, discard_matrix, unsure_matrix], dim=2)

    async def prepare(self, database: Database, logger: FileLogger):
        """Prepare KV caches for all text columns (both left and right sides)."""
        text_cols = database.get_concrete_columns_by_type(DataType.TEXT)
        for text_col in text_cols:
            logger.info(__name__, f"ApeTextQaFilter: preparing KV cache for {text_col}.")
            texts = [
                row[0]
                for row in database.sql(
                    f"SELECT {text_col.column_name} FROM {text_col.table_name}"
                ).fetchall()
            ]
            await self.text_qa_backend.prepare(
                column=text_col, texts=texts, cache_dir=database.cache_dir
            )

    async def wind_down(self):
        await self.text_qa_backend.wind_down()

    @property
    def prefers_run_outside_db(self) -> bool:
        return True

    async def _run_outside_db(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        input_data: pd.DataFrame,
        llm_parameters: Dict[str, Any],
        database_state: IntermediateState,
        observation: Observation,
        labels: Optional["LabelsDefinition"],
        logger: FileLogger,
    ):
        question, context_a, context_b, keep_answer = self._get_params(llm_parameters)
        context_a_concrete = database_state.get_concrete_column_from_virtual(
            context_a, avoid_materialization_points=True
        )

        # Collect unique left and right texts in insertion order
        left_text_to_idx: Dict[str, int] = {}
        right_text_to_idx: Dict[str, int] = {}
        for _, row in input_data.iterrows():
            lt = row[context_a.column_name]
            rt = row[context_b.column_name]
            if lt not in left_text_to_idx:
                left_text_to_idx[lt] = len(left_text_to_idx)
            if rt not in right_text_to_idx:
                right_text_to_idx[rt] = len(right_text_to_idx)

        left_texts = list(left_text_to_idx.keys())
        right_texts = list(right_text_to_idx.keys())

        logger.info(
            __name__,
            f"ApeTextQaFilter: {len(input_data)} pairs, "
            f"{len(left_texts)} unique left × {len(right_texts)} unique right, "
            f"question='{question}'",
        )

        result_dict = await self.text_qa_backend.run_ape_join(
            left_column_concrete=context_a_concrete,
            left_texts=left_texts,
            right_texts=right_texts,
            question=question,
            cache_dir=database_state.cache_dir,
            boolean_question=True,
            temperature=self.temperature,
            scale=self.scale,
            max_batch_size=self.max_batch_size,
        )

        runtime = result_dict["runtime"]
        log_odds_grid = result_dict["log_odds"]  # N×M nested list

        mask = []
        for data_id, row in input_data.iterrows():
            lt = row[context_a.column_name]
            rt = row[context_b.column_name]
            lo = log_odds_grid[left_text_to_idx[lt]][right_text_to_idx[rt]]
            mask.append((data_id, lo))

        logger.info(
            __name__,
            f"ApeTextQaFilter: done in {runtime:.1f}s, {len(mask)} results",
        )
        return RunOutsideResult(
            mask,
            ProfilingCost(runtime=runtime, monetary_cost=0.0),
            input_data=input_data,
        )

    def _get_params(self, llm_parameters: Dict[str, Any]):
        question: str = llm_parameters["question"]
        context_a: VirtualColumnIdentifier = llm_parameters["context_a"]
        context_b: VirtualColumnIdentifier = llm_parameters["context_b"]
        keep_answer: str = llm_parameters["keep_answer"].lower().strip()
        return question, context_a, context_b, keep_answer

    def get_operation_identifier(self) -> str:
        return "ApeTextQaFilter-" + self.text_qa_backend.get_operation_identifier()

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="ApeTextQaFilter",
            explanation=(
                "Ask a binary question about a PAIR of text documents. "
                "Both documents are provided as pre-built KV caches (APE: Adaptive Parallel Encoding). "
                "The question must be a static string that applies to both documents together — "
                "do NOT include a placeholder for either document."
            ),
            parameters=[
                LLMParameter(
                    "question",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation=(
                        "A static binary question about both documents, e.g. "
                        "'Do they both express a positive sentiment?' or "
                        "'Are both texts about the same topic?'. "
                        "Do NOT include any column placeholders — both texts are already "
                        "provided as KV caches and are not part of the question string."
                    ),
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
                LLMParameter(
                    "context_a",
                    LLMParameterColumnDtype(DataTypes.TEXT),
                    explanation="The LEFT context column (fully qualified: table.column). Its KV cache is the left document.",
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "context_b",
                    LLMParameterColumnDtype(DataTypes.TEXT),
                    explanation="The RIGHT context column (fully qualified: table.column). Its KV cache is the right document.",
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "keep_answer",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="Keep all pairs where the question is answered with this value (either 'yes' or 'no').",
                    optional=False,
                    choices=["yes", "no"],
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
            ],
        )

    def get_tuning_parameters(self) -> Sequence[TuningParameter]:
        return [
            TuningParameterContinuous(
                name="logodds_threshold_lower",
                default=DEFAULT_THRESHOLD_LOWER,
                init=INITIAL_THRESHOLD_LOWER,
                min=THRESHOLD_RANGE[0],
                max=THRESHOLD_RANGE[1],
                log_scale=False,
            ),
            TuningParameterContinuous(
                name="logodds_threshold_upper",
                default=DEFAULT_THRESHOLD_UPPER,
                init=INITIAL_THRESHOLD_UPPER,
                min=THRESHOLD_RANGE[0],
                max=THRESHOLD_RANGE[1],
                log_scale=False,
            ),
        ]

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalFilter

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.TEXT_ANALYSIS_FILTER,)

    def get_free_form_equivalence_prompt(self, incoming_text: str, db_text: str) -> Prompt:
        return PromptTemplate(
            [
                Message(
                    "Are these two questions semantically equivalent (yes/no)?\n"
                    "1) {{incoming_text}}\n"
                    "2) {{db_text}}",
                    "user",
                ),
            ]
        ).fill(incoming_text=incoming_text, db_text=db_text)

    def get_hidden_column_type(self) -> HiddenColumnType:
        return HiddenColumnType.FILTER_COLUMN

    def get_is_expensive(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True

    def get_is_potentially_flawed(self) -> bool:
        return True

    def setup(self, database: Database, logger: FileLogger):
        try:
            self.text_qa_backend.setup(logger)
        except Exception as e:
            logger.warning(
                __name__,
                f"Failed to setup ApeTextQa backend: {e}. This operator will not be available.",
            )

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [False]

    def is_tuned(self) -> bool:
        return True
