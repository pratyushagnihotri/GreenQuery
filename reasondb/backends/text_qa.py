import hashlib
import os
import sqlite3
import requests
import time
from abc import abstractmethod
from pathlib import Path
import pandas as pd
import asyncio
from typing import Any, List, Optional, Sequence, Tuple
from reasondb.backends.backend import Backend
from reasondb.database.indentifier import (
    ConcreteColumnIdentifier,
    DataType,
    VirtualColumnIdentifier,
)
from reasondb.query_plan.llm_parameters import LlmParameterTemplate
from reasondb.reasoning.llm import (
    LargeLanguageModel,
    Message,
    PromptTemplate,
)
from reasondb.utils.cache import CACHE_DIR
from reasondb.utils.logging import FileLogger


PORT_KV_TEXT_QA = {
    "meta-llama/Llama-3.1-8B-Instruct": 5010,
    "meta-llama/Llama-3.1-16B-Instruct": 5011,
    "meta-llama/Llama-3.1-70B-Instruct": 5012,
    "Qwen/Qwen2.5-7B-Instruct": 5013,
    "Qwen/Qwen2.5-14B-Instruct": 5014,
}
TEXT_MODEL_CACHE_DB = CACHE_DIR / Path("text_model_cache.sqlite")


def text_qa_server_port(model_id: str) -> int:
    """Use a per-job service port when local model jobs run concurrently."""
    configured = os.environ.get("GREENQUERY_LOCAL_TEXT_QA_PORT")
    return int(configured) if configured else PORT_KV_TEXT_QA[model_id]


def direct_qa_batch_size() -> int:
    """Return the bounded local direct-inference batch size."""
    value = os.environ.get("GREENQUERY_DIRECT_QA_BATCH_SIZE", "128")
    try:
        batch_size = int(value)
    except ValueError as error:
        raise ValueError(
            "GREENQUERY_DIRECT_QA_BATCH_SIZE must be a positive integer"
        ) from error
    if batch_size < 1:
        raise ValueError("GREENQUERY_DIRECT_QA_BATCH_SIZE must be a positive integer")
    return batch_size


class TextQaBackend(Backend):
    def __init__(self):
        pass

    @abstractmethod
    async def run(
        self,
        question_template: LlmParameterTemplate,
        columns: Sequence[VirtualColumnIdentifier],
        context_column_virtual: VirtualColumnIdentifier,
        context_column_concrete: ConcreteColumnIdentifier,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        raise NotImplementedError

    @abstractmethod
    def setup(
        self,
        logger: FileLogger,
    ):
        pass

    @abstractmethod
    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        texts: Sequence[str],
    ):
        pass

    @abstractmethod
    async def wind_down(self):
        pass

    @property
    @abstractmethod
    def returns_log_odds(self) -> bool:
        pass

    async def run_join(
        self,
        question_template: LlmParameterTemplate,
        columns: Sequence[VirtualColumnIdentifier],
        context_column_virtual: VirtualColumnIdentifier,
        context_column_concrete: ConcreteColumnIdentifier,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        """Default: fall back to run() (e.g. for LLMTextQABackend which has no KV caches)."""
        return await self.run(
            question_template=question_template,
            columns=[context_column_virtual] + list(columns),
            context_column_virtual=context_column_virtual,
            context_column_concrete=context_column_concrete,
            data=data,
            data_type=data_type,
            cache_dir=cache_dir,
            boolean_question=boolean_question,
            logger=logger,
        )

    @abstractmethod
    async def run_direct(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[List[Tuple[str, float, float]], float, float]:
        """Evaluate questions with explicit left-side contexts (no KV cache needed).

        contexts[i] is the left document text, prepended before the instruction in the prompt,
        matching the token structure of TextQaFilter where the left document is in the KV cache.
        Returns a list of (answer, log_odds, runtime) tuples, plus total runtime and cost.
        """
        raise NotImplementedError


class LLMTextQABackend(TextQaBackend):
    def __init__(self, llm: LargeLanguageModel):
        self.llm = llm

    @property
    def returns_log_odds(self) -> bool:
        return False

    def setup(
        self,
        logger: FileLogger,
    ):
        pass

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        texts: Sequence[str],
    ):
        await self.llm.prepare()

    async def wind_down(self):
        await self.llm.close()

    async def run(
        self,
        question_template: LlmParameterTemplate,
        columns: Sequence[VirtualColumnIdentifier],
        context_column_virtual: VirtualColumnIdentifier,
        context_column_concrete: ConcreteColumnIdentifier,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        args = []
        context_column = data[context_column_virtual.column_name]
        fill_columns = [col for col in columns if col != context_column_virtual]
        for row_pos, data_id in enumerate(data.index):
            context = context_column.iat[row_pos]
            question = question_template.fill(
                {
                    col: data[col.column_name].iat[row_pos]
                    for col in fill_columns
                }
            )
            if boolean_question:
                question = f"Answer only with 'Yes' or 'No'. Do not add any other comments: {question}"
            args.append((data_id, question, context))
        responses_runtimes_costs = await asyncio.gather(
            *(
                self.run_single(data_id, question, context, logger=logger)
                for data_id, question, context in args
            )
        )
        responses, runtimes, costs = tuple(zip(*responses_runtimes_costs))
        responses = [resp.strip() for resp in responses]
        response_no_quotes = []
        for response in responses:
            if (
                response.startswith('"')
                and response.endswith('"')
                and response.count('"') == 2
            ):
                response_no_quotes.append(response[1:-1])
            elif (
                response.startswith("'")
                and response.endswith("'")
                and response.count("'") == 2
            ):
                response_no_quotes.append(response[1:-1])
            else:
                response_no_quotes.append(response)
        result = [
            (dt[0], data_type.convert(resp), 0.0)
            for dt, resp in zip(args, response_no_quotes)
        ]
        return result, sum(runtimes), sum(costs)

    async def run_single(
        self,
        data_id: Tuple,
        question: str,
        context: Optional[str],
        logger: FileLogger,
    ) -> Tuple[str, float, float]:
        if context is not None:
            prompt = PromptTemplate(
                messages=[
                    Message(
                        text="Answer the question by the user based on the context!",
                        role="system",
                    ),
                    Message(
                        text="Question: {{question}}. Context: {{context}}",
                        role="user",
                    ),
                ]
            ).fill(question=question, context=context)
        else:
            prompt = PromptTemplate(
                messages=[
                    Message(
                        text="Answer the question by the user based on the context!",
                        role="system",
                    ),
                    Message(
                        text="Question: {{question}}.",
                        role="user",
                    ),
                ]
            ).fill(question=question)
        return await self.llm.invoke_with_runtime_and_cost(prompt=prompt, logger=logger)

    async def run_direct(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[List[Tuple[str, float, float]], float, float]:
        results, total_runtime, total_cost = [], 0.0, 0.0
        for context, q in zip(contexts, questions):
            full_q = (
                context + q
                if not boolean_question
                else (
                    f"Answer only with 'Yes' or 'No'. Do not add any other comments: {context}{q}"
                )
            )
            answer, runtime, cost = await self.run_single(
                data_id=None, question=full_q, context=None, logger=logger
            )
            results.append((answer, 0.0, runtime))
            total_runtime += runtime
            total_cost += cost
        return results, total_runtime, total_cost

    def get_operation_identifier(self) -> str:
        return f"LLMTextQABackend-{self.llm.model_id}"


class KvTextQABackend(TextQaBackend):
    def __init__(
        self, model_id: str, compression_ratio: float, cache_enabled: bool = True
    ):
        self._model_id = model_id
        self.compression_ratio = compression_ratio
        self.cache_enabled = cache_enabled

    @property
    def returns_log_odds(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
        return f"{self._model_id}-cr{self.compression_ratio}"

    def setup(
        self,
        logger: FileLogger,
    ):
        result = requests.get(
            f"http://localhost:{text_qa_server_port(self._model_id)}/status"
        )
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self._model_id
        assert self.compression_ratio in json_response["compression_ratios"]
        logger.info(
            __name__,
            f"KV Text model {self._model_id} with compression ratio {self.compression_ratio} is ready",
        )
        if self.cache_enabled:
            self.cache_init()

    def shutdown(self, logger: FileLogger):
        self.close_cache()

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        texts: Sequence[str],
    ):
        if not self.cache_enabled:
            return
        response = requests.post(
            f"http://localhost:{text_qa_server_port(self._model_id)}/prepare_caches",
            json={
                "column_name": column.name,
                "texts": texts,
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["status"] == "cache_ready"

    async def wind_down(self):
        pass

    async def run(
        self,
        question_template: LlmParameterTemplate,
        columns: Sequence[VirtualColumnIdentifier],
        context_column_virtual: VirtualColumnIdentifier,
        context_column_concrete: ConcreteColumnIdentifier,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        result = [None] * len(data)
        total_runtime = 0.0  # accumulated once per unique pair, not per duplicate row

        # Deduplicate by (question, context): identical pairs share one LLM call.
        # pair_to_indices: (question, context) → [(row_idx, data_id)]
        pair_to_indices = {}
        for i, (data_id, row) in enumerate(data.iterrows()):
            context = row[context_column_virtual.column_name]
            question = question_template.fill(
                {
                    col: row[col.column_name]
                    for col in columns
                    if col != context_column_virtual
                }
            )
            pair_to_indices.setdefault((question, context), []).append((i, data_id))

        logger.info(
            __name__,
            f"[run] {len(data)} input rows → {len(pair_to_indices)} unique (question, context) pairs after deduplication",
        )
        non_cached_questions = []
        non_cached_contexts = []
        non_cached_pairs = []
        for (question, context), indices in pair_to_indices.items():
            cached = self.get_cached(question, context) if self.cache_enabled else None
            if cached:
                response, log_odd, runtime = cached
                total_runtime += (
                    runtime  # counted once regardless of number of duplicate rows
                )
                for i, data_id in indices:
                    result[i] = (
                        data_id,
                        data_type.convert(response.strip("\"'")),
                        log_odd,
                    )
            else:
                non_cached_questions.append(question)
                non_cached_contexts.append(context)
                non_cached_pairs.append((question, context))

        if non_cached_questions:
            if self.cache_enabled:
                responses_runtimes, telemetry = self._invoke(
                    column=context_column_concrete,
                    questions=non_cached_questions,
                    texts=non_cached_contexts,
                    cache_dir=cache_dir,
                    boolean_question=boolean_question,
                )
            else:
                responses_runtimes, telemetry = self._invoke_direct_no_cache(
                    questions=non_cached_questions,
                    texts=non_cached_contexts,
                    boolean_question=boolean_question,
                )
            for (response, log_odd, runtime), (question, context) in zip(
                responses_runtimes, non_cached_pairs
            ):
                total_runtime += (
                    runtime  # counted once regardless of number of duplicate rows
                )
                for i, data_id in pair_to_indices[(question, context)]:
                    result[i] = (
                        data_id,
                        data_type.convert(response.strip("\"'")),
                        log_odd,
                    )
                if self.cache_enabled:
                    self.cache(question, context, response, log_odd, runtime)
            if telemetry:
                predictions = []
                telemetry_rows = telemetry.get("rows") or []
                for unique_index, ((response, log_odd, _), pair) in enumerate(
                    zip(responses_runtimes, non_cached_pairs)
                ):
                    token_detail = (
                        telemetry_rows[unique_index]
                        if unique_index < len(telemetry_rows)
                        else {}
                    )
                    for _, data_id in pair_to_indices[pair]:
                        predictions.append(
                            {
                                "data_id": list(data_id)
                                if isinstance(data_id, tuple)
                                else data_id,
                                "context": pair[1],
                                "answer": str(response).strip(),
                                "predicted_label": int(float(log_odd) > 0.0),
                                "log_odds": float(log_odd),
                                "input_tokens": token_detail.get("input_tokens"),
                                "output_tokens": token_detail.get("output_tokens"),
                            }
                        )
                telemetry["predictions"] = predictions
                try:
                    from evaluation.measurement.local_inference import (
                        record_local_inference,
                    )

                    record_local_inference(telemetry)
                except Exception:
                    pass
        return result, total_runtime, 0.0

    def cache_init(self):
        """
        Initializes the SQLite database and table for multi-process caching.
        """
        self.cache_conn = sqlite3.connect(
            TEXT_MODEL_CACHE_DB, check_same_thread=False, timeout=60.0
        )

        with self.cache_conn:
            # Enable WAL mode for high concurrency read/writes
            self.cache_conn.execute("PRAGMA journal_mode=WAL;")
            self.cache_conn.execute("PRAGMA synchronous=NORMAL;")

            # Create table with (model_id, hash) as the highly-optimized Primary Key
            self.cache_conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    model_id TEXT,
                    hash TEXT,
                    question TEXT,
                    context TEXT,
                    response TEXT,
                    log_odds REAL,
                    runtime REAL,
                    PRIMARY KEY (model_id, hash) 
                )
            """)

    def _ensure_cache(self):
        if not hasattr(self, "cache_conn"):
            self.cache_init()

    def get_cached(
        self, question: str, context: str
    ) -> Optional[Tuple[str, float, float]]:
        self._ensure_cache()
        # 1. Generate the short hash for the index lookup
        hash_val = self.hash_text(f"{question}-{context}")

        cursor = self.cache_conn.cursor()
        # 2. Fast database index lookup
        cursor.execute(
            """
            SELECT question, context, response, log_odds, runtime 
            FROM cache 
            WHERE model_id = ? AND hash = ?
            """,
            (self.model_id, hash_val),
        )
        row = cursor.fetchone()

        if row:
            db_question, db_context, response, log_odds, runtime = row
            # 3. Double-check exact text match to prevent hash collisions
            if db_question == str(question) and db_context == str(context):
                return response, log_odds, runtime

        return None

    def cache(
        self,
        question: str,
        context: str,
        response: str,
        log_odds: float,
        runtime: float,
    ):
        self._ensure_cache()
        hash_val = self.hash_text(f"{question}-{context}")

        with self.cache_conn:
            # Insert or update based on the model_id + hash primary key
            self.cache_conn.execute(
                """
                INSERT OR REPLACE INTO cache 
                (model_id, hash, question, context, response, log_odds, runtime)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.model_id,
                    hash_val,
                    str(question),
                    str(context),
                    str(response),
                    log_odds,
                    runtime,
                ),
            )

    def close_cache(self):
        """Cleanly close the database connection when shutting down."""
        if hasattr(self, "conn") and self.cache_conn:
            self.cache_conn.close()

    def hash_text(self, text: str) -> str:
        """Generate a sha256hash for a given path."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        questions: List[str],
        texts: List[str],
        cache_dir: Path,
        boolean_question: bool,
    ) -> Tuple[List[Tuple[str, float, float]], dict]:
        assert len(questions) == len(texts)
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{text_qa_server_port(self._model_id)}/text_qa",
            json={
                "column_name": column.name,
                "texts": texts,
                "questions": questions,
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
                "boolean": boolean_question,
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        time_end = time.time()
        runtime = time_end - time_start
        result = []
        for answer, log_odd in zip(json_response["answers"], json_response["log_odds"]):
            result.append((answer, log_odd, runtime / len(texts)))
        return result, json_response.get("telemetry") or {}

    def _invoke_direct_no_cache(
        self,
        questions: List[str],
        texts: List[str],
        boolean_question: bool,
    ) -> Tuple[List[Tuple[str, float, float]], dict]:
        time_start = time.time()
        batch_size = direct_qa_batch_size()
        response = requests.post(
            f"http://localhost:{text_qa_server_port(self._model_id)}/text_qa_direct",
            json={
                "questions": questions,
                "contexts": texts,
                "boolean": boolean_question,
                "cache_policy": "disabled",
                "batch_size": batch_size,
            },
        )
        response.raise_for_status()
        payload = response.json()
        answers = payload.get("answers") or []
        log_odds = payload.get("log_odds") or []
        if len(answers) != len(texts) or len(log_odds) != len(texts):
            raise RuntimeError("Direct TextQA response length does not match input rows")
        elapsed = time.time() - time_start
        runtime_per_row = elapsed / max(len(texts), 1)
        telemetry = payload.get("telemetry") or {}
        telemetry.setdefault("inference_seconds", elapsed)
        telemetry.setdefault("processed_rows", len(texts))
        telemetry.setdefault("request_count", 1)
        telemetry.setdefault("answer_cache_hits", 0)
        telemetry.setdefault("kv_cache_hits", 0)
        telemetry.setdefault("execution_mode", "direct_no_cache")
        telemetry.setdefault("inference_batch_size", batch_size)
        return [
            (answer, float(log_odd), runtime_per_row)
            for answer, log_odd in zip(answers, log_odds)
        ], telemetry

    async def run_join(
        self,
        question_template: LlmParameterTemplate,
        columns: Sequence[VirtualColumnIdentifier],
        context_column_virtual: VirtualColumnIdentifier,
        context_column_concrete: ConcreteColumnIdentifier,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        """Load each unique context KV cache once, answer all its questions before moving on."""
        context_col_name = context_column_virtual.column_name
        result = [None] * len(data)
        total_runtime = 0.0  # accumulated once per unique pair, not per duplicate row
        # Group by context_text: all left rows with the same text share one KV cache pass.
        # Within each group, questions are deduplicated: identical (left_text, right_text)
        # pairs (same question string) share one LLM call and fan out the answer.
        context_to_rows = {}  # context_text → {question → [(row_idx, data_id)]}

        for i, (data_id, row) in enumerate(data.iterrows()):
            context = row[context_col_name]
            question = question_template.fill(
                {col: row[col.column_name] for col in columns}
            )
            context_to_rows.setdefault(context, {}).setdefault(question, []).append(
                (i, data_id)
            )

        # Check cache before calling the server — cached pairs skip the LLM entirely
        # (mirrors run() filter path). Runtime is counted once per unique pair.
        non_cached_texts = []
        non_cached_questions_per_text = []
        for text, q_dict in context_to_rows.items():
            non_cached_qs = []
            for question in q_dict:
                cached = self.get_cached(question, text) if self.cache_enabled else None
                if cached:
                    response, log_odd, runtime = cached
                    total_runtime += (
                        runtime  # counted once regardless of number of duplicate rows
                    )
                    for row_idx, data_id in q_dict[question]:
                        result[row_idx] = (
                            data_id,
                            data_type.convert(response.strip("\"'")),
                            log_odd,
                        )
                else:
                    non_cached_qs.append(question)
            if non_cached_qs:
                non_cached_texts.append(text)
                non_cached_questions_per_text.append(non_cached_qs)

        total_pairs = sum(len(q_dict) for q_dict in context_to_rows.values())
        total_non_cached = sum(len(qs) for qs in non_cached_questions_per_text)
        unique_right_count = len(
            set(q for qs in non_cached_questions_per_text for q in qs)
        )
        logger.info(
            __name__,
            f"[run_join] Dispatching join: {len(non_cached_texts)} unique left contexts, "
            f"{unique_right_count} unique right contexts, "
            f"{total_non_cached} total (context, question) pairs "
            f"({total_pairs - total_non_cached} served from cache)",
        )

        if non_cached_texts:
            time_start = time.time()
            resp = self._invoke_join(
                column=context_column_concrete,
                unique_texts=non_cached_texts,
                questions_per_text=non_cached_questions_per_text,
                cache_dir=cache_dir,
                boolean_question=boolean_question,
            )
            elapsed = time.time() - time_start
            runtime_per = elapsed / max(total_non_cached, 1)
            logger.info(
                __name__,
                f"[run_join] Server responded in {elapsed:.2f}s for {total_non_cached} pairs "
                f"({runtime_per*1000:.1f}ms per pair)",
            )
            for text, questions, answers_list, log_odds_list in zip(
                non_cached_texts,
                non_cached_questions_per_text,
                resp["answers_per_text"],
                resp["log_odds_per_text"],
            ):
                q_dict = context_to_rows[text]
                for question, answer, log_odd in zip(
                    questions, answers_list, log_odds_list
                ):
                    total_runtime += runtime_per  # counted once regardless of number of duplicate rows
                    for row_idx, data_id in q_dict[question]:
                        result[row_idx] = (
                            data_id,
                            data_type.convert(answer.strip("\"'")),
                            log_odd,
                        )
                    self.cache(question, text, answer, log_odd, runtime_per)

        assert all(
            r is not None for r in result
        ), f"BUG: {result.count(None)} of {len(result)} results are None after run_join"
        return result, total_runtime, 0.0

    def _invoke_join(
        self,
        column: ConcreteColumnIdentifier,
        unique_texts: List[str],
        questions_per_text: List[List[str]],
        cache_dir: Path,
        boolean_question: bool,
    ) -> dict:
        response = requests.post(
            f"http://localhost:{text_qa_server_port(self._model_id)}/text_qa_join",
            json={
                "column_name": column.name,
                "unique_texts": unique_texts,
                "questions_per_text": questions_per_text,
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
                "boolean": boolean_question,
            },
        )
        assert response.status_code == 200
        return response.json()

    async def run_ape_join(
        self,
        left_column_concrete: ConcreteColumnIdentifier,
        left_texts: List[str],
        right_texts: List[str],
        question: str,
        cache_dir: Path,
        boolean_question: bool,
        temperature: float = 0.9,
        scale: float = 0.9,
        max_batch_size: int = 8,
    ) -> dict:
        """Call /text_qa_ape_join: cartesian product of left×right texts, both from KV caches.

        Returns dict with keys: answers, log_odds, runtime (seconds).
        Runtime is measured around the server call, matching run_join's approach.
        """
        time_start = time.time()
        result = self._invoke_ape_join(
            column=left_column_concrete,
            left_texts=left_texts,
            right_texts=right_texts,
            question=question,
            cache_dir=cache_dir,
            boolean_question=boolean_question,
            temperature=temperature,
            scale=scale,
            max_batch_size=max_batch_size,
        )
        result["runtime"] = time.time() - time_start
        return result

    def _invoke_ape_join(
        self,
        column: ConcreteColumnIdentifier,
        left_texts: List[str],
        right_texts: List[str],
        question: str,
        cache_dir: Path,
        boolean_question: bool,
        temperature: float = 0.9,
        scale: float = 0.9,
        max_batch_size: int = 8,
    ) -> dict:
        response = requests.post(
            f"http://localhost:{text_qa_server_port(self._model_id)}/text_qa_ape_join",
            json={
                "column_name": column.name,
                "left_texts": left_texts,
                "right_texts": right_texts,
                "question": question,
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
                "boolean": boolean_question,
                "temperature": temperature,
                "scale": scale,
                "max_batch_size": max_batch_size,
            },
        )
        if response.status_code != 200:
            body = (
                response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else {}
            )
            tb = body.get("traceback", response.text)
            raise RuntimeError(
                f"text_qa_ape_join returned {response.status_code}: {body.get('error', response.text)}\n{tb}"
            )
        return response.json()

    async def run_ape_batch_join(
        self,
        left_column_concrete: ConcreteColumnIdentifier,
        left_texts: List[str],
        right_texts: List[str],
        question: str,
        cache_dir: Path,
        boolean_question: bool,
        temperature: float = 0.9,
        scale: float = 0.9,
        max_batch_size: int = 8,
    ) -> dict:
        """Call /text_qa_ape_batch_join: batched APE join using flash_attn_varlen_func.

        Returns dict with keys: answers, log_odds, runtime (seconds).
        Runtime is measured around the server call, matching run_join's approach.
        """
        time_start = time.time()
        result = self._invoke_ape_batch_join(
            column=left_column_concrete,
            left_texts=left_texts,
            right_texts=right_texts,
            question=question,
            cache_dir=cache_dir,
            boolean_question=boolean_question,
            temperature=temperature,
            scale=scale,
            max_batch_size=max_batch_size,
        )
        result["runtime"] = time.time() - time_start
        return result

    def _invoke_ape_batch_join(
        self,
        column: ConcreteColumnIdentifier,
        left_texts: List[str],
        right_texts: List[str],
        question: str,
        cache_dir: Path,
        boolean_question: bool,
        temperature: float = 0.9,
        scale: float = 0.9,
        max_batch_size: int = 8,
    ) -> dict:
        response = requests.post(
            f"http://localhost:{text_qa_server_port(self._model_id)}/text_qa_ape_batch_join",
            json={
                "column_name": column.name,
                "left_texts": left_texts,
                "right_texts": right_texts,
                "question": question,
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
                "boolean": boolean_question,
                "temperature": temperature,
                "scale": scale,
                "max_batch_size": max_batch_size,
            },
        )
        if response.status_code != 200:
            body = (
                response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else {}
            )
            tb = body.get("traceback", response.text)
            raise RuntimeError(
                f"text_qa_ape_batch_join returned {response.status_code}: {body.get('error', response.text)}\n{tb}"
            )
        return response.json()

    async def run_hidden_state(
        self,
        column: ConcreteColumnIdentifier,
        texts: List[str],
        suffix_prompt: str,
        cache_dir: Path,
    ) -> List[List[float]]:
        """Call /hidden_state and return a list of hidden-state vectors (one per text)."""
        response = requests.post(
            f"http://localhost:{text_qa_server_port(self._model_id)}/hidden_state",
            json={
                "column_name": column.name,
                "texts": texts,
                "suffix_prompt": suffix_prompt,
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
            },
        )
        if response.status_code != 200:
            body = (
                response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else {}
            )
            tb = body.get("traceback", response.text)
            raise RuntimeError(
                f"hidden_state returned {response.status_code}: {body.get('error', response.text)}\n{tb}"
            )
        return response.json()["hidden_states"]

    async def run_hidden_state_similarity(
        self,
        left_column: ConcreteColumnIdentifier,
        right_column: ConcreteColumnIdentifier,
        left_texts: List[str],
        right_texts: List[str],
        pair_left_indices: List[int],
        pair_right_indices: List[int],
        suffix_prompt: str,
        cache_dir: Path,
    ) -> Tuple[List[float], float, float, float]:
        """Call /hidden_state_similarity and return (similarities, runtime_left, runtime_right, runtime_sim).

        The server computes hidden states for left and right texts separately (for accurate
        per-side timing), then computes cosine similarity on GPU.
        Returns only scalars — no large vector transfer over HTTP.
        """
        response = requests.post(
            f"http://localhost:{text_qa_server_port(self._model_id)}/hidden_state_similarity",
            json={
                "left_column_name": left_column.name,
                "right_column_name": right_column.name,
                "left_texts": left_texts,
                "right_texts": right_texts,
                "pair_left_indices": pair_left_indices,
                "pair_right_indices": pair_right_indices,
                "suffix_prompt": suffix_prompt,
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
            },
        )
        if response.status_code != 200:
            body = (
                response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else {}
            )
            tb = body.get("traceback", response.text)
            raise RuntimeError(
                f"hidden_state_similarity returned {response.status_code}: {body.get('error', response.text)}\n{tb}"
            )
        data = response.json()
        return (
            data["similarities"],
            data["runtime_left"],
            data["runtime_right"],
            data["runtime_sim"],
        )

    def _invoke_direct(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
    ) -> List[Tuple[str, float, float]]:
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{text_qa_server_port(self._model_id)}/text_qa_direct",
            json={
                "questions": questions,
                "contexts": contexts,
                "boolean": boolean_question,
            },
        )
        assert (
            response.status_code == 200
        ), f"text_qa_direct returned {response.status_code}: {response.text}"
        json_response = response.json()
        runtime = time.time() - time_start
        return [
            (answer, log_odd, runtime / len(questions))
            for answer, log_odd in zip(
                json_response["answers"], json_response["log_odds"]
            )
        ]

    async def run_direct(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[List[Tuple[str, float, float]], float, float]:
        """Evaluate questions via the /text_qa_direct endpoint. Cache key = (question, context)."""
        non_cached_questions: List[str] = []
        non_cached_contexts: List[str] = []
        non_cached_idx: List[int] = []
        results: List[Optional[Tuple[str, float, float]]] = [None] * len(questions)

        for i, (q, ctx) in enumerate(zip(questions, contexts)):
            cached = self.get_cached(q, ctx) if self.cache_enabled else None
            if cached is not None:
                response, log_odd, runtime = cached
                results[i] = (response, log_odd, runtime)
            else:
                non_cached_questions.append(q)
                non_cached_contexts.append(ctx)
                non_cached_idx.append(i)

        if non_cached_questions:
            raw = self._invoke_direct(
                non_cached_questions, non_cached_contexts, boolean_question
            )
            for (answer, log_odd, rt), i, q, ctx in zip(
                raw, non_cached_idx, non_cached_questions, non_cached_contexts
            ):
                results[i] = (answer, log_odd, rt)
                self.cache(q, ctx, answer, log_odd, rt)

        total_runtime = sum(r[2] for r in results)
        return results, total_runtime, 0.0

    def get_operation_identifier(self) -> str:
        return f"LLMTextQABackend-{self.model_id}"
