"""SemBench runner for Stretto (ReasonDB), Movie use case.

Installed into SemBench by integrations/sembench/install.sh as
src/scenario/movie/runner/stretto_runner/stretto_runner.py, so Stretto runs the same
SemBench queries on the same data and is scored by the same evaluator as LOTUS,
Palimpzest, ThalamusDB and CAESURA.

Queries follow SemBench's definitions and do what the other systems' runners do: the movie
condition (id == ...) is a plain relational filter applied before the semantic part (LOTUS and
Palimpzest filter in pandas / a lambda, ThalamusDB in SQL WHERE); only "clearly positive" and
"same sentiment" go through Stretto's planner and optimizer. Q5 joins the movie's reviews with
themselves and keeps ten pairs; Q4's ratio is (positive reviews found by Stretto) / (all reviews
of the movie, plain count). Model calls go through OPENAI_BASE_URL (the GreenQuery logging
proxy), so the model is whatever the proxy serves for "gpt-4o-mini" (or the real OpenAI
model when no proxy is used).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd

# GreenQuery repo (reasondb, evaluation.measurement) must be importable.
GREENQUERY_ROOT = os.environ.get("GREENQUERY_ROOT", str(Path.home() / "GreenQuery"))
if GREENQUERY_ROOT not in sys.path:
    sys.path.insert(0, GREENQUERY_ROOT)
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

os.environ.setdefault("GREENQUERY_EMBEDDING_MODEL", "local")  # table metadata embeddings in-process

from runner.generic_runner import GenericQueryMetric, GenericRunner  # noqa: E402

PATTERN_MOVIE = "ant_man_and_the_wasp_quantumania"
NEGATIVE_MOVIE = "taken_3"


class StrettoRunner(GenericRunner):
    """Runs SemBench Movie queries through Stretto's planner, optimizer and executor."""

    def __init__(self, use_case: str, scale_factor: int, model_name: str = "gpt-4o-mini",
                 concurrent_llm_worker: int = 20, skip_setup: bool = False):
        super().__init__(use_case, scale_factor, model_name, concurrent_llm_worker, skip_setup)
        reviews = self.load_data("Reviews.csv")
        reviews.columns = [column.lower() for column in reviews.columns]  # Stretto refers to {reviewtext}
        self.reviews = reviews

    @staticmethod
    def _csv(frame: pd.DataFrame, suffix: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
        frame.to_csv(handle.name, index=False)
        return Path(handle.name)

    def _movie(self, movie: str | None) -> pd.DataFrame:
        return self.reviews if movie is None else self.reviews[self.reviews["id"] == movie]

    def get_system_name(self) -> str:
        return "stretto"

    # --------------------------------------------------------------- execution
    def _run(self, build, movie: str | None = None):
        """Run one Stretto query on a fresh, cold database; returns the result DataFrame.

        `movie` restricts the reviews relationally first (as in the other SemBench runners).
        """
        from reasondb.database.indentifier import InPlaceColumn
        from reasondb.interface.config import Config as ReasonDBConfig
        from reasondb.interface.connect import RaccoonDB
        from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee

        reviews = self._movie(movie)
        name = f"sembench_stretto_{os.getpid()}_{int(time.time() * 1000)}"  # unique: no metadata cache reuse
        config = ReasonDBConfig(name, backend_profile="api", openai_model_id=self.model_name,
                                openai_cache_enabled=False)
        with RaccoonDB(name, config=config) as rc:
            reviews_table = rc.add_table(path=self._csv(reviews, "_reviews.csv"), table_name="reviews",
                                         text_columns=[InPlaceColumn("reviews.reviewtext")])
            target = float(os.environ.get("GREENQUERY_QUALITY_TARGET", "0.8"))  # precision and recall guarantee
            result = build(reviews_table).execute(name, PrecisionGuarantee(target), RecallGuarantee(target))
            return result.to_df()

    def execute_query(self, query_id: int) -> GenericQueryMetric:
        from evaluation.measurement.api_usage import start_collection, stop_collection

        metric = GenericQueryMetric(query_id=query_id, status="pending")
        token = start_collection()
        try:
            query_fn = self._discover_query_impl(query_id)
            started = time.time()
            metric.results = query_fn()
            metric.execution_time = time.time() - started
            metric.status = "success"
        except Exception as error:  # SemBench records failures as data
            metric.status = "failed"
            metric.error = f"{type(error).__name__}: {error}"
            metric.results = self._get_empty_results_dataframe(query_id)
            print(f"  Error in Q{query_id}: {metric.error}")
            traceback.print_exc()
        finally:
            events = stop_collection(token)
        uncached = [event for event in events if not event.cached]
        metric.token_usage = sum(event.total_tokens for event in uncached)
        metric.money_cost = sum(event.total_cost_usd for event in uncached)
        return metric

    # ----------------------------------------------------------------- queries
    @staticmethod
    def _review_ids(frame: pd.DataFrame) -> pd.DataFrame:
        column = next((c for c in frame.columns if c.lower().endswith("reviewid")), None)
        ids = frame[column].astype(str).tolist() if column else []
        return pd.DataFrame({"reviewId": ids})

    def _execute_q1(self) -> pd.DataFrame:
        """Five clearly positive reviews (any movie)."""
        frame = self._run(lambda r: r.filter("{reviewtext} is clearly positive")
                          .project("Keep {reviewid}").limit("Keep 5 rows"))
        return self._review_ids(frame)

    def _execute_q2(self) -> pd.DataFrame:
        """Five positive reviews for movie taken_3."""
        frame = self._run(lambda r: r.filter("{reviewtext} is clearly positive")
                          .project("Keep {reviewid}").limit("Keep 5 rows"), movie=NEGATIVE_MOVIE)
        return self._review_ids(frame)

    def _execute_q4(self) -> pd.DataFrame:
        """Positivity ratio of reviews for taken_3: Stretto finds positives; total is a plain count."""
        frame = self._run(lambda r: r.filter("{reviewtext} is clearly positive").project("Keep {reviewid}"),
                          movie=NEGATIVE_MOVIE)
        total = len(self._movie(NEGATIVE_MOVIE))
        positives = len(self._review_ids(frame))
        return pd.DataFrame({"positivity_ratio": [positives / total if total else 0.0]})

    def _execute_q5(self) -> pd.DataFrame:
        """Ten pairs of reviews with the same sentiment for the pattern movie."""
        # Same form as GreenQuery's Movie Q3 (validated in E1): self-join of one table via rename.
        frame = self._run(lambda r: r.join(
            r.rename("Rename {reviewtext} to [reviewtext_other]").rename("Rename {reviewid} to [reviewid_other]"),
            "{left.reviewtext} and {right.reviewtext_other} share the same sentiment").limit("Keep 10 rows"),
            movie=PATTERN_MOVIE)
        print(f"  Q5 result: {len(frame)} rows, columns {list(frame.columns)}")
        left = next((c for c in frame.columns if c.lower().endswith("reviewid")), None)
        right = next((c for c in frame.columns if c.lower().endswith("reviewid_other")), None)
        if left is None or right is None:
            return self._get_empty_results_dataframe(5)
        return pd.DataFrame({"id": PATTERN_MOVIE, "reviewId": frame[left].astype(str),
                             "reviewId2": frame[right].astype(str)})
