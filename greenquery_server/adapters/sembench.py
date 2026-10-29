"""Real LOTUS, Palimpzest (Abacus), ThalamusDB and CAESURA through SemBench.

Each system runs from its own SemBench venv via evaluation/scripts/run_sembench_smoke.sbatch
on the DGX (GREENQUERY_EXECUTOR=slurm). SemBench Movie Q1/Q2/Q4/Q5 are Stretto's Movie
Q1/Q2/Q4/Q3, so records reuse Stretto's query names and the same dataset label.
Quality comes from SemBench's own evaluator. Local (Ollama) models run through
run_sembench_measured.sbatch with measured energy; API models through the smoke script (no energy).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from greenquery_server import settings
from greenquery_server.adapters.base import AVAILABLE, PLANNED, RunSpec, SlurmJob, Step, SystemAdapter
from reasondb.reasoning.providers import get_model_spec, load_custom_registry

# Stretto Movie query name -> SemBench Movie query id.
SEMBENCH_MOVIE_QUERIES: Dict[str, str] = {
    "Q1_positive_reviews": "1",
    "Q2_positive_reviews_for_taken_3": "2",
    "Q3_reviews_same_sentiment": "5",
    "Q4_positive_reviews_for_taken_3_ratio": "4",
}
QUERY_NAME_BY_SEMBENCH_ID = {f"Q{qid}": name for name, qid in SEMBENCH_MOVIE_QUERIES.items()}
# SemBench sample with valid ground truth for all four queries: all 120 "taken_3"
# reviews (Q2/Q4) and the same 256 "ant_man" reviews as SemBench's default sf=2000.
# (sf=100 contains a single movie, so Q2/Q4 ground truth is empty.)
SCALE_FACTOR = 1000


class SemBenchAdapter(SystemAdapter):
    workloads = ["movie"]
    # API models run through run_sembench_smoke.sbatch; local (Ollama) models through
    # run_sembench_measured.sbatch on one GPU with measured energy.
    providers = ["openai", "openai_compatible"]
    multi_model = False

    def __init__(self, key: str, label: str, description: str, notes: str = "", sembench_name: str = ""):
        self.key, self.label, self.description = key, label, description
        self.sembench_name = sembench_name or key  # system name inside SemBench (src/run.py registry)
        cluster = settings.EXECUTOR == "slurm"
        self.status = AVAILABLE if cluster else PLANNED
        self.notes = notes if cluster else "Runs on the DGX through SemBench (start the server with GREENQUERY_EXECUTOR=slurm)."

    def slurm_job(self, spec: RunSpec, out_dir: Path) -> SlurmJob:
        queries = [SEMBENCH_MOVIE_QUERIES[spec.query]] if spec.query else list(SEMBENCH_MOVIE_QUERIES.values())
        model = get_model_spec(spec.model)
        if model is not None and model.provider == "openai_compatible":
            return SlurmJob(
                script="evaluation/scripts/run_sembench_measured.sbatch",
                env={
                    "SYSTEMS": self.sembench_name,
                    "QUERIES": " ".join(queries),
                    "LOCAL_MODEL": model.api_model,
                    "REPETITIONS": str(spec.repetitions),
                    "SCALE_FACTOR": str(SCALE_FACTOR),
                    "IDLE_SECONDS": str(settings.MEASURED_IDLE_SECONDS),
                    "QUERY_TIMEOUT": str(settings.MEASURED_QUERY_TIMEOUT),
                    "GREENQUERY_ROOT": str(settings.REPO_ROOT),
                    "SEMBENCH_ROOT": str(settings.SEMBENCH_ROOT),
                    # Models added from the dashboard: Ollama's default context (32k x 4 slots) can push the
                    # KV cache of models without grouped-query attention (e.g. phi3) partly onto the CPU.
                    **({"OLLAMA_CONTEXT_LENGTH": "8192"} if spec.model in load_custom_registry() else {}),
                },
                gres="gpu:1",
            )
        return SlurmJob(
            script="evaluation/scripts/run_sembench_smoke.sbatch",
            env={
                "SYSTEMS": self.sembench_name,
                "QUERIES": " ".join(queries),
                "MODEL": spec.model,
                "REPETITIONS": str(spec.repetitions),
                "SCALE_FACTOR": str(SCALE_FACTOR),
                "SEMBENCH_ROOT": str(settings.SEMBENCH_ROOT),
            },
        )

    def collect_slurm(self, spec: RunSpec, out_dir: Path) -> List[dict]:
        if (out_dir / "energy_records.json").exists():
            return self._collect_measured(spec, out_dir)
        records = []
        for rep_dir in sorted(out_dir.glob("rep_*")):
            path = rep_dir / f"{self.key}.json"
            if not path.exists():
                continue
            for sembench_id, metric in json.loads(path.read_text()).items():
                query = QUERY_NAME_BY_SEMBENCH_ID.get(sembench_id)
                if query is None or (spec.query and query != spec.query):
                    continue
                ok = metric.get("status") == "success"
                records.append(
                    {
                        "demo": "movie",
                        "run": int(rep_dir.name.removeprefix("rep_")),
                        "query": query,
                        "sembench_query": sembench_id,
                        "model": metric.get("model_name") or spec.model,
                        "planner_model": None,
                        "runtime_seconds": metric.get("execution_time") or 0.0,
                        "total_tokens": metric.get("token_usage"),
                        "api_cost_usd": metric.get("money_cost"),
                        # Energy is not measured for SemBench systems yet (proxy + counters planned).
                        "energy_kwh": 0.0,
                        "co2_g": None,
                        "energy_source": "not_available",
                        "physical_operators": [],
                        "quality": {
                            "status": "evaluated" if ok else "failed",
                            "metric": "sembench_precision_recall_f1",
                            "precision": metric.get("precision"),
                            "recall": metric.get("recall"),
                            "f1": metric.get("f1_score"),
                            "returned_rows": metric.get("row_count"),
                            "ground_truth": "SemBench evaluator",
                        },
                        "error": metric.get("error"),
                        "scale_factor": SCALE_FACTOR,
                    }
                )
        return records

    def _collect_measured(self, spec: RunSpec, out_dir: Path) -> List[dict]:
        """Records of a run_sembench_measured.sbatch job: SemBench metrics + measured energy per query."""
        records = []
        for energy in json.loads((out_dir / "energy_records.json").read_text()):
            query = QUERY_NAME_BY_SEMBENCH_ID.get(energy["query"])
            metric_path = out_dir / "metrics" / f"rep_{energy['repetition']}" / f"{self.sembench_name}_{energy['query']}.json"
            metric = next(iter(json.loads(metric_path.read_text()).values()), {}) if metric_path.exists() else {}
            records.append(measured_record(energy, metric, query, spec.model))
        return records

    # Local (non-cluster) execution is not supported for SemBench systems.
    def steps(self, spec: RunSpec, run_dir: Path, repetition: int) -> List[Step]:
        raise NotImplementedError("SemBench systems run on the DGX (GREENQUERY_EXECUTOR=slurm)")

    def finalize_steps(self, spec: RunSpec, job_dir: Path) -> List[Step]:
        return []

    def collect(self, spec: RunSpec, job_dir: Path) -> List[dict]:
        return []


def quality_score(metric: dict) -> Optional[float]:
    """F1 (Q1/Q2/Q5) or 1 - relative error (Q4), as in the GreenQuery results table."""
    if "f1_score" in metric and metric["f1_score"] is not None:
        return float(metric["f1_score"])
    if "relative_error" in metric and metric["relative_error"] is not None:
        return max(0.0, 1.0 - float(metric["relative_error"]))
    return None


def measured_record(energy: dict, metric: dict, query: Optional[str], model: str) -> dict:
    """Compact record with measured energy (GPU counter + CPU share) from join_energy.py output."""
    gpu = energy.get("gpu_energy_j") or 0.0
    cpu = energy.get("cpu_job_energy_j") or 0.0
    timed_out = energy.get("exit_code") == 124
    ok = metric.get("status") == "success" and not timed_out
    return {
        "demo": "movie",
        "run": energy.get("repetition", 1),
        "query": query,
        "sembench_query": energy.get("query"),
        "model": model,
        "planner_model": None,
        "runtime_seconds": energy.get("window_s"),
        "total_tokens": (energy.get("prompt_tokens") or 0) + (energy.get("completion_tokens") or 0),
        "api_cost_usd": 0.0,
        "energy_kwh": (gpu + cpu) / 3.6e6,
        "co2_g": None,
        "energy_source": "measured",
        "physical_operators": [],
        "measurement": {
            "gpu_wh": gpu / 3600,
            "gpu_dynamic_wh": (energy.get("gpu_dynamic_energy_j") or 0.0) / 3600,
            "cpu_job_wh": cpu / 3600,
            "gpu_idle_w": energy.get("gpu_idle_w"),
            "llm_calls": energy.get("llm_calls"),
            "llm_errors": energy.get("llm_errors"),
            "prompt_tokens": energy.get("prompt_tokens"),
            "completion_tokens": energy.get("completion_tokens"),
            "timed_out": timed_out,
        },
        "quality": {
            "status": "evaluated" if ok else "failed",
            "metric": "sembench",
            "score": quality_score(metric) if ok else 0.0,
            "f1": metric.get("f1_score"),
            "relative_error": metric.get("relative_error"),
            "returned_rows": metric.get("row_count"),
            "ground_truth": "SemBench evaluator",
        },
        "error": "timeout (query limit)" if timed_out else metric.get("error"),
        "scale_factor": SCALE_FACTOR,
    }


SEMBENCH_SYSTEMS = [
    SemBenchAdapter("lotus", "LOTUS", "Semantic operators over pandas with model cascades (real LOTUS library)."),
    SemBenchAdapter("palimpzest", "Palimpzest", "Declarative pipelines optimized by Abacus (real Palimpzest library)."),
    SemBenchAdapter("thalamusdb", "ThalamusDB", "SQL with natural-language predicates and approximate processing."),
    SemBenchAdapter("caesura", "CAESURA", "LLM-planned multi-modal query pipelines.",
                    notes="Uses CAESURA's 2023 LangChain/OpenAI stack."),
    SemBenchAdapter("sembench-stretto", "Stretto (SemBench runner)",
                    "Stretto on SemBench's queries, data and evaluator (GreenQuery runner).", sembench_name="stretto"),
    SemBenchAdapter("sembench-abacus", "Abacus in Stretto (SemBench runner)",
                    "Stretto with the Abacus optimizer on SemBench's queries (GreenQuery runner).",
                    sembench_name="stretto_abacus"),
]
