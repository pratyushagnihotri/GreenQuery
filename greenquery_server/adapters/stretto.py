"""Stretto (ReasonDB) through the existing measurement runner scripts.

Each repetition mirrors ``evaluation/scripts/run_online_repeated.sh``: runner ->
summary builder -> validator. After all repetitions the workload's compact
builder turns the raw measurements into dashboard records.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List

from greenquery_server import settings
from greenquery_server.adapters.base import AVAILABLE, EXPERIMENTAL, RunSpec, SlurmJob, Step, SystemAdapter
from greenquery_server.catalog import WORKLOADS

SCRIPTS = "evaluation/scripts"


def embedding_choice(model: str | None = None) -> str:
    """Embedding model ReasonDB will use for table metadata (recorded per run).

    Local-server models stay fully offline: in ``auto`` mode they use the local
    embedding model even when an OpenAI key is configured.
    """
    choice = os.environ.get("GREENQUERY_EMBEDDING_MODEL", "auto")
    if choice == "auto":
        from reasondb.reasoning.providers import get_model_spec

        spec = get_model_spec(model) if model is not None else None
        if spec is not None and spec.provider == "openai_compatible" and not spec.remote:
            return "local"
        choice = "openai" if os.environ.get("OPENAI_API_KEY") else "local"
    return choice


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def raw_root(job_dir: Path) -> Path:
    return job_dir / "raw"


def run_dir_for(job_dir: Path, model: str, repetition: int) -> Path:
    return raw_root(job_dir) / safe_name(model) / f"run_{repetition:02d}"


def compact_path(job_dir: Path) -> Path:
    return job_dir / "compact.json"


def latest_measurement(run_dir: Path, workload: str) -> Path:
    candidates = sorted(
        path for path in run_dir.glob(f"{workload}_*.json") if not path.name.endswith("_summary.json")
    )
    if not candidates:
        raise FileNotFoundError(f"No {workload} measurement in {run_dir}")
    return candidates[-1]


def gpus_needed(models: List[str]) -> int:
    """GPUs for the local models of a run (registry `gpus`); API models need none."""
    from reasondb.reasoning.providers import get_model_spec

    counts = [
        spec.gpus for spec in (get_model_spec(model) for model in models)
        if spec is not None and spec.provider == "openai_compatible" and not spec.remote
    ]
    return max(counts) if counts else 0


class StrettoAdapter(SystemAdapter):
    # The DGX evaluation script covers the text workloads; Medical stays local/API-only.
    workloads = ["movie", "rotowire"] if settings.EXECUTOR == "slurm" else list(WORKLOADS)
    multi_model = True

    def __init__(self, optimizer: str = "stretto"):
        self.optimizer = optimizer
        if optimizer == "stretto":
            self.key, self.label, self.status = "stretto", "Stretto", AVAILABLE
            self.description = "ReasonDB/Stretto with its gradient-descent plan optimizer."
        elif optimizer == "abacus":
            self.key, self.label, self.status = "stretto-abacus", "Abacus in Stretto", AVAILABLE
            self.description = "Palimpzest's Abacus (Pareto cascades) optimizer re-implemented inside ReasonDB."
            self.notes = ("Same executor and operators as Stretto. Use multi-model mode to give it choices; "
                          "with one model it picks the same plan as Stretto.")
        elif optimizer == "lotus":
            self.key, self.label, self.status = "stretto-lotus", "Lotus in Stretto", EXPERIMENTAL
            self.description = "Lotus-style proxy/oracle optimizer re-implemented inside ReasonDB."
            self.notes = ("Needs proxy confidence scores, which the API/Ollama text backend does not provide: "
                          "fails with API and Ollama models. Use the real LOTUS system instead.")
        else:
            raise ValueError(optimizer)

    # ------------------------------------------------------------------ steps
    def _env(self, spec: RunSpec, repetition: int) -> dict:
        return {
            "GREENQUERY_OPTIMIZER": self.optimizer,
            "GREENQUERY_EMBEDDING_MODEL": embedding_choice(spec.model),
            # Validators expect the online_<workload>_ prefix used by the repeat scripts.
            "GREENQUERY_RUN_NAMESPACE": f"online_{spec.workload}_{safe_name(spec.job_id)}_{repetition:02d}",
        }

    def steps(self, spec: RunSpec, run_dir: Path, repetition: int) -> List[Step]:
        py = settings.PYTHON_BIN
        env = self._env(spec, repetition)
        cache = ["--cold-cache", "--online-cache-policy", "disabled"] if spec.cache_policy == "disabled" else []
        measurement = f"$LATEST:{spec.workload}:{run_dir}"
        if spec.workload == "movie":
            run = [
                py, f"{SCRIPTS}/run_movie_measurement.py",
                "--query-set", "demo", "--demo-q4-mode", "ratio", "--backend-profile", "api",
                "--model-name", spec.model, "--split-artifacts",
                "--run-namespace", env["GREENQUERY_RUN_NAMESPACE"], "--output-dir", str(run_dir),
                *cache,
            ]
            validate = [py, f"{SCRIPTS}/validate_online_movie_run.py", measurement, f"$SUMMARY:{measurement}", "--model", spec.model]
            if spec.query:
                run += ["--demo-query", WORKLOADS["movie"].query_flags[spec.query]]
                validate += ["--only-query", spec.query]
            return [
                Step("run", run, env),
                Step("summarize", [py, f"{SCRIPTS}/build_movie_demo_summary.py", measurement]),
                Step("validate", validate, fatal=False, validation=True),
            ]
        if spec.workload == "medical":
            return [
                Step("run", [
                    py, f"{SCRIPTS}/run_medical_measurement.py", "--backend-profile", "api",
                    "--model-name", spec.model, "--output-dir", str(run_dir), "--split-artifacts", *cache,
                ], env),
                Step("summarize", [py, f"{SCRIPTS}/build_medical_demo_summary.py", measurement]),
                Step("validate", [py, f"{SCRIPTS}/validate_online_medical_run.py", measurement, "--model", spec.model],
                     fatal=False, validation=True),
            ]
        if spec.workload == "rotowire":
            return [
                Step("run", [
                    py, f"{SCRIPTS}/run_rotowire_measurement.py", "--backend-profile", "api",
                    "--model-name", spec.model, "--output-dir", str(run_dir), "--split-artifacts", *cache,
                ], env),
                Step("validate", [py, f"{SCRIPTS}/validate_online_rotowire_run.py", measurement, "--model", spec.model],
                     fatal=False, validation=True),
            ]
        raise ValueError(f"Unsupported workload {spec.workload}")

    def finalize_steps(self, spec: RunSpec, job_dir: Path) -> List[Step]:
        builder = {
            "movie": "build_online_movie_query_metrics.py",
            "medical": "build_online_medical_query_metrics.py",
            "rotowire": "build_online_rotowire_query_metrics.py",
        }[spec.workload]
        return [Step("compact", [settings.PYTHON_BIN, f"{SCRIPTS}/{builder}", str(raw_root(job_dir)), str(compact_path(job_dir))])]

    # ------------------------------------------------------------------ slurm
    def slurm_job(self, spec: RunSpec, out_dir: Path) -> SlurmJob:
        if spec.workload == "movie":
            flags = WORKLOADS["movie"].query_flags
            units = [f"movie_{flags[spec.query]}"] if spec.query else [f"movie_{flag}" for flag in flags.values()]
        elif spec.workload == "rotowire":
            units = ["rotowire"]  # R1 and R2 run together in one process
        else:
            raise ValueError(f"{spec.workload} is not supported by the DGX evaluation script")
        operator_models = [m.strip() for m in (spec.operator_models or "").split(",") if m.strip()]
        gpus = gpus_needed([spec.model, *operator_models])
        return SlurmJob(
            script="evaluation/scripts/run_dgx_eval.sbatch",
            env={
                "MODELS": spec.model,
                "UNITS": " ".join(units),
                "REPETITIONS": str(spec.repetitions),
                "GREENQUERY_OPTIMIZER": self.optimizer,
                "GREENQUERY_OPERATOR_MODELS": ",".join(operator_models),
            },
            gres=f"gpu:{gpus}" if gpus else "none",
        )

    def collect_slurm(self, spec: RunSpec, out_dir: Path) -> List[dict]:
        path = out_dir / "all_query_metrics.json"
        records = json.loads(path.read_text()) if path.exists() else []
        if spec.query:
            records = [record for record in records if record.get("query") == spec.query]
        return records

    def collect(self, spec: RunSpec, job_dir: Path) -> List[dict]:
        path = compact_path(job_dir)
        records = json.loads(path.read_text()) if path.exists() else []
        if spec.query:
            records = [record for record in records if record.get("query") == spec.query]
        return records
