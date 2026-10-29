"""Interface every benchmarked system implements.

A run is a list of subprocess ``Step``s per repetition followed by ``collect``,
which returns records in the shared compact schema (the same fields the
dashboard reads from ``dashboard/src/data/*.json``). Running systems as
subprocesses keeps their dependencies in separate environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

AVAILABLE = "available"
EXPERIMENTAL = "experimental"
PLANNED = "planned"


@dataclass(frozen=True)
class RunSpec:
    job_id: str
    system: str
    workload: str
    model: str
    query: Optional[str] = None
    repetitions: int = 1
    cache_policy: str = "disabled"
    # Multi-model mode: comma-separated operator models, cheapest first (None = single-model).
    operator_models: Optional[str] = None


@dataclass
class Step:
    name: str
    argv: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    # Non-fatal steps (validators) record their outcome instead of failing the job.
    fatal: bool = True
    # Set on validator steps so the job can report validation status.
    validation: bool = False


@dataclass
class SlurmJob:
    """One sbatch submission. Values go through the environment (sbatch --export=ALL)."""

    script: str
    env: Dict[str, str]
    gres: str = "none"
    out_dir_env: str = "OUT_DIR"


class SystemAdapter(ABC):
    key: str
    label: str
    description: str
    status: str
    workloads: List[str]
    notes: str = ""
    # Model providers this system can use, and whether multi-model mode is supported.
    providers: List[str] = ["openai", "anthropic", "openai_compatible"]
    multi_model: bool = False

    def to_public_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "status": self.status,
            "workloads": self.workloads,
            "notes": self.notes,
            "providers": self.providers,
            "multi_model": self.multi_model,
        }

    def slurm_job(self, spec: RunSpec, out_dir: Path) -> Optional[SlurmJob]:
        """Submission for cluster execution; None if the system cannot run under SLURM."""
        return None

    def collect_slurm(self, spec: RunSpec, out_dir: Path) -> List[dict]:
        """Compact records produced by a finished SLURM job."""
        return []

    @abstractmethod
    def steps(self, spec: RunSpec, run_dir: Path, repetition: int) -> List[Step]:
        """Subprocess steps producing raw measurements for one repetition."""

    @abstractmethod
    def finalize_steps(self, spec: RunSpec, job_dir: Path) -> List[Step]:
        """Steps run once after all repetitions (e.g. compact builders)."""

    @abstractmethod
    def collect(self, spec: RunSpec, job_dir: Path) -> List[dict]:
        """Compact records for the finished job."""
