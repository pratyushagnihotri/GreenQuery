"""Server configuration from environment variables (and an optional repo-root .env)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Load KEY=VALUE lines without overriding variables already set in the shell."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()

DATA_DIR = Path(os.environ.get("GREENQUERY_DATA_DIR", REPO_ROOT / "server_data")).resolve()
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "greenquery.sqlite3"
ARCHIVE_DIR = REPO_ROOT / "dashboard" / "src" / "data"

# Interpreter used for runner subprocesses (the env with reasondb installed).
PYTHON_BIN = os.environ.get("GREENQUERY_PYTHON", sys.executable)

# Refuse runs whose estimated API cost exceeds this (USD, whole job incl. repetitions).
MAX_RUN_COST_USD = float(os.environ.get("GREENQUERY_MAX_RUN_COST_USD", "5"))
MAX_REPETITIONS = int(os.environ.get("GREENQUERY_MAX_REPETITIONS", "10"))
MAX_CONCURRENT_JOBS = int(os.environ.get("GREENQUERY_MAX_CONCURRENT_JOBS", "1"))

# local: run jobs as subprocesses of the server (laptop). slurm: submit each job with
# sbatch on a cluster login node (DGX) and stream its log.
EXECUTOR = os.environ.get("GREENQUERY_EXECUTOR", "local")
# SemBench checkout with per-system venvs (real LOTUS/Palimpzest/ThalamusDB/CAESURA).
SEMBENCH_ROOT = Path(os.environ.get("SEMBENCH_ROOT", REPO_ROOT / "third_party" / "sembench"))  # vendored SemBench
SLURM_POLL_SECONDS = float(os.environ.get("GREENQUERY_SLURM_POLL_SECONDS", "5"))
# Measured SemBench runs started from the dashboard (Challenge): idle-baseline length and per-query limit.
MEASURED_IDLE_SECONDS = int(os.environ.get("GREENQUERY_MEASURED_IDLE_SECONDS", "20"))
MEASURED_QUERY_TIMEOUT = int(os.environ.get("GREENQUERY_MEASURED_QUERY_TIMEOUT", "900"))
# Ollama model store used by the jobs (manifests decide which ollama/* models exist).
OLLAMA_MODELS_DIR = Path(os.environ.get("OLLAMA_MODELS", Path.home() / "ollama-models"))

CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "GREENQUERY_CORS_ORIGINS", "http://localhost:3000,http://localhost:3100"
    ).split(",")
    if origin.strip()
]

# Largest model (download size) a visitor may add from the dashboard; must fit one GPU.
CUSTOM_MODEL_MAX_BYTES = int(float(os.environ.get("GREENQUERY_CUSTOM_MODEL_MAX_GB", "20")) * 1e9)
