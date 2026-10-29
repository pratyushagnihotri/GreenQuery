#!/bin/bash
# Download Qwen 2.5 14B once through Slurm; this is not a measured benchmark.
#SBATCH --job-name=qwen25-14b-preload
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/asharma/greenQueryLab-phase4/.venv/bin/python}
"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download("Qwen/Qwen2.5-14B-Instruct")
print(f"Qwen 2.5 14B cached at: {path}")
PY
