#!/bin/bash
# Usage: sbatch run_movie_demo_repeated.sbatch MODEL_NAME [REPETITIONS] [OUTPUT_ROOT]
# Runs the four fixed Movie demo queries repeatedly with all caches disabled.
# Each repetition keeps its own measurement JSON, predictions, and summary.
#
# Example:
#   sbatch evaluation/scripts/run_movie_demo_repeated.sbatch Qwen/Qwen2.5-7B-Instruct 10
#
# Llama should be submitted with an afterok dependency on the Qwen job.
# This prevents two local text-QA servers from competing for the same host.
#
#SBATCH --job-name=movie-demo-repeat
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

MODEL_NAME=${1:?pass Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct, or meta-llama/Llama-3.1-8B-Instruct}
REPETITIONS=${2:-10}
OUTPUT_ROOT=${3:-evaluation/demo_suites/matched_planner/movie}
case "$MODEL_NAME" in
  Qwen/Qwen2.5-7B-Instruct) MODEL_LABEL=qwen; LOCAL_TEXT_QA_PORT=5013 ;;
  Qwen/Qwen2.5-14B-Instruct) MODEL_LABEL=qwen14; LOCAL_TEXT_QA_PORT=5014; DIRECT_QA_BATCH_SIZE=4 ;;
  meta-llama/Llama-3.1-8B-Instruct) MODEL_LABEL=llama; LOCAL_TEXT_QA_PORT=5010 ;;
  *) echo "unsupported model: $MODEL_NAME" >&2; exit 2 ;;
esac
[[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]] || { echo "repetitions must be a positive integer" >&2; exit 2; }

REPO_ROOT=${GREENQUERY_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RUN_ROOT="${OUTPUT_ROOT%/}/${MODEL_LABEL}_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"
export GREENQUERY_RUN_NAMESPACE="movie_${MODEL_LABEL}_${SLURM_JOB_ID}"

export PYTHON_BIN=/home/asharma/greenQueryLab-phase4/.venv/bin/python
export LOCAL_GPU_ID=0
export LOCAL_MODEL_NAME="$MODEL_NAME"
export LOCAL_TEXT_QA_PORT
export LOCAL_TEXT_EMBED_PORT=$((20000 + SLURM_JOB_ID % 20000))
export GREENQUERY_TEXT_EMBED_PORT="$LOCAL_TEXT_EMBED_PORT"
export GREENQUERY_GPU_INDEX=0
export GREENQUERY_LOCAL_TEXT_MODEL="$MODEL_NAME"
export GREENQUERY_DIRECT_QA_BATCH_SIZE="${DIRECT_QA_BATCH_SIZE:-128}"
export GREENQUERY_CO2_G_PER_KWH=344
export KILL_SCRIPT="/tmp/kill_movie_demo_repeat_${MODEL_LABEL}_${SLURM_JOB_ID}.sh"

GPU_CHECK_ID=${CUDA_VISIBLE_DEVICES%%,*}
if nvidia-smi --id="${GPU_CHECK_ID:-0}" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "Allocated GPU already has a compute process; refusing to share it." >&2
  exit 1
fi

scripts/start_dgx_text_servers.sh
trap '"$KILL_SCRIPT" || true' EXIT

for iteration in $(seq -w 1 "$REPETITIONS"); do
  ITERATION_DIR="$RUN_ROOT/run_${iteration}"
  mkdir -p "$ITERATION_DIR"
  echo "START iteration=${iteration} model=${MODEL_NAME}"
  "$PYTHON_BIN" evaluation/scripts/run_movie_measurement.py \
    --query-set demo \
    --cold-cache \
    --matched-planner \
    --split-artifacts \
    --demo-q4-mode ratio \
    --operator-policy semantic-text-only \
    --backend-profile dgx-local-text \
    --device cuda:0 \
    --model-provider local \
    --model-name "$MODEL_NAME" \
    --model-type dgx \
    --energy-measurement nvml \
    --local-cache-policy disabled \
    --model-cache-state disabled \
    --planner-cache-state disabled \
    --run-namespace "$GREENQUERY_RUN_NAMESPACE" \
    --output-dir "$ITERATION_DIR"
  MEASUREMENT=$(find "$ITERATION_DIR" -maxdepth 1 -type f -name 'movie_*.json' | sort | tail -n 1)
  "$PYTHON_BIN" evaluation/scripts/build_movie_demo_summary.py "$MEASUREMENT"
  "$PYTHON_BIN" evaluation/scripts/validate_matched_demo.py "$MEASUREMENT" --model "$MODEL_NAME" --demo movie
  echo "DONE iteration=${iteration} measurement=${MEASUREMENT}"
done

"$PYTHON_BIN" evaluation/scripts/build_matched_demo_summary.py "${OUTPUT_ROOT%/}/.."
