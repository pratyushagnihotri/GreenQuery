#!/bin/bash
# Usage: sbatch run_rotowire_demo_repeated.sbatch MODEL_NAME [REPETITIONS] [OUTPUT_ROOT]
# Runs both ground-truth RotoWire queries repeatedly with the same local model planning and executing.
#SBATCH --job-name=rotowire-matched-repeat
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail
MODEL_NAME=${1:?pass Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct, or meta-llama/Llama-3.1-8B-Instruct}
REPETITIONS=${2:-10}
OUTPUT_ROOT=${3:-evaluation/demo_suites/matched_planner}
case "$MODEL_NAME" in
  Qwen/Qwen2.5-7B-Instruct) LABEL=qwen ;;
  Qwen/Qwen2.5-14B-Instruct) LABEL=qwen14; DIRECT_QA_BATCH_SIZE=4 ;;
  meta-llama/Llama-3.1-8B-Instruct) LABEL=llama ;;
  *) echo "unsupported model: $MODEL_NAME" >&2; exit 2 ;;
esac
REPO_ROOT=${GREENQUERY_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RUN_ROOT="$OUTPUT_ROOT/rotowire/${LABEL}_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"
export PYTHON_BIN=${PYTHON_BIN:-python3}
export LOCAL_GPU_ID=0 LOCAL_MODEL_NAME="$MODEL_NAME"
export LOCAL_TEXT_QA_PORT=$((10000 + SLURM_JOB_ID % 10000))
export GREENQUERY_LOCAL_TEXT_QA_PORT="$LOCAL_TEXT_QA_PORT"
export LOCAL_TEXT_EMBED_PORT=$((20000 + SLURM_JOB_ID % 20000))
export GREENQUERY_TEXT_EMBED_PORT="$LOCAL_TEXT_EMBED_PORT"
export GREENQUERY_GPU_INDEX=0 GREENQUERY_LOCAL_TEXT_MODEL="$MODEL_NAME" GREENQUERY_CO2_G_PER_KWH=344
export GREENQUERY_DIRECT_QA_BATCH_SIZE=${DIRECT_QA_BATCH_SIZE:-8}
export GREENQUERY_RUN_NAMESPACE="rotowire_matched_demo_${LABEL}_${SLURM_JOB_ID}"
export KILL_SCRIPT="/tmp/kill_rotowire_${LABEL}_${SLURM_JOB_ID}.sh"
GPU_CHECK_ID=${CUDA_VISIBLE_DEVICES%%,*}
if nvidia-smi --id="${GPU_CHECK_ID:-0}" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "Allocated GPU already has a compute process; refusing to share it." >&2
  exit 1
fi
scripts/start_dgx_text_servers.sh
trap '"$KILL_SCRIPT" || true' EXIT
for iteration in $(seq -w 1 "$REPETITIONS"); do
  dir="$RUN_ROOT/run_${iteration}"; mkdir -p "$dir"
  "$PYTHON_BIN" evaluation/scripts/run_rotowire_measurement.py --model-name "$MODEL_NAME" --output-dir "$dir" --cold-cache --matched-planner
  measurement=$(find "$dir" -maxdepth 1 -type f -name 'rotowire_*.json' | sort | tail -n 1)
  "$PYTHON_BIN" evaluation/scripts/validate_matched_demo.py "$measurement" --model "$MODEL_NAME" --demo rotowire
done
"$PYTHON_BIN" evaluation/scripts/build_matched_demo_summary.py "$OUTPUT_ROOT"
