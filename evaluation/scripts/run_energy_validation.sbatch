#!/bin/bash
# Usage: sbatch evaluation/scripts/run_energy_validation.sbatch [MODEL_NAME] [REPETITIONS] [OUTPUT_ROOT]
#
# Validates the energy measurement before any model-size sweep:
#   1. probe which counters are readable (NVML energy counter, RAPL, IPMI);
#   2. measure idle power with the model server loaded (GREENQUERY_IDLE_BASELINE);
#   3. run the four Movie demo queries REPETITIONS times in two passes:
#        A = nvidia-smi sampler + energy counters   (--energy-measurement nvml)
#        B = energy counters only                    (--energy-measurement none)
#      alternating A/B order per repetition so thermal drift does not bias one pass;
#   4. build compact records and compare the methods
#      (evaluation/scripts/compare_energy_methods.py).
#
# CPU energy: RAPL package counters cover the whole CPU socket (other jobs would be
# counted) and on our DGX energy_uj is root-only, so CPU/DRAM energy is reported as
# unavailable there. GPU energy (sampler + NVML counter) works without root.
# Do not add --exclusive on the shared single-node DGX: it would wait indefinitely.
#
#SBATCH --job-name=energy-validation
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

MODEL_NAME=${1:-Qwen/Qwen2.5-7B-Instruct}
REPETITIONS=${2:-5}
OUTPUT_ROOT=${3:-evaluation/demo_suites/energy_validation}
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

export PYTHON_BIN=${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}
export LOCAL_GPU_ID=0
export LOCAL_MODEL_NAME="$MODEL_NAME"
export LOCAL_TEXT_QA_PORT
export LOCAL_TEXT_EMBED_PORT=$((20000 + SLURM_JOB_ID % 20000))
export GREENQUERY_TEXT_EMBED_PORT="$LOCAL_TEXT_EMBED_PORT"
export GREENQUERY_GPU_INDEX=0
export GREENQUERY_LOCAL_TEXT_MODEL="$MODEL_NAME"
export GREENQUERY_DIRECT_QA_BATCH_SIZE="${DIRECT_QA_BATCH_SIZE:-128}"
export GREENQUERY_CO2_G_PER_KWH=${GREENQUERY_CO2_G_PER_KWH:-344}
export GREENQUERY_ENERGY_COUNTERS=1
export KILL_SCRIPT="/tmp/kill_energy_validation_${MODEL_LABEL}_${SLURM_JOB_ID}.sh"

GPU_CHECK_ID=${CUDA_VISIBLE_DEVICES%%,*}
if nvidia-smi --id="${GPU_CHECK_ID:-0}" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "Allocated GPU already has a compute process; refusing to share it." >&2
  exit 1
fi

# 1. What can this node measure?
"$PYTHON_BIN" -m evaluation.measurement.energy_counters probe | tee "$RUN_ROOT/probe.json"
nvidia-smi --query-gpu=index,name,driver_version,power.limit --format=csv | tee "$RUN_ROOT/gpu.csv"

# Optional whole-node reference from the BMC (needs ipmitool and permission).
IPMI_PID=""
if command -v ipmitool >/dev/null && ipmitool dcmi power reading >/dev/null 2>&1; then
  ( echo "timestamp,watts"
    while true; do
      watts=$(ipmitool dcmi power reading 2>/dev/null | awk -F: '/Instantaneous power reading/ {gsub(/[^0-9]/,"",$2); print $2}')
      echo "$(date +%s.%N),${watts}"
      sleep 1
    done ) > "$RUN_ROOT/ipmi_power.csv" &
  IPMI_PID=$!
else
  echo "ipmitool dcmi power reading unavailable; skipping node-level reference" | tee "$RUN_ROOT/ipmi_unavailable.txt"
fi

scripts/start_dgx_text_servers.sh
trap '"$KILL_SCRIPT" || true; [[ -n "$IPMI_PID" ]] && kill "$IPMI_PID" 2>/dev/null || true' EXIT

# 2. Idle baseline with the model loaded and nothing running.
sleep 30
"$PYTHON_BIN" -m evaluation.measurement.energy_counters idle --seconds 60 --out "$RUN_ROOT/idle.json"
export GREENQUERY_IDLE_BASELINE="$REPO_ROOT/$RUN_ROOT/idle.json"

run_pass() {  # $1 = pass label, $2 = --energy-measurement value, $3 = iteration
  local dir="$RUN_ROOT/$1/run_$3"
  mkdir -p "$dir"
  echo "START pass=$1 iteration=$3"
  "$PYTHON_BIN" evaluation/scripts/run_movie_measurement.py \
    --query-set demo --cold-cache --matched-planner --split-artifacts --demo-q4-mode ratio \
    --operator-policy semantic-text-only --backend-profile dgx-local-text --device cuda:0 \
    --model-provider local --model-name "$MODEL_NAME" --model-type dgx \
    --energy-measurement "$2" --local-cache-policy disabled --model-cache-state disabled \
    --planner-cache-state disabled \
    --run-namespace "energy_${MODEL_LABEL}_${SLURM_JOB_ID}_$1_$3" --output-dir "$dir"
  local measurement
  measurement=$(find "$dir" -maxdepth 1 -type f -name 'movie_*.json' ! -name '*_summary.json' | sort | tail -n 1)
  "$PYTHON_BIN" evaluation/scripts/build_movie_demo_summary.py "$measurement"
  echo "DONE pass=$1 iteration=$3 measurement=$measurement"
}

# 3. Alternate pass order so drift affects both passes equally.
for iteration in $(seq -w 1 "$REPETITIONS"); do
  if (( 10#$iteration % 2 )); then
    run_pass sampler_and_counters nvml "$iteration"; run_pass counters_only none "$iteration"
  else
    run_pass counters_only none "$iteration"; run_pass sampler_and_counters nvml "$iteration"
  fi
done

# 4. Compact records and method comparison.
"$PYTHON_BIN" evaluation/scripts/build_matched_demo_summary.py "$RUN_ROOT" --output "$RUN_ROOT/query_metrics.json"
"$PYTHON_BIN" evaluation/scripts/compare_energy_methods.py "$RUN_ROOT/query_metrics.json" \
  --json "$RUN_ROOT/energy_method_comparison.json" | tee "$RUN_ROOT/energy_method_comparison.md"
