#!/bin/bash
# Evaluation data on the DGX: Stretto with local Ollama models, measured GPU energy.
#
#   sbatch --gres=gpu:1 evaluation/scripts/run_dgx_eval.sbatch
#   sbatch --gres=gpu:2 --export=ALL,MODELS="ollama/llama3.3:70b" evaluation/scripts/run_dgx_eval.sbatch
#
# Optimizer: GREENQUERY_OPTIMIZER=stretto|abacus|lotus (default stretto).
# Operator mode: single-model by default (MODEL plans and runs every operator).
# Multi-model (for comparing optimizers): GREENQUERY_OPERATOR_MODELS lists the
# alternative operator models, cheapest first; MODEL then only plans, e.g.
#   sbatch --export=ALL,MODELS="ollama/qwen2.5:32b-instruct",GREENQUERY_OPTIMIZER=lotus,\
#     GREENQUERY_OPERATOR_MODELS="ollama/qwen2.5:3b-instruct,ollama/qwen2.5:32b-instruct" ...
#
# Per model: load it, measure idle power with the model resident (60 s), then run
# REPETITIONS x (Movie Q1..Q4 each as its own process, RotoWire R1+R2) with a cold
# LLM cache. A query whose plan fails is recorded as failed and the job moves on;
# Stretto itself is not modified. GPU energy comes from both the nvidia-smi sampler
# and the NVML energy counter (GREENQUERY_ENERGY_COUNTERS=1). No root needed.
#
# Output: evaluation/demo_suites/dgx_ollama/<job id>/
#   raw/<unit>/<model>/run_NN/   measurements, summaries, logs
#   idle/<model>.json            idle baseline with the model loaded
#   status.tsv                   one row per (model, repetition, unit)
#   <unit>_query_metrics.json    compact records per unit; all_query_metrics.json merged
#
#SBATCH --job-name=gq-eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

REPO_ROOT=${GREENQUERY_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}
OLLAMA_HOME=${OLLAMA_HOME:-$HOME/ollama}
export OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/ollama-models}
MODELS=${MODELS:-"ollama/qwen2.5:3b-instruct ollama/llama3.2:3b ollama/qwen2.5:7b-instruct ollama/llama3.1:8b ollama/qwen2.5:32b-instruct"}
REPETITIONS=${REPETITIONS:-3}
UNITS=${UNITS:-"movie_Q1 movie_Q2 movie_Q3 movie_Q4 rotowire"}

# OUT_DIR lets greenquery_server collect results from its own job folder.
OUT=${OUT_DIR:-"evaluation/demo_suites/dgx_ollama/${SLURM_JOB_ID}"}
mkdir -p "$OUT/idle" logs

GPU_CHECK_ID=${CUDA_VISIBLE_DEVICES:-}; GPU_CHECK_ID=${GPU_CHECK_ID%%,*}
if nvidia-smi --id="${GPU_CHECK_ID:-0}" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "Allocated GPU already has a compute process; refusing to share it." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.draw,power.limit --format=csv | tee "$OUT/gpu.csv"
GPU_NAME=$(sed -n 2p "$OUT/gpu.csv" | cut -d, -f2 | xargs)
GPU_COUNT=$(($(wc -l < "$OUT/gpu.csv") - 1))
"$PYTHON_BIN" -m evaluation.measurement.energy_counters probe > "$OUT/probe.json" 2>&1
git rev-parse HEAD > "$OUT/git_commit.txt" 2>/dev/null; git status --short > "$OUT/git_status.txt" 2>/dev/null
printf "optimizer=%s\noperator_models=%s\nmodels=%s\nrepetitions=%s\nunits=%s\n" \
  "${GREENQUERY_OPTIMIZER:-stretto}" "${GREENQUERY_OPERATOR_MODELS:-<single-model>}" "$MODELS" "$REPETITIONS" "$UNITS" > "$OUT/run_config.txt"

OLLAMA_PORT=$((30000 + SLURM_JOB_ID % 20000))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE=2h
export GREENQUERY_OLLAMA_BASE_URL="http://127.0.0.1:${OLLAMA_PORT}/v1"
export LD_LIBRARY_PATH="$OLLAMA_HOME/lib/ollama:${LD_LIBRARY_PATH:-}"
"$OLLAMA_HOME/bin/ollama" serve > "$OUT/ollama-serve.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
for i in $(seq 1 60); do curl -s "http://${OLLAMA_HOST}" >/dev/null && break; sleep 1; done

export GREENQUERY_GPU_INDEX=$(seq -s, 0 $((GPU_COUNT - 1)))
export GREENQUERY_ENERGY_COUNTERS=1
export GREENQUERY_EMBEDDING_MODEL=local
export GREENQUERY_CO2_G_PER_KWH=${GREENQUERY_CO2_G_PER_KWH:-344}

safe() { echo "$1" | sed 's/[^A-Za-z0-9]/_/g'; }
printf "model\trepetition\tunit\tstatus\tvalidation\tdetail\n" > "$OUT/status.tsv"

run_unit() {  # $1 model, $2 repetition (2 digits), $3 unit
  local model=$1 rep=$2 unit=$3 dir log meas detail
  dir="$OUT/raw/$unit/$(safe "$model")/run_$rep"
  mkdir -p "$dir"; log="$dir/run.log"
  local energy=nvml
  case "$model" in ollama/*) ;; *) energy=none ;; esac   # API models: EcoLogits, not this GPU
  local common=(--model-name "$model" --backend-profile api --cold-cache --online-cache-policy disabled
                --split-artifacts --energy-measurement "$energy" --output-dir "$dir")
  local ns="eval_$(safe "$model")_${unit}_${rep}_${SLURM_JOB_ID}"
  if [[ $unit == movie_* ]]; then
    local q=${unit#movie_}
    # GPU metadata flags exist only in the Movie runner.
    "$PYTHON_BIN" evaluation/scripts/run_movie_measurement.py --query-set demo --demo-q4-mode ratio \
      --demo-query "$q" --run-namespace "$ns" --gpu-model "$GPU_NAME" --gpu-count "$GPU_COUNT" \
      --hardware-provider dgx "${common[@]}" > "$log" 2>&1
  else
    GREENQUERY_RUN_NAMESPACE="online_rotowire_${ns}" "$PYTHON_BIN" evaluation/scripts/run_rotowire_measurement.py \
      "${common[@]}" > "$log" 2>&1
  fi
  local code=$?
  meas=$(find "$dir" -maxdepth 1 -name "${unit%%_*}_*.json" ! -name '*_summary.json' 2>/dev/null | sort | tail -1)
  if [[ $code -ne 0 || -z "$meas" ]]; then
    detail=$(grep -E 'Error|Exception|DeadEnd' "$log" | tail -1 | tr '\t' ' ' | cut -c1-160)
    printf "%s\t%s\t%s\tfailed\t-\t%s\n" "$model" "$rep" "$unit" "$detail" >> "$OUT/status.tsv"
    mv "$dir" "$(dirname "$dir")/failed_run_$rep"   # keep the log; builders only read run_* dirs
    return
  fi
  local validation=validated
  if [[ $unit == movie_* ]]; then
    "$PYTHON_BIN" evaluation/scripts/build_movie_demo_summary.py "$meas" >> "$log" 2>&1
    local query_name
    query_name=$(case ${unit#movie_} in Q1) echo Q1_positive_reviews;; Q2) echo Q2_positive_reviews_for_taken_3;;
                 Q3) echo Q3_reviews_same_sentiment;; Q4) echo Q4_positive_reviews_for_taken_3_ratio;; esac)
    "$PYTHON_BIN" evaluation/scripts/validate_online_movie_run.py "$meas" "${meas%.json}_summary.json" \
      --model "$model" --only-query "$query_name" >> "$log" 2>&1 || validation=validation_failed
  else
    "$PYTHON_BIN" evaluation/scripts/validate_online_rotowire_run.py "$meas" --model "$model" >> "$log" 2>&1 \
      || validation=validation_failed
  fi
  printf "%s\t%s\t%s\tsucceeded\t%s\t-\n" "$model" "$rep" "$unit" "$validation" >> "$OUT/status.tsv"
}

for MODEL in $MODELS; do
  API_MODEL=${MODEL#ollama/}
  if [[ $MODEL == ollama/* ]]; then
    # Load the planner and, in multi-model mode, every operator model before the
    # idle baseline so no query pays for model loading.
    for LOAD in $MODEL ${GREENQUERY_OPERATOR_MODELS:+${GREENQUERY_OPERATOR_MODELS//,/ }}; do
      echo "=== load $LOAD $(date)"
      curl -s "http://${OLLAMA_HOST}/api/generate" -d "{\"model\": \"${LOAD#ollama/}\", \"prompt\": \"ok\", \"stream\": false}" > /dev/null
    done
    sleep 10
    "$PYTHON_BIN" -m evaluation.measurement.energy_counters idle --seconds 60 --out "$OUT/idle/$(safe "$MODEL").json" > /dev/null
    export GREENQUERY_IDLE_BASELINE="$REPO_ROOT/$OUT/idle/$(safe "$MODEL").json" GREENQUERY_ENERGY_COUNTERS=1 GREENQUERY_EMBEDDING_MODEL=local
  else
    unset GREENQUERY_IDLE_BASELINE
    export GREENQUERY_ENERGY_COUNTERS=0 GREENQUERY_EMBEDDING_MODEL=openai
  fi
  for rep in $(seq -w 1 "$REPETITIONS"); do
    for unit in $UNITS; do
      echo "--- $MODEL rep=$rep $unit $(date)"
      run_unit "$MODEL" "$rep" "$unit"
    done
  done
  [[ $MODEL == ollama/* ]] && curl -s "http://${OLLAMA_HOST}/api/generate" -d "{\"model\": \"${API_MODEL}\", \"keep_alive\": 0}" > /dev/null
done

# Compact records per unit, then merged.
for unit in $UNITS; do
  root="$OUT/raw/$unit"
  [[ -d "$root" ]] || continue
  find "$root" -mindepth 1 -maxdepth 1 -type d -empty -delete
  case $unit in
    movie_*) builder=build_online_movie_query_metrics.py ;;
    rotowire) builder=build_online_rotowire_query_metrics.py ;;
  esac
  "$PYTHON_BIN" evaluation/scripts/$builder "$root" "$OUT/${unit}_query_metrics.json" || echo "compact build failed for $unit"
done
"$PYTHON_BIN" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
merged = []
for path in sorted(out.glob("*_query_metrics.json")):
    if path.name != "all_query_metrics.json":
        merged += json.loads(path.read_text())
(out / "all_query_metrics.json").write_text(json.dumps(merged, indent=2) + "\n")
print(f"{len(merged)} compact records")
EOF
echo "done $(date)"
