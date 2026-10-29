#!/usr/bin/env bash
# Run validated, application-cache-disabled Movie repetitions for one OpenAI model.
set -euo pipefail

MODEL="${1:?Usage: $0 MODEL RAW_ROOT [REPETITIONS]}"
RAW_ROOT="${2:?Usage: $0 MODEL RAW_ROOT [REPETITIONS]}"
REPETITIONS="${3:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${RAW_ROOT}/${MODEL//\//_}"

for RUN in $(seq -w 1 "$REPETITIONS"); do
  RUN_DIR="${MODEL_DIR}/run_${RUN}"
  mkdir -p "$RUN_DIR"
  NAMESPACE="online_movie_${MODEL//[^A-Za-z0-9]/_}_${RUN}"
  "$PYTHON_BIN" evaluation/scripts/run_movie_measurement.py \
    --query-set demo --demo-q4-mode ratio --backend-profile api \
    --model-provider openai --model-name "$MODEL" --cold-cache \
    --online-cache-policy disabled --split-artifacts --run-namespace "$NAMESPACE" \
    --output-dir "$RUN_DIR"
  MEASUREMENT=$(find "$RUN_DIR" -maxdepth 1 -type f -name 'movie_*.json' ! -name '*_summary.json' | sort | tail -n 1)
  "$PYTHON_BIN" evaluation/scripts/build_movie_demo_summary.py "$MEASUREMENT"
  SUMMARY="${MEASUREMENT%.json}_summary.json"
  "$PYTHON_BIN" evaluation/scripts/validate_online_movie_run.py "$MEASUREMENT" "$SUMMARY" --model "$MODEL"
done
