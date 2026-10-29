#!/usr/bin/env bash
# Refresh Movie Q3 only, preserving Q1/Q2/Q4 from the existing collection.
set -euo pipefail

MODEL="${1:?Usage: $0 MODEL OUTPUT_ROOT [REPETITIONS]}"
OUTPUT_ROOT="${2:?Usage: $0 MODEL OUTPUT_ROOT [REPETITIONS]}"
REPETITIONS="${3:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${OUTPUT_ROOT}/${MODEL//\//_}"

for RUN in $(seq -w 1 "$REPETITIONS"); do
  RUN_DIR="${MODEL_DIR}/run_${RUN}"
  mkdir -p "$RUN_DIR"
  NAMESPACE="online_movie_q3_${MODEL//[^A-Za-z0-9]/_}_${RUN}"
  "$PYTHON_BIN" evaluation/scripts/run_movie_measurement.py \
    --query-set demo --demo-query Q3 --demo-q4-mode ratio --backend-profile api \
    --model-provider openai --model-name "$MODEL" --cold-cache \
    --online-cache-policy disabled --split-artifacts --run-namespace "$NAMESPACE" \
    --output-dir "$RUN_DIR"
  MEASUREMENT=$(find "$RUN_DIR" -maxdepth 1 -type f -name 'movie_*.json' ! -name '*_summary.json' | sort | tail -n 1)
  "$PYTHON_BIN" evaluation/scripts/build_movie_demo_summary.py "$MEASUREMENT"
  "$PYTHON_BIN" evaluation/scripts/validate_online_movie_run.py "$MEASUREMENT" "${MEASUREMENT%.json}_summary.json" --model "$MODEL" --only-query Q3_reviews_same_sentiment
done
