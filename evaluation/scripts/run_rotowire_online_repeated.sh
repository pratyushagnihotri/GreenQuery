#!/usr/bin/env bash
# Run validated, application-cache-disabled RotoWire repetitions for one OpenAI model.
set -euo pipefail

MODEL="${1:?Usage: $0 MODEL RAW_ROOT [REPETITIONS]}"
RAW_ROOT="${2:?Usage: $0 MODEL RAW_ROOT [REPETITIONS]}"
REPETITIONS="${3:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${RAW_ROOT}/${MODEL//\//_}"

for RUN in $(seq -w 1 "$REPETITIONS"); do
  RUN_DIR="${MODEL_DIR}/run_${RUN}"
  mkdir -p "$RUN_DIR"
  NAMESPACE="online_rotowire_${MODEL//[^A-Za-z0-9]/_}_${RUN}"
  GREENQUERY_RUN_NAMESPACE="$NAMESPACE" "$PYTHON_BIN" evaluation/scripts/run_rotowire_measurement.py \
    --backend-profile api --model-name "$MODEL" --output-dir "$RUN_DIR" --cold-cache \
    --online-cache-policy disabled --split-artifacts
  MEASUREMENT=$(find "$RUN_DIR" -maxdepth 1 -type f -name 'rotowire_*.json' ! -name '*_summary.json' | sort | tail -n 1)
  "$PYTHON_BIN" evaluation/scripts/validate_online_rotowire_run.py "$MEASUREMENT" --model "$MODEL"
done
