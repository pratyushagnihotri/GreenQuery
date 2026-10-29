#!/usr/bin/env bash
# Run validated, application-cache-disabled online demo repetitions.
set -euo pipefail

DEMO="${1:?Usage: $0 DEMO MODEL RAW_ROOT [REPETITIONS]}"
MODEL="${2:?Usage: $0 DEMO MODEL RAW_ROOT [REPETITIONS]}"
RAW_ROOT="${3:?Usage: $0 DEMO MODEL RAW_ROOT [REPETITIONS]}"
REPETITIONS="${4:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${RAW_ROOT}/${MODEL//\//_}"

for RUN in $(seq -w 1 "$REPETITIONS"); do
  RUN_DIR="${MODEL_DIR}/run_${RUN}"
  mkdir -p "$RUN_DIR"
  NAMESPACE="online_${DEMO}_${MODEL//[^A-Za-z0-9]/_}_${RUN}"

  case "$DEMO" in
    medical)
      GREENQUERY_RUN_NAMESPACE="$NAMESPACE" "$PYTHON_BIN" evaluation/scripts/run_medical_measurement.py \
        --backend-profile api --model-name "$MODEL" --output-dir "$RUN_DIR" --cold-cache \
        --online-cache-policy disabled --split-artifacts
      MEASUREMENT=$(find "$RUN_DIR" -maxdepth 1 -type f -name 'medical_*.json' ! -name '*_summary.json' | sort | tail -n 1)
      "$PYTHON_BIN" evaluation/scripts/build_medical_demo_summary.py "$MEASUREMENT"
      "$PYTHON_BIN" evaluation/scripts/validate_online_medical_run.py "$MEASUREMENT" --model "$MODEL"
      ;;
    movie)
      "$PYTHON_BIN" evaluation/scripts/run_movie_measurement.py \
        --query-set demo --demo-q4-mode ratio --backend-profile api \
        --model-provider openai --model-name "$MODEL" --cold-cache \
        --online-cache-policy disabled --split-artifacts --run-namespace "$NAMESPACE" \
        --output-dir "$RUN_DIR"
      MEASUREMENT=$(find "$RUN_DIR" -maxdepth 1 -type f -name 'movie_*.json' ! -name '*_summary.json' | sort | tail -n 1)
      "$PYTHON_BIN" evaluation/scripts/build_movie_demo_summary.py "$MEASUREMENT"
      SUMMARY="${MEASUREMENT%.json}_summary.json"
      "$PYTHON_BIN" evaluation/scripts/validate_online_movie_run.py "$MEASUREMENT" "$SUMMARY" --model "$MODEL"
      ;;
    rotowire)
      GREENQUERY_RUN_NAMESPACE="$NAMESPACE" "$PYTHON_BIN" evaluation/scripts/run_rotowire_measurement.py \
        --backend-profile api --model-name "$MODEL" --output-dir "$RUN_DIR" --cold-cache \
        --online-cache-policy disabled --split-artifacts
      MEASUREMENT=$(find "$RUN_DIR" -maxdepth 1 -type f -name 'rotowire_*.json' ! -name '*_summary.json' | sort | tail -n 1)
      "$PYTHON_BIN" evaluation/scripts/validate_online_rotowire_run.py "$MEASUREMENT" --model "$MODEL"
      ;;
    *)
      echo "Unknown demo: $DEMO" >&2
      exit 2
      ;;
  esac
done
