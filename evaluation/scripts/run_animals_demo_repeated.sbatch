#!/bin/bash
# Usage: sbatch run_animals_demo_repeated.sbatch MODEL_NAME [REPETITIONS] [OUTPUT_ROOT]
# Start with REPETITIONS=1; submit ten repetitions only after both model preflights validate.
#SBATCH --job-name=animals-matched-repeat
#SBATCH --partition=batch
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G

set -euo pipefail

MODEL_NAME=${1:?pass Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct, or meta-llama/Llama-3.1-8B-Instruct}
REPETITIONS=${2:-10}
OUTPUT_ROOT=${3:-evaluation/demo_suites/matched_planner/animals}
case "$MODEL_NAME" in
  Qwen/Qwen2.5-7B-Instruct) LABEL=qwen ;;
  Qwen/Qwen2.5-14B-Instruct) LABEL=qwen14; DIRECT_QA_BATCH_SIZE=4 ;;
  meta-llama/Llama-3.1-8B-Instruct) LABEL=llama ;;
  *) echo "unsupported model: $MODEL_NAME" >&2; exit 2 ;;
esac
[[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]] || { echo "repetitions must be a positive integer" >&2; exit 2; }

REPO_ROOT=${GREENQUERY_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:?set PYTHON_BIN to the DGX project virtualenv Python}
[[ -x "$PYTHON_BIN" ]] || { echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2; exit 2; }
export PYTHON_BIN
"$PYTHON_BIN" evaluation/scripts/prepare_animals_dataset.py

export GREENQUERY_RUN_NAMESPACE="animals_matched_demo_${LABEL}_${SLURM_JOB_ID}"
export GREENQUERY_CO2_G_PER_KWH=344
export GREENQUERY_DIRECT_QA_BATCH_SIZE=${DIRECT_QA_BATCH_SIZE:-8}
export GREENQUERY_VISION_QA_BATCH_SIZE=8
export GREENQUERY_AUDIO_QA_BATCH_SIZE=8
# Animals aggregate configurations need a complete JSON response; 256 stays below
# the previous 512-token OOM setting while avoiding truncated planner output.
export GREENQUERY_PLANNER_MAX_NEW_TOKENS=256
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GREENQUERY_LOCAL_TEXT_MODEL="$MODEL_NAME"
export GREENQUERY_LOCAL_VISION_MODEL=llava-hf/llama3-llava-next-8b-hf
export GREENQUERY_LOCAL_AUDIO_MODEL=Qwen/Qwen2-Audio-7B-Instruct
export LOCAL_MODEL_NAME="$MODEL_NAME"
export LOCAL_TEXT_QA_PORT=$((40000 + SLURM_JOB_ID % 10000))
export GREENQUERY_LOCAL_TEXT_QA_PORT="$LOCAL_TEXT_QA_PORT"
export LOCAL_TEXT_EMBED_PORT=$((20000 + SLURM_JOB_ID % 10000))
export GREENQUERY_TEXT_EMBED_PORT="$LOCAL_TEXT_EMBED_PORT"
export GREENQUERY_LOCAL_VISION_PORT=$((30000 + SLURM_JOB_ID % 10000))
export GREENQUERY_LOCAL_AUDIO_PORT=$((10000 + SLURM_JOB_ID % 10000))
export KILL_SCRIPT="/tmp/kill_animals_${LABEL}_${SLURM_JOB_ID}.sh"
RUN_ROOT="${OUTPUT_ROOT%/}/${LABEL}_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"

IFS=',' read -r TEXT_GPU VISION_GPU AUDIO_GPU EXTRA <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
[[ -n "$TEXT_GPU" && -n "$VISION_GPU" && -n "$AUDIO_GPU" && -z "${EXTRA:-}" ]] || { echo "Animals requires exactly three allocated GPUs" >&2; exit 1; }
export LOCAL_GPU_ID="$TEXT_GPU"
export GREENQUERY_GPU_INDEX="$TEXT_GPU,$VISION_GPU,$AUDIO_GPU"
export GREENQUERY_TEXT_GPU_INDEX="$TEXT_GPU"
export GREENQUERY_VISION_GPU_INDEX="$VISION_GPU"
export GREENQUERY_AUDIO_GPU_INDEX="$AUDIO_GPU"
for GPU in "$TEXT_GPU" "$VISION_GPU" "$AUDIO_GPU"; do
  if nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "Allocated GPU $GPU already has a compute process; refusing to share it." >&2; exit 1
  fi
done

scripts/start_dgx_text_servers.sh
CUDA_VISIBLE_DEVICES="$VISION_GPU" "$PYTHON_BIN" reasondb/backends/kv_cache_image_qa_server.py \
  --model-name "$GREENQUERY_LOCAL_VISION_MODEL" --device-id 0 --port "$GREENQUERY_LOCAL_VISION_PORT" &
VISION_PID=$!
CUDA_VISIBLE_DEVICES="$AUDIO_GPU" "$PYTHON_BIN" reasondb/backends/kv_cache_audio_qa_server.py \
  --model-name "$GREENQUERY_LOCAL_AUDIO_MODEL" --device-id 0 --port "$GREENQUERY_LOCAL_AUDIO_PORT" &
AUDIO_PID=$!
printf '#!/bin/bash\nkill %s %s\n' "$VISION_PID" "$AUDIO_PID" >> "$KILL_SCRIPT"
chmod +x "$KILL_SCRIPT"
trap '"$KILL_SCRIPT" || true' EXIT

for SERVICE in "vision:$VISION_PID:$GREENQUERY_LOCAL_VISION_PORT" "audio:$AUDIO_PID:$GREENQUERY_LOCAL_AUDIO_PORT"; do
  IFS=: read -r NAME PID PORT <<< "$SERVICE"
  for attempt in $(seq 1 120); do
    if "$PYTHON_BIN" - "$PORT" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import urlopen
urlopen(f"http://127.0.0.1:{sys.argv[1]}/status", timeout=2)
PY
    then break; fi
    kill -0 "$PID" 2>/dev/null || { echo "$NAME service exited before readiness" >&2; exit 1; }
    sleep 5
    [[ "$attempt" -lt 120 ]] || { echo "$NAME service did not become ready" >&2; exit 1; }
  done
done

for iteration in $(seq -w 1 "$REPETITIONS"); do
  DIR="$RUN_ROOT/run_${iteration}"
  mkdir -p "$DIR"
  QUERY_ARGS=()
  if [[ -n "${GREENQUERY_ANIMALS_ONLY_QUERIES:-}" ]]; then
    IFS=',' read -ra QUERY_IDS <<< "${GREENQUERY_ANIMALS_ONLY_QUERIES//:/,}"
    for QUERY_ID in "${QUERY_IDS[@]}"; do QUERY_ARGS+=(--only-query "$QUERY_ID"); done
  fi
  "$PYTHON_BIN" evaluation/scripts/run_animals_measurement.py --model-name "$MODEL_NAME" --output-dir "$DIR" --cold-cache --matched-planner "${QUERY_ARGS[@]}"
  MEASUREMENT=$(find "$DIR" -maxdepth 1 -type f -name 'animals_*.json' ! -name '*_summary.json' | sort | tail -n 1)
  if [[ -z "${GREENQUERY_ANIMALS_ONLY_QUERIES:-}" ]]; then
    "$PYTHON_BIN" evaluation/scripts/validate_matched_demo.py "$MEASUREMENT" --model "$MODEL_NAME" --demo animals
  fi
done
"$PYTHON_BIN" evaluation/scripts/build_matched_demo_summary.py "${OUTPUT_ROOT%/}"
