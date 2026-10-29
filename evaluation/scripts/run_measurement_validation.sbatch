#!/bin/bash
# Measurement validation (V4) on ONE GPU: the same queries in alternating phases that differ only in
# how energy is sampled, so GPU-to-GPU differences cannot masquerade as sampling overhead.
#
#   sbatch evaluation/scripts/run_measurement_validation.sbatch
#   sbatch --export=ALL,PHASES="0.1:1 5:0 0.1:1 5:0",SYSTEMS="lotus stretto" evaluation/scripts/run_measurement_validation.sbatch
#
# PHASES: space-separated "interval:nvsmi" (trace sampling period in s; 1 = also sample nvidia-smi
# power at 100 ms). Default ABAB: fast+nvidia-smi, slow, fast+nvidia-smi, slow.
# Each phase writes phase_<k>/{trace.jsonl,windows.jsonl,energy_records.json}; Ollama and the
# logging proxy are shared (calls.jsonl at the top level, linked into every phase).
# Output: tests/results/measurement-validation-<job id>/
#
#SBATCH --job-name=measurement-validation
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

GREENQUERY_ROOT=${GREENQUERY_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/GreenQuery}}
SEMBENCH_ROOT=${SEMBENCH_ROOT:-$GREENQUERY_ROOT/third_party/sembench}  # vendored SemBench (see its GREENQUERY.md)
GQ_PY="$GREENQUERY_ROOT/.venv/bin/python"
OLLAMA_HOME=${OLLAMA_HOME:-$HOME/ollama}
export OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/ollama-models}
LOCAL_MODEL=${LOCAL_MODEL:-qwen2.5:7b-instruct}
EMBED_MODEL=${EMBED_MODEL:-nomic-embed-text}
SYSTEMS=${SYSTEMS:-"thalamusdb lotus stretto"}
QUERIES=${QUERIES:-"1 2 4"}
PHASES=${PHASES:-"0.1:1 5:0 0.1:1 5:0"}
SCALE_FACTOR=${SCALE_FACTOR:-1000}
IDLE_SECONDS=${IDLE_SECONDS:-120}
QUERY_TIMEOUT=${QUERY_TIMEOUT:-3600}
OUT=${OUT_DIR:-"$GREENQUERY_ROOT/tests/results/measurement-validation-${SLURM_JOB_ID}"}
mkdir -p "$OUT" "$GREENQUERY_ROOT/logs"

GPU_CHECK_ID=${CUDA_VISIBLE_DEVICES:-}; GPU_CHECK_ID=${GPU_CHECK_ID%%,*}
if nvidia-smi --id="${GPU_CHECK_ID:-0}" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "Allocated GPU already has a compute process; refusing to share it." >&2; exit 1
fi
nvidia-smi --id="${GPU_CHECK_ID:-0}" --query-gpu=index,name,pci.bus_id,serial,power.draw --format=csv | tee "$OUT/gpu.csv"
printf "local_model=%s\nsystems=%s\nqueries=%s\nphases=%s\nscale_factor=%s\n" \
  "$LOCAL_MODEL" "$SYSTEMS" "$QUERIES" "$PHASES" "$SCALE_FACTOR" > "$OUT/run_config.txt"

PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done; }
trap cleanup EXIT
free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
OLLAMA_PORT=$(free_port)
PROXY_PORT=$(free_port); while [[ "$PROXY_PORT" == "$OLLAMA_PORT" ]]; do PROXY_PORT=$(free_port); done

export LD_LIBRARY_PATH="$OLLAMA_HOME/lib/ollama:${LD_LIBRARY_PATH:-}"
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=24h \
  "$OLLAMA_HOME/bin/ollama" serve > "$OUT/ollama-serve.log" 2>&1 &
OLLAMA_PID=$!; PIDS+=($OLLAMA_PID)
for i in $(seq 1 60); do curl -s "http://127.0.0.1:$OLLAMA_PORT" >/dev/null && break; sleep 1; done
kill -0 "$OLLAMA_PID" 2>/dev/null || { echo "Ollama is not running; aborting." >&2; exit 1; }
for m in "$LOCAL_MODEL" "$EMBED_MODEL"; do
  curl -s "http://127.0.0.1:$OLLAMA_PORT/api/generate" -d "{\"model\": \"$m\", \"prompt\": \"ok\", \"stream\": false}" >/dev/null || true
done
curl -s "http://127.0.0.1:$OLLAMA_PORT/api/embed" -d "{\"model\": \"$EMBED_MODEL\", \"input\": \"ok\"}" >/dev/null || true

export GREENQUERY_GPU_INDEX=0 PYTHONPATH="$GREENQUERY_ROOT"
"$GQ_PY" -m evaluation.measurement.llm_proxy --port "$PROXY_PORT" --upstream "http://127.0.0.1:$OLLAMA_PORT/v1" \
  --log "$OUT/calls.jsonl" --alias "gpt-4o-mini=$LOCAL_MODEL" --alias "gpt-4o-mini-2024-07-18=$LOCAL_MODEL" \
  --alias "text-embedding-3-small=$EMBED_MODEL" > "$OUT/proxy.log" 2>&1 &
PROXY_PID=$!; PIDS+=($PROXY_PID)
for i in $(seq 1 30); do curl -s "http://127.0.0.1:$PROXY_PORT/proxy-health" >/dev/null && break; sleep 1; done
curl -s "http://127.0.0.1:$PROXY_PORT/proxy-health" | grep -q "\"gpt-4o-mini\": \"$LOCAL_MODEL\"" \
  || { echo "Proxy on port $PROXY_PORT does not serve $LOCAL_MODEL; aborting." >&2; exit 1; }

export OPENAI_BASE_URL="http://127.0.0.1:$PROXY_PORT/v1" OPENAI_API_BASE="http://127.0.0.1:$PROXY_PORT/v1" OPENAI_API_KEY="sk-local-dummy-not-a-real-key"
cd "$SEMBENCH_ROOT"
source .venvs/sembench/bin/activate
SKIP=(); [[ -d "files/movie/data/sf_${SCALE_FACTOR}" ]] && SKIP=(--skip-setup)

K=0
for PHASE in $PHASES; do
  K=$((K + 1)); INTERVAL=${PHASE%%:*}; NVSMI=${PHASE##*:}
  P="$OUT/phase_$K"; mkdir -p "$P"; ln -sf ../calls.jsonl "$P/calls.jsonl"
  printf "interval_s=%s\nnvsmi_sample=%s\n" "$INTERVAL" "$NVSMI" > "$P/phase.txt"
  "$GQ_PY" -m evaluation.measurement.energy_trace --out "$P/trace.jsonl" --interval "$INTERVAL" &
  TRACE_PID=$!
  NVSMI_PID=""
  if [[ "$NVSMI" == 1 ]]; then
    nvidia-smi --id="${GPU_CHECK_ID:-0}" --query-gpu=timestamp,power.draw --format=csv,noheader,nounits -lms 100 > "$P/nvsmi_power.csv" &
    NVSMI_PID=$!
  fi
  sleep 10
  T0=$(date +%s.%N); sleep "$IDLE_SECONDS"
  printf '{"kind":"idle","started_at":%s,"finished_at":%s,"system":"","query":"","repetition":%s,"model":"gpt-4o-mini","served_model":"%s","exit_code":0}\n' \
    "$T0" "$(date +%s.%N)" "$K" "$LOCAL_MODEL" >> "$P/windows.jsonl"
  for SYSTEM in $SYSTEMS; do
    for Q in $QUERIES; do
      echo "--- phase $K ($PHASE) $SYSTEM Q$Q $(date)"
      rm -f "files/movie/metrics/${SYSTEM}.json"
      T0=$(date +%s.%N)
      timeout --signal=TERM --kill-after=60 "$QUERY_TIMEOUT" \
        python src/run.py --systems "$SYSTEM" --use-cases movie --queries "$Q" --model gpt-4o-mini \
        --scale-factor "$SCALE_FACTOR" "${SKIP[@]}" > "$P/${SYSTEM}_Q${Q}.log" 2>&1
      CODE=$?
      printf '{"kind":"query","started_at":%s,"finished_at":%s,"system":"%s","query":"Q%s","repetition":%s,"model":"gpt-4o-mini","served_model":"%s","exit_code":%s}\n' \
        "$T0" "$(date +%s.%N)" "$SYSTEM" "$Q" "$K" "$LOCAL_MODEL" "$CODE" >> "$P/windows.jsonl"
      mkdir -p "$P/metrics/rep_$K"
      cp "files/movie/metrics/${SYSTEM}.json" "$P/metrics/rep_$K/${SYSTEM}_Q${Q}.json" 2>/dev/null
    done
  done
  sleep 3
  kill "$TRACE_PID" 2>/dev/null; [[ -n "$NVSMI_PID" ]] && kill "$NVSMI_PID" 2>/dev/null
  wait "$TRACE_PID" 2>/dev/null
  "$GQ_PY" "$GREENQUERY_ROOT/evaluation/scripts/join_energy.py" "$P"
done

cleanup; sleep 2
echo "done $(date)"
