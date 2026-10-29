#!/bin/bash
# Real engines (LOTUS, Palimpzest, ThalamusDB, CAESURA) on a LOCAL model with measured energy.
#
#   sbatch evaluation/scripts/run_sembench_measured.sbatch
#   sbatch --export=ALL,LOCAL_MODEL=gemma3:12b,SYSTEMS="lotus palimpzest",QUERIES="1 2" evaluation/scripts/run_sembench_measured.sbatch
#   Multi-model Stretto/Abacus (LOCAL_MODEL plans, operators offered once per listed model):
#   (the model list contains commas, which sbatch --export would split: set it in the environment)
#   GREENQUERY_OPERATOR_MODELS="ollama/qwen2.5:3b-instruct,ollama/qwen2.5:32b-instruct" \
#     sbatch --export=ALL,LOCAL_MODEL=qwen2.5:32b-instruct,SYSTEMS="stretto stretto_abacus",OLLAMA_CONTEXT_LENGTH=8192 \
#     evaluation/scripts/run_sembench_measured.sbatch
#   OLLAMA_CONTEXT_LENGTH (default: Ollama's) caps the KV cache; needed when a 32B model and a second
#   model share one 40 GB GPU at OLLAMA_NUM_PARALLEL=4.
#
# Every engine believes it calls OpenAI "gpt-4o-mini"; the logging proxy rewrites the model to
# LOCAL_MODEL and forwards to Ollama on this job's GPU. Engines get a dummy key, so a call that
# escapes the proxy fails instead of billing the real account. One window per
# (repetition, system, query); energy is joined afterwards (evaluation/scripts/join_energy.py).
# Output: tests/results/sembench-measured-<job id>/
#
#SBATCH --job-name=sembench-measured
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

GREENQUERY_ROOT=${GREENQUERY_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/GreenQuery}}
SEMBENCH_ROOT=${SEMBENCH_ROOT:-$GREENQUERY_ROOT/third_party/sembench}  # vendored SemBench (see its GREENQUERY.md)
GQ_PY="$GREENQUERY_ROOT/.venv/bin/python"
OLLAMA_HOME=${OLLAMA_HOME:-$HOME/ollama}
export OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/ollama-models}
LOCAL_MODEL=${LOCAL_MODEL:-qwen2.5:7b-instruct}
EMBED_MODEL=${EMBED_MODEL:-nomic-embed-text}
SYSTEMS=${SYSTEMS:-"lotus palimpzest thalamusdb caesura"}
QUERIES=${QUERIES:-"1 2 4 5"}          # SemBench Movie Q1/Q2/Q4/Q5 = Stretto Movie Q1/Q2/Q4/Q3
REPETITIONS=${REPETITIONS:-1}
REP_OFFSET=${REP_OFFSET:-0}            # number repetitions REP_OFFSET+1 .. REP_OFFSET+REPETITIONS (extra runs later)
SCALE_FACTOR=${SCALE_FACTOR:-1000}
IDLE_SECONDS=${IDLE_SECONDS:-60}
QUERY_TIMEOUT=${QUERY_TIMEOUT:-3600}   # seconds per (system, query); exit 124 = timed out
TRACE_INTERVAL=${TRACE_INTERVAL:-0.1}  # energy trace sampling period (s); validation runs vary it
NVSMI_SAMPLE=${NVSMI_SAMPLE:-0}        # 1 = also sample nvidia-smi power.draw (validation vs NVML counter)
OUT=${OUT_DIR:-"$GREENQUERY_ROOT/tests/results/sembench-measured-${SLURM_JOB_ID}"}
mkdir -p "$OUT" "$GREENQUERY_ROOT/logs"

GPU_CHECK_ID=${CUDA_VISIBLE_DEVICES:-}; GPU_CHECK_ID=${GPU_CHECK_ID%%,*}
if nvidia-smi --id="${GPU_CHECK_ID:-0}" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "Allocated GPU already has a compute process; refusing to share it." >&2; exit 1
fi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.draw --format=csv | tee "$OUT/gpu.csv"
OPERATOR_MODELS=${GREENQUERY_OPERATOR_MODELS:-}
printf "operator_models=%s\nollama_context_length=%s\nquality_target=%s\ntrace_interval_s=%s\nnvsmi_sample=%s\n" "$OPERATOR_MODELS" "${OLLAMA_CONTEXT_LENGTH:-default}" "${GREENQUERY_QUALITY_TARGET:-0.8}" "$TRACE_INTERVAL" "$NVSMI_SAMPLE" > "$OUT/run_config.txt"
printf "local_model=%s\nsystems=%s\nqueries=%s\nrepetitions=%s\nrep_offset=%s\nscale_factor=%s\nollama_num_parallel=4\nalias=gpt-4o-mini(+-2024-07-18)->%s\nquery_timeout_s=%s\n" \
  "$LOCAL_MODEL" "$SYSTEMS" "$QUERIES" "$REPETITIONS" "$REP_OFFSET" "$SCALE_FACTOR" "$LOCAL_MODEL" "$QUERY_TIMEOUT" >> "$OUT/run_config.txt"
(cd "$SEMBENCH_ROOT" && cat UPSTREAM_COMMIT 2>/dev/null; git rev-parse HEAD; git status --short .) > "$OUT/sembench_version.txt"

PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done; }
trap cleanup EXIT

# Ollama on this job's GPU, job-specific port, fixed parallelism for every engine.
# Free ports chosen at start (fixed job-id-based ports collided between consecutive jobs:
# job N's proxy port was job N+1's Ollama port).
free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
OLLAMA_PORT=$(free_port)
PROXY_PORT=$(free_port); while [[ "$PROXY_PORT" == "$OLLAMA_PORT" ]]; do PROXY_PORT=$(free_port); done
echo "ollama_port=$OLLAMA_PORT proxy_port=$PROXY_PORT" >> "$OUT/run_config.txt"
alive_or_die() { kill -0 "$1" 2>/dev/null || { echo "$2 is not running (see $3); aborting." >&2; exit 1; }; }
export LD_LIBRARY_PATH="$OLLAMA_HOME/lib/ollama:${LD_LIBRARY_PATH:-}"
[[ -n "${OLLAMA_CONTEXT_LENGTH:-}" ]] && export OLLAMA_CONTEXT_LENGTH
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=24h \
  "$OLLAMA_HOME/bin/ollama" serve > "$OUT/ollama-serve.log" 2>&1 &
PIDS+=($!)
OLLAMA_PID=$!
for i in $(seq 1 60); do curl -s "http://127.0.0.1:$OLLAMA_PORT" >/dev/null && break; sleep 1; done
alive_or_die "$OLLAMA_PID" "Ollama" "$OUT/ollama-serve.log"
OPERATOR_TAGS=$(echo "$OPERATOR_MODELS" | tr ',' ' ' | sed 's#ollama/##g')
# Models added from the dashboard may not be downloaded yet; pull them before any measurement.
for m in "$LOCAL_MODEL" "$EMBED_MODEL" $OPERATOR_TAGS; do
  if ! OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" "$OLLAMA_HOME/bin/ollama" show "$m" >/dev/null 2>&1; then
    echo "$m" > "$OUT/pulling"
    OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" "$OLLAMA_HOME/bin/ollama" pull "$m" > "$OUT/ollama-pull.log" 2>&1 \
      || { echo "Could not download $m (see ollama-pull.log); aborting." >&2; rm -f "$OUT/pulling"; exit 1; }
    rm -f "$OUT/pulling"
  fi
done
for m in "$LOCAL_MODEL" "$EMBED_MODEL" $OPERATOR_TAGS; do  # load into GPU memory before measuring
  curl -s "http://127.0.0.1:$OLLAMA_PORT/api/generate" -d "{\"model\": \"$m\", \"prompt\": \"ok\", \"stream\": false, \"options\": {\"num_predict\": 1}}" >/dev/null || true
done
curl -s "http://127.0.0.1:$OLLAMA_PORT/api/embed" -d "{\"model\": \"$EMBED_MODEL\", \"input\": \"ok\"}" >/dev/null || true

# Energy trace (whole job) and logging proxy.
export GREENQUERY_GPU_INDEX=0 PYTHONPATH="$GREENQUERY_ROOT"
"$GQ_PY" -m evaluation.measurement.energy_trace --out "$OUT/trace.jsonl" --interval "$TRACE_INTERVAL" &
PIDS+=($!)
if [[ "$NVSMI_SAMPLE" == 1 ]]; then  # independent power samples (W) for comparison with the counter
  nvidia-smi --id="${GPU_CHECK_ID:-0}" --query-gpu=timestamp,power.draw --format=csv,noheader,nounits -lms 100 > "$OUT/nvsmi_power.csv" &
  PIDS+=($!)
fi
"$GQ_PY" -m evaluation.measurement.llm_proxy --port "$PROXY_PORT" --upstream "http://127.0.0.1:$OLLAMA_PORT/v1" \
  --log "$OUT/calls.jsonl" --alias "gpt-4o-mini=$LOCAL_MODEL" --alias "gpt-4o-mini-2024-07-18=$LOCAL_MODEL" \
  --alias "text-embedding-3-small=$EMBED_MODEL" \
  > "$OUT/proxy.log" 2>&1 &
PIDS+=($!)
PROXY_PID=$!
for i in $(seq 1 30); do curl -s "http://127.0.0.1:$PROXY_PORT/proxy-health" >/dev/null && break; sleep 1; done
alive_or_die "$PROXY_PID" "Logging proxy" "$OUT/proxy.log"
# The proxy on PROXY_PORT must be ours: it must serve this job's model.
curl -s "http://127.0.0.1:$PROXY_PORT/proxy-health" | grep -q "\"gpt-4o-mini\": \"$LOCAL_MODEL\"" \
  || { echo "Proxy on port $PROXY_PORT does not serve $LOCAL_MODEL; aborting." >&2; exit 1; }

window() {  # kind started finished [system query repetition exit]
  printf '{"kind":"%s","started_at":%s,"finished_at":%s,"system":"%s","query":"%s","repetition":%s,"model":"gpt-4o-mini","served_model":"%s","exit_code":%s}\n' \
    "$1" "$2" "$3" "${4:-}" "${5:-}" "${6:-0}" "$LOCAL_MODEL" "${7:-0}" >> "$OUT/windows.jsonl"
}

# Idle baseline with the models loaded.
sleep 10; T0=$(date +%s.%N); sleep "$IDLE_SECONDS"; window idle "$T0" "$(date +%s.%N)"

# Engines talk to the proxy only, with a dummy key.
# Operator models (multi-model mode) are called by their own names through the proxy, unaliased.
export GREENQUERY_OLLAMA_BASE_URL="http://127.0.0.1:$PROXY_PORT/v1"
export OPENAI_BASE_URL="http://127.0.0.1:$PROXY_PORT/v1" OPENAI_API_BASE="http://127.0.0.1:$PROXY_PORT/v1" OPENAI_API_KEY="sk-local-dummy-not-a-real-key"
cd "$SEMBENCH_ROOT"
source .venvs/sembench/bin/activate
SKIP=(); [[ -d "files/movie/data/sf_${SCALE_FACTOR}" ]] && SKIP=(--skip-setup)
for REP in $(seq $((REP_OFFSET + 1)) $((REP_OFFSET + REPETITIONS))); do
  for SYSTEM in $SYSTEMS; do
    for Q in $QUERIES; do
      echo "--- $SYSTEM Q$Q rep=$REP $(date)"
      rm -f "files/movie/metrics/${SYSTEM}.json"
      T0=$(date +%s.%N)
      timeout --signal=TERM --kill-after=60 "$QUERY_TIMEOUT" \
        python src/run.py --systems "$SYSTEM" --use-cases movie --queries "$Q" --model gpt-4o-mini \
        --scale-factor "$SCALE_FACTOR" "${SKIP[@]}" > "$OUT/${SYSTEM}_Q${Q}_rep${REP}.log" 2>&1
      CODE=$?
      window query "$T0" "$(date +%s.%N)" "$SYSTEM" "Q$Q" "$REP" "$CODE"
      mkdir -p "$OUT/metrics/rep_$REP"
      cp "files/movie/metrics/${SYSTEM}.json" "$OUT/metrics/rep_$REP/${SYSTEM}_Q${Q}.json" 2>/dev/null
      mkdir -p "$OUT/raw_results/rep_$REP/$SYSTEM"
      cp "files/movie/raw_results/$SYSTEM/Q${Q}.csv" "$OUT/raw_results/rep_$REP/$SYSTEM/" 2>/dev/null
    done
  done
done

sleep 3   # keep the trace running past the last window
cleanup; sleep 2
"$GQ_PY" "$GREENQUERY_ROOT/evaluation/scripts/join_energy.py" "$OUT"
echo "done $(date)"
