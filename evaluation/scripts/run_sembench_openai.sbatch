#!/bin/bash
# All engines on an API model (OpenAI, or Anthropic via its OpenAI-compatible endpoint) through the
# logging proxy, with a hard spending cap. Engines always request gpt-4o-mini; SERVED_MODEL is what runs.
#
#   sbatch evaluation/scripts/run_sembench_openai.sbatch
#   sbatch --export=ALL,BUDGET_USD=5,SYSTEMS="lotus thalamusdb",QUERIES="1 2" evaluation/scripts/run_sembench_openai.sbatch
#
# Same windows/metrics layout as run_sembench_measured.sbatch, so join_energy.py and
# combine_sembench_measured.py work unchanged. CPU only: GPU energy is not measured (the model
# runs at OpenAI); the trace records the engines' CPU share only. Model energy is estimated
# afterwards (EcoLogits) from the per-call tokens in calls.jsonl.
#
# The real key stays in the proxy process; engines get a dummy key, so every call is logged
# and counted against BUDGET_USD. Once spend reaches the cap the proxy answers HTTP 402.
# Output: tests/results/sembench-openai-<job id>/
#
#SBATCH --job-name=sembench-openai
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

GREENQUERY_ROOT=${GREENQUERY_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/GreenQuery}}
SEMBENCH_ROOT=${SEMBENCH_ROOT:-$GREENQUERY_ROOT/third_party/sembench}  # vendored SemBench (see its GREENQUERY.md)
GQ_PY="$GREENQUERY_ROOT/.venv/bin/python"
MODEL=${MODEL:-gpt-4o-mini}
# SERVED_MODEL: what the proxy actually calls at OpenAI (engines keep requesting MODEL). For the gpt-5
# family set REASONING_EFFORT (minimal|low|medium|high) so requests are adapted for reasoning models.
SERVED_MODEL=${SERVED_MODEL:-$MODEL}
REASONING_EFFORT=${REASONING_EFFORT:-}
# PROVIDER: openai (default), anthropic (Anthropic's OpenAI-compatible endpoint), or openrouter.
PROVIDER=${PROVIDER:-openai}
case "$PROVIDER" in
  openai)    UPSTREAM=https://api.openai.com/v1;    KEY_NAME=OPENAI_API_KEY ;;
  anthropic) UPSTREAM=https://api.anthropic.com/v1; KEY_NAME=ANTHROPIC_API_KEY ;;
  # OpenRouter (OpenAI-compatible); models as "anthropic/claude-haiku-5.5". Key name overridable because
  # an OpenRouter key may be stored under another name in .env.
  openrouter) UPSTREAM=https://openrouter.ai/api/v1; KEY_NAME=${KEY_NAME:-OPENROUTER_API_KEY} ;;
  *) echo "Unknown PROVIDER: $PROVIDER" >&2; exit 1 ;;
esac
PRICE_IN=${PRICE_IN:-0.15}             # USD / 1M tokens (config/models.yaml); embeddings counted at this price too
PRICE_OUT=${PRICE_OUT:-0.60}
PRICE_CACHED_IN=${PRICE_CACHED_IN:-$PRICE_IN}   # cached input tokens (OpenAI bills them at a discount)
SKIP_PAIRS=${SKIP_PAIRS:-}             # system:query pairs not to run, e.g. "lotus:5" (cost reasons)
BUDGET_USD=${BUDGET_USD:-10}
SYSTEMS=${SYSTEMS:-"thalamusdb palimpzest caesura stretto stretto_abacus lotus"}
QUERIES=${QUERIES:-"1 2 4 5"}
REPETITIONS=${REPETITIONS:-1}
SCALE_FACTOR=${SCALE_FACTOR:-1000}
QUERY_TIMEOUT=${QUERY_TIMEOUT:-3600}
OUT=${OUT_DIR:-"$GREENQUERY_ROOT/tests/results/sembench-openai-${SLURM_JOB_ID}"}
mkdir -p "$OUT" "$GREENQUERY_ROOT/logs"

printf "skip_pairs=%s\nprice_cached_in=%s\nserved_model=%s\nreasoning_effort=%s\nno_prefill=%s\n" "$SKIP_PAIRS" "$PRICE_CACHED_IN" "$SERVED_MODEL" "${REASONING_EFFORT:-none}" "$( [[ "$SERVED_MODEL" == *claude* ]] && echo yes || echo no )" > "$OUT/run_config.txt"
printf "local_model=%s\nprovider=%s\nsystems=%s\nqueries=%s\nrepetitions=%s\nscale_factor=%s\nbudget_usd=%s\nprice_in=%s\nprice_out=%s\nquery_timeout_s=%s\n" \
  "$SERVED_MODEL" "$PROVIDER" "$SYSTEMS" "$QUERIES" "$REPETITIONS" "$SCALE_FACTOR" "$BUDGET_USD" "$PRICE_IN" "$PRICE_OUT" "$QUERY_TIMEOUT" >> "$OUT/run_config.txt"
(cd "$SEMBENCH_ROOT" && cat UPSTREAM_COMMIT 2>/dev/null; git rev-parse HEAD; git status --short .) > "$OUT/sembench_version.txt"

# Real key from GreenQuery's .env into GQ_UPSTREAM_KEY only (never echoed, never given to engines).
GQ_UPSTREAM_KEY=$(python3 - "$GREENQUERY_ROOT/.env" "$KEY_NAME" <<'PYEOF'
import sys
from pathlib import Path
for raw in Path(sys.argv[1]).read_text().splitlines():
    key, _, value = raw.strip().partition("=")
    if key.strip() == sys.argv[2]:
        print(value.strip().strip('"').strip("'"))
PYEOF
)
[[ -n "$GQ_UPSTREAM_KEY" ]] || { echo "$KEY_NAME missing in .env" >&2; exit 1; }

PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done; }
trap cleanup EXIT

PROXY_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export PYTHONPATH="$GREENQUERY_ROOT"
"$GQ_PY" -m evaluation.measurement.energy_trace --out "$OUT/trace.jsonl" --interval 0.5 --no-gpu &
PIDS+=($!)
GQ_UPSTREAM_KEY="$GQ_UPSTREAM_KEY" "$GQ_PY" -m evaluation.measurement.llm_proxy --port "$PROXY_PORT" --upstream "$UPSTREAM" \
  --log "$OUT/calls.jsonl" --api-key-env GQ_UPSTREAM_KEY \
  --price-in "$PRICE_IN" --price-out "$PRICE_OUT" --price-cached-in "$PRICE_CACHED_IN" --budget-usd "$BUDGET_USD" \
  $( [[ "$PROVIDER" == openrouter ]] && echo --request-cost ) $( [[ "${LOG_CONTENT:-0}" == 1 ]] && echo --log-content ) \
  ${SERVED_MODEL:+--alias "$MODEL=$SERVED_MODEL" --alias "$MODEL-2024-07-18=$SERVED_MODEL"} \
  ${REASONING_EFFORT:+--reasoning-effort "$REASONING_EFFORT"} \
  $( [[ "$SERVED_MODEL" == *claude* ]] && echo --no-prefill ) > "$OUT/proxy.log" 2>&1 &
PIDS+=($!)
PROXY_PID=$!
unset GQ_UPSTREAM_KEY
for i in $(seq 1 30); do curl -s "http://127.0.0.1:$PROXY_PORT/proxy-health" >/dev/null && break; sleep 1; done
kill -0 "$PROXY_PID" 2>/dev/null || { echo "Logging proxy is not running (see $OUT/proxy.log); aborting." >&2; exit 1; }

window() {  # kind started finished [system query repetition exit]
  printf '{"kind":"%s","started_at":%s,"finished_at":%s,"system":"%s","query":"%s","repetition":%s,"model":"%s","served_model":"%s","exit_code":%s}\n' \
    "$1" "$2" "$3" "${4:-}" "${5:-}" "${6:-0}" "$MODEL" "$SERVED_MODEL" "${7:-0}" >> "$OUT/windows.jsonl"
}

export OPENAI_BASE_URL="http://127.0.0.1:$PROXY_PORT/v1" OPENAI_API_BASE="http://127.0.0.1:$PROXY_PORT/v1" OPENAI_API_KEY="sk-local-dummy-not-a-real-key"
cd "$SEMBENCH_ROOT"
source .venvs/sembench/bin/activate
SKIP=(); [[ -d "files/movie/data/sf_${SCALE_FACTOR}" ]] && SKIP=(--skip-setup)
for REP in $(seq 1 "$REPETITIONS"); do
  for SYSTEM in $SYSTEMS; do
    for Q in $QUERIES; do
      [[ " $SKIP_PAIRS " == *" $SYSTEM:$Q "* ]] && { echo "--- skip $SYSTEM Q$Q (SKIP_PAIRS)"; continue; }
      echo "--- $SYSTEM Q$Q rep=$REP $(date) spent=$(curl -s "http://127.0.0.1:$PROXY_PORT/proxy-health" | python3 -c 'import json,sys; print(round(json.load(sys.stdin)["spent_usd"],3))')"
      rm -f "files/movie/metrics/${SYSTEM}.json"
      T0=$(date +%s.%N)
      timeout --signal=TERM --kill-after=60 "$QUERY_TIMEOUT" \
        python src/run.py --systems "$SYSTEM" --use-cases movie --queries "$Q" --model "$MODEL" \
        --scale-factor "$SCALE_FACTOR" "${SKIP[@]}" > "$OUT/${SYSTEM}_Q${Q}_rep${REP}.log" 2>&1
      CODE=$?
      window query "$T0" "$(date +%s.%N)" "$SYSTEM" "Q$Q" "$REP" "$CODE"
      mkdir -p "$OUT/metrics/rep_$REP" "$OUT/raw_results/rep_$REP/$SYSTEM"
      cp "files/movie/metrics/${SYSTEM}.json" "$OUT/metrics/rep_$REP/${SYSTEM}_Q${Q}.json" 2>/dev/null
      cp "files/movie/raw_results/$SYSTEM/Q${Q}.csv" "$OUT/raw_results/rep_$REP/$SYSTEM/" 2>/dev/null
    done
  done
done

sleep 3
cleanup; sleep 2
"$GQ_PY" "$GREENQUERY_ROOT/evaluation/scripts/join_energy.py" "$OUT"
echo "done $(date)"
