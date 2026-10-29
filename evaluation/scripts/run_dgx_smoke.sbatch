#!/bin/bash
# Smoke test: does every model run end to end through Stretto on the DGX?
#
#   sbatch --gres=gpu:1 evaluation/scripts/run_dgx_smoke.sbatch
#   sbatch --gres=gpu:2 --export=ALL,MODELS="ollama/llama3.3:70b" evaluation/scripts/run_dgx_smoke.sbatch
#
# Runs Movie Q1 once per model (cold cache) with Ollama served from the user's own
# install on a job-specific port, GPU energy from the nvidia-smi sampler and the
# NVML energy counter, and local embeddings for local models. OpenAI models are
# added automatically when .env holds OPENAI_API_KEY. No root access is needed.
# Results: tests/results/dgx-smoke-<job id>/ (raw runs, logs, compact.json, summary.md).
#
#SBATCH --job-name=gq-smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

REPO_ROOT=${GREENQUERY_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}
OLLAMA_HOME=${OLLAMA_HOME:-$HOME/ollama}
export OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/ollama-models}
DEFAULT_LOCAL="ollama/llama3.2:3b ollama/qwen2.5:3b-instruct ollama/qwen2.5:7b-instruct ollama/llama3.1:8b ollama/qwen2.5:32b-instruct"
MODELS=${MODELS:-$DEFAULT_LOCAL}
if [[ -z "${SKIP_API:-}" ]] && grep -qE '^OPENAI_API_KEY=.{20,}' .env 2>/dev/null; then
  MODELS="$MODELS gpt-4o-mini gpt-4.1"
fi

OUT="tests/results/dgx-smoke-${SLURM_JOB_ID}"
RAW="$OUT/raw"
mkdir -p "$RAW" logs

GPU_CHECK_ID=${CUDA_VISIBLE_DEVICES:-}; GPU_CHECK_ID=${GPU_CHECK_ID%%,*}
if nvidia-smi --id="${GPU_CHECK_ID:-0}" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "Allocated GPU already has a compute process; refusing to share it." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.draw --format=csv | tee "$OUT/gpu.csv"
"$PYTHON_BIN" -m evaluation.measurement.energy_counters probe > "$OUT/probe.json" 2>&1

# Ollama on a port derived from the job id so concurrent jobs never collide.
OLLAMA_PORT=$((30000 + SLURM_JOB_ID % 20000))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export GREENQUERY_OLLAMA_BASE_URL="http://127.0.0.1:${OLLAMA_PORT}/v1"
export LD_LIBRARY_PATH="$OLLAMA_HOME/lib/ollama:${LD_LIBRARY_PATH:-}"
"$OLLAMA_HOME/bin/ollama" serve > "$OUT/ollama-serve.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
for i in $(seq 1 60); do curl -s "http://${OLLAMA_HOST}" >/dev/null && break; sleep 1; done

export GREENQUERY_GPU_INDEX=0
export GREENQUERY_ENERGY_COUNTERS=1
export GREENQUERY_CO2_G_PER_KWH=${GREENQUERY_CO2_G_PER_KWH:-344}

safe() { echo "$1" | sed 's/[^A-Za-z0-9]/_/g'; }
printf "model\tstatus\tvalidation\n" > "$OUT/status.tsv"

for MODEL in $MODELS; do
  DIR="$RAW/$(safe "$MODEL")/run_01"
  mkdir -p "$DIR"
  # Local models: measure this job's GPU. API models run elsewhere: no local GPU
  # measurement (it would replace the EcoLogits estimate with idle-GPU energy).
  case "$MODEL" in
    ollama/*) export GREENQUERY_EMBEDDING_MODEL=local GREENQUERY_ENERGY_COUNTERS=1; ENERGY=nvml ;;
    *) export GREENQUERY_EMBEDDING_MODEL=openai GREENQUERY_ENERGY_COUNTERS=0; ENERGY=none ;;
  esac
  echo "=== $MODEL (embedding: $GREENQUERY_EMBEDDING_MODEL) $(date)"
  if "$PYTHON_BIN" evaluation/scripts/run_movie_measurement.py \
      --query-set demo --demo-q4-mode ratio --demo-query Q1 --backend-profile api \
      --model-name "$MODEL" --cold-cache --online-cache-policy disabled --split-artifacts \
      --energy-measurement "$ENERGY" --gpu-model "$(head -2 "$OUT/gpu.csv" | tail -1 | cut -d, -f2 | xargs)" \
      --hardware-provider dgx --run-namespace "smoke_$(safe "$MODEL")_${SLURM_JOB_ID}" \
      --output-dir "$DIR" > "$DIR/run.log" 2>&1; then
    MEAS=$(find "$DIR" -maxdepth 1 -name 'movie_*.json' ! -name '*_summary.json' | sort | tail -1)
    "$PYTHON_BIN" evaluation/scripts/build_movie_demo_summary.py "$MEAS" >> "$DIR/run.log" 2>&1
    if "$PYTHON_BIN" evaluation/scripts/validate_online_movie_run.py "$MEAS" "${MEAS%.json}_summary.json" \
        --model "$MODEL" --only-query Q1_positive_reviews >> "$DIR/run.log" 2>&1; then
      printf "%s\tsucceeded\tvalidated\n" "$MODEL" >> "$OUT/status.tsv"
    else
      printf "%s\tsucceeded\tvalidation_failed\n" "$MODEL" >> "$OUT/status.tsv"
    fi
  else
    printf "%s\tfailed\t%s\n" "$MODEL" "$(grep -E 'Error|Exception' "$DIR/run.log" | tail -1 | cut -c1-120)" >> "$OUT/status.tsv"
    mv "$DIR" "$(dirname "$DIR")/failed_run_01"  # keep the log; builders only read run_* dirs
  fi
done

"$PYTHON_BIN" evaluation/scripts/build_online_movie_query_metrics.py "$RAW" "$OUT/compact.json"
"$PYTHON_BIN" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
records = {r["model"]: r for r in json.loads((out / "compact.json").read_text())} if (out / "compact.json").exists() else {}
rows = [line.split("\t") for line in (out / "status.tsv").read_text().splitlines()[1:]]
lines = ["# DGX smoke test", "", "Movie Q1, 1 repetition, cold cache, Stretto. Energy: local models = measured GPU (nvidia-smi sampler); API models = EcoLogits estimate.", "",
         "| Model | Status | Validation | Runtime (s) | Tokens | Cost (USD) | Energy (Wh) | Precision |", "|---|---|---|---|---|---|---|---|"]
for model, status, detail in rows:
    r = records.get(model, {})
    energy = r.get("energy_kwh")
    lines.append(f"| {model} | {status} | {detail} | {r.get('runtime_seconds', 0):.1f} | {r.get('total_tokens', '—')} | "
                 f"{r.get('api_cost_usd') or '—'} | {'%.3f' % (energy * 1000) if energy else '—'} | {(r.get('quality') or {}).get('precision', '—')} |")
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
