#!/bin/bash
# Smoke test of the real LOTUS, Palimpzest (Abacus), ThalamusDB and CAESURA through
# SemBench (vendored in third_party/sembench, override with $SEMBENCH_ROOT).
#
#   sbatch --gres=none evaluation/scripts/run_sembench_smoke.sbatch
#   sbatch --gres=none --export=ALL,QUERIES="1 2 4 5",SYSTEMS="lotus palimpzest",REPETITIONS=3 evaluation/scripts/run_sembench_smoke.sbatch
#
# API keys come from GreenQuery's .env (exported, never copied). Results are copied
# to tests/results/sembench-<job id>/ in GreenQuery. SemBench Movie Q1/Q2/Q4/Q5
# correspond to Stretto's Movie Q1/Q2/Q4/Q3.
#
#SBATCH --job-name=sembench-smoke
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

GREENQUERY_ROOT=${GREENQUERY_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/GreenQuery}}
SEMBENCH_ROOT=${SEMBENCH_ROOT:-$GREENQUERY_ROOT/third_party/sembench}  # vendored SemBench (see its GREENQUERY.md)
SYSTEMS=${SYSTEMS:-"lotus palimpzest thalamusdb caesura"}
QUERIES=${QUERIES:-"1"}
MODEL=${MODEL:-gpt-4o-mini}
# 1000: smallest SemBench sample where all four Movie queries have valid ground truth
# (sf=100 is a single movie, so Q2/Q4 are empty).
SCALE_FACTOR=${SCALE_FACTOR:-1000}
REPETITIONS=${REPETITIONS:-1}
OUT=${OUT_DIR:-"$GREENQUERY_ROOT/tests/results/sembench-${SLURM_JOB_ID}"}
mkdir -p "$OUT"

# Load GreenQuery's .env without executing it (bash `source` would run a value
# containing spaces as a command and print it). Values are never echoed.
eval "$(python3 - "$GREENQUERY_ROOT/.env" <<'PYEOF'
import shlex, sys
from pathlib import Path
for raw in Path(sys.argv[1]).read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key, value = key.strip(), value.strip().strip('"').strip("'")
    if key.isidentifier() and value:
        print(f"export {key}={shlex.quote(value)}")
PYEOF
)"
[[ -n "${OPENAI_API_KEY:-}" ]] || echo "warning: OPENAI_API_KEY is empty after loading .env" >&2
cd "$SEMBENCH_ROOT"
source .venvs/sembench/bin/activate
# Never move files/movie/raw_results wholesale: raw_results/ground_truth/ holds the
# answers SemBench scores against. Each repetition below clears only the current
# system's own metrics file and raw results (restorable with `git checkout`).
git rev-parse HEAD > "$OUT/sembench_commit.txt"
printf "systems=%s\nqueries=%s\nmodel=%s\nscale_factor=%s\nrepetitions=%s\n" "$SYSTEMS" "$QUERIES" "$MODEL" "$SCALE_FACTOR" "$REPETITIONS" > "$OUT/run_config.txt"

for REP in $(seq -w 1 "$REPETITIONS"); do
  for SYSTEM in $SYSTEMS; do
    echo "=== $SYSTEM rep=$REP $(date)"
    # SemBench keeps one metrics file per system: clear it so each repetition is its own record.
    rm -f "files/movie/metrics/${SYSTEM}.json"; rm -rf "files/movie/raw_results/${SYSTEM}"
    SKIP=(); [[ -d "files/movie/data/sf_${SCALE_FACTOR}" ]] && SKIP=(--skip-setup)
    python src/run.py --systems "$SYSTEM" --use-cases movie --queries $QUERIES \
      --model "$MODEL" --scale-factor "$SCALE_FACTOR" "${SKIP[@]}" > "$OUT/${SYSTEM}_rep${REP}.log" 2>&1
    echo "exit=$? $SYSTEM rep=$REP" | tee -a "$OUT/exit_codes.txt"
    mkdir -p "$OUT/rep_$REP/raw_results/$SYSTEM"
    cp -r "files/movie/raw_results/$SYSTEM/." "$OUT/rep_$REP/raw_results/$SYSTEM/" 2>/dev/null
    cp "files/movie/metrics/${SYSTEM}.json" "$OUT/rep_$REP/${SYSTEM}.json" 2>/dev/null
  done
done
echo "done $(date)"
