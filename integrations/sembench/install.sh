#!/bin/bash
# Install GreenQuery's Stretto runners into a SemBench checkout (idempotent). Default: the vendored copy in
# third_party/sembench (runners already included there); pass another SemBench root to install elsewhere.
#   integrations/sembench/install.sh [SEMBENCH_ROOT]
set -euo pipefail
GQ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SB_ROOT=${1:-${SEMBENCH_ROOT:-$GQ_ROOT/third_party/sembench}}

mkdir -p "$SB_ROOT/src/scenario/movie/runner/stretto_runner"
cp "$GQ_ROOT/integrations/sembench/stretto_runner.py" "$SB_ROOT/src/scenario/movie/runner/stretto_runner/stretto_runner.py"
touch "$SB_ROOT/src/scenario/movie/runner/stretto_runner/__init__.py"

mkdir -p "$SB_ROOT/src/scenario/movie/runner/stretto_abacus_runner"
cp "$GQ_ROOT/integrations/sembench/stretto_abacus_runner.py" "$SB_ROOT/src/scenario/movie/runner/stretto_abacus_runner/stretto_abacus_runner.py"
touch "$SB_ROOT/src/scenario/movie/runner/stretto_abacus_runner/__init__.py"

# Register the system names with SemBench's runner registry.
if ! grep -q '"stretto": "StrettoRunner"' "$SB_ROOT/src/run.py"; then
  sed -i 's/        "caesura": "CaesuraRunner",/        "caesura": "CaesuraRunner",\n        "stretto": "StrettoRunner",  # GreenQuery/' "$SB_ROOT/src/run.py"
fi
if ! grep -q '"stretto_abacus": "StrettoAbacusRunner"' "$SB_ROOT/src/run.py"; then
  sed -i 's/        "stretto": "StrettoRunner",  # GreenQuery/        "stretto": "StrettoRunner",  # GreenQuery\n        "stretto_abacus": "StrettoAbacusRunner",  # GreenQuery/' "$SB_ROOT/src/run.py"
fi

# Stretto runs in GreenQuery's own environment (reasondb + measurement code).
ln -sfn "$GQ_ROOT/.venv" "$SB_ROOT/.venvs/stretto"
ln -sfn "$GQ_ROOT/.venv" "$SB_ROOT/.venvs/stretto_abacus"
grep -n '"stretto' "$SB_ROOT/src/run.py"
echo "Stretto runner installed into $SB_ROOT"
