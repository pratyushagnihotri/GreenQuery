#!/bin/bash
# Start (or restart) the GreenQuery backend on the DGX login node in SLURM mode.
# It listens on localhost only; reach it from your laptop through an SSH tunnel:
#   ssh -N -L 8010:localhost:8765 <user>@<cluster-login-node>
# and start the dashboard with that API port (8000 is often used by other local servers):
#   cd dashboard && NEXT_PUBLIC_GREENQUERY_API=http://localhost:8010 npx next dev --webpack -p 3100
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${GREENQUERY_PORT:-8765}
if [[ -f logs/greenquery_server.pid ]]; then
  kill "$(cat logs/greenquery_server.pid)" 2>/dev/null || true
  sleep 1
fi
mkdir -p logs
GREENQUERY_EXECUTOR=slurm \
GREENQUERY_DATA_DIR="$PWD/server_data" \
GREENQUERY_PYTHON="$PWD/.venv/bin/python" \
nohup .venv/bin/uvicorn greenquery_server.app:app --host 127.0.0.1 --port "$PORT" \
  --timeout-graceful-shutdown 2 > logs/greenquery_server.log 2>&1 < /dev/null &
echo $! > logs/greenquery_server.pid
for _ in $(seq 1 30); do
  curl -s "localhost:$PORT/api/health" >/dev/null && { echo "GreenQuery server up on 127.0.0.1:$PORT (pid $(cat logs/greenquery_server.pid))"; exit 0; }
  sleep 1
done
echo "server did not start; see logs/greenquery_server.log" >&2
tail -20 logs/greenquery_server.log >&2
exit 1
