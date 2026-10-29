#!/bin/bash

# Start the text-only local model services needed by the Phase 4 Movie semantic
# run. Use this on DGX, CloudLab, or any machine where the local text model and
# embedding backend should run on a visible CUDA GPU.

set -e

PYTHON_BIN=${PYTHON_BIN:-python3}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOCAL_GPU_ID=${LOCAL_GPU_ID:-${DGX_GPU_ID:-0}}
LOCAL_MODEL_NAME=${LOCAL_MODEL_NAME:-${DGX_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}}
LOCAL_TEXT_QA_PORT=${LOCAL_TEXT_QA_PORT:-5010}
LOCAL_TEXT_EMBED_PORT=${LOCAL_TEXT_EMBED_PORT:-5324}
KILL_SCRIPT=${KILL_SCRIPT:-scripts/kill_local_text_servers.sh}
LOCAL_GOLD_VANILLA=${LOCAL_GOLD_VANILLA:-0}

KV_SERVER_ARGS=()
if [ "$LOCAL_GOLD_VANILLA" = "1" ]; then
    KV_SERVER_ARGS+=(--gold-vanilla)
fi

echo "Starting local text services"
echo "Python: $PYTHON_BIN"
echo "GPU id: $LOCAL_GPU_ID"
echo "Model: $LOCAL_MODEL_NAME"

CUDA_VISIBLE_DEVICES="$LOCAL_GPU_ID" "$PYTHON_BIN" \
    reasondb/backends/kv_cache_text_qa_server.py \
    --model-name "$LOCAL_MODEL_NAME" \
    --device-id 0 \
    --port "$LOCAL_TEXT_QA_PORT" \
    "${KV_SERVER_ARGS[@]}" &
KV_TEXT_QA_SERVER_PID=$!

CUDA_VISIBLE_DEVICES="$LOCAL_GPU_ID" "$PYTHON_BIN" \
    reasondb/backends/text_embed_server.py \
    --device-id 0 &
TEXT_EMBED_SERVER_PID=$!

cat > "$KILL_SCRIPT" <<EOF
#!/bin/bash
kill $KV_TEXT_QA_SERVER_PID $TEXT_EMBED_SERVER_PID
EOF
chmod +x "$KILL_SCRIPT"

echo "Local text services started"
echo "KV TextQA PID: $KV_TEXT_QA_SERVER_PID"
echo "Text embed PID: $TEXT_EMBED_SERVER_PID"
echo "Stop with: $KILL_SCRIPT"

wait_for_http() {
    local name="$1"
    local url="$2"
    local max_attempts="${3:-120}"
    local attempt=1

    while [ "$attempt" -le "$max_attempts" ]; do
        if "$PYTHON_BIN" - "$url" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import urlopen

with urlopen(sys.argv[1], timeout=2) as response:
    if response.status < 500:
        raise SystemExit(0)
raise SystemExit(1)
PY
        then
            echo "$name is ready"
            return 0
        fi

        if ! kill -0 "$KV_TEXT_QA_SERVER_PID" 2>/dev/null; then
            echo "KV TextQA server exited before readiness" >&2
            return 1
        fi
        if ! kill -0 "$TEXT_EMBED_SERVER_PID" 2>/dev/null; then
            echo "Text embedding server exited before readiness" >&2
            return 1
        fi

        sleep 5
        attempt=$((attempt + 1))
    done

    echo "$name did not become ready after $max_attempts attempts" >&2
    return 1
}

wait_for_http "KV TextQA server" "http://127.0.0.1:${LOCAL_TEXT_QA_PORT}/status"
wait_for_http "Text embedding server" "http://127.0.0.1:${LOCAL_TEXT_EMBED_PORT}/status"
