#!/bin/bash

# Compatibility wrapper for the original DGX script name. The actual local text
# service startup lives in scripts/start_local_text_servers.sh and is shared by
# DGX and CloudLab runs.

set -e

export LOCAL_GPU_ID=${LOCAL_GPU_ID:-${DGX_GPU_ID:-0}}
export LOCAL_MODEL_NAME=${LOCAL_MODEL_NAME:-${DGX_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}}
export KILL_SCRIPT=${KILL_SCRIPT:-scripts/kill_dgx_text_servers.sh}

scripts/start_local_text_servers.sh
