#!/bin/sh

set -eu

if [ -z "${MODEL_PATH:-}" ]; then
    echo "Error: MODEL_PATH must be set to a GGUF model file." >&2
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH does not exist or is not a file: $MODEL_PATH" >&2
    exit 1
fi

exec llama-server \
    --model "$MODEL_PATH" \
    --host "${LLAMA_SERVER_HOST:-127.0.0.1}" \
    --port "${LLAMA_SERVER_PORT:-8080}" \
    --ctx-size "${LLAMA_CONTEXT_SIZE:-8192}" \
    --n-gpu-layers "${LLAMA_GPU_LAYERS:-99}" \
    --jinja
