#!/usr/bin/env bash

set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
backend_dir="$project_dir/backend"
frontend_dir="$project_dir/frontend"

export BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"
export ENV_FILE="${BACKEND_ENV_FILE:-$backend_dir/.env}"
export PYTHONPATH="$backend_dir/app${PYTHONPATH:+:$PYTHONPATH}"

export FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
export FRONTEND_PORT="${FRONTEND_PORT:-5173}"
export VITE_API_TARGET="${VITE_API_TARGET:-http://127.0.0.1:$BACKEND_PORT}"
export NODE_ENV="${NODE_ENV:-development}"

(
    cd "$backend_dir/app"
    uvicorn main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
) &
backend_pid=$!

(
    cd "$frontend_dir"
    npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
) &
frontend_pid=$!

stop_services() {
    kill "$backend_pid" "$frontend_pid" 2>/dev/null || true
    wait "$backend_pid" "$frontend_pid" 2>/dev/null || true
}

trap stop_services EXIT INT TERM

while kill -0 "$backend_pid" 2>/dev/null \
    && kill -0 "$frontend_pid" 2>/dev/null; do
    sleep 0.1
done

set +e
if ! kill -0 "$backend_pid" 2>/dev/null; then
    wait "$backend_pid"
    exit_code=$?
else
    wait "$frontend_pid"
    exit_code=$?
fi
set -e

exit "$exit_code"
