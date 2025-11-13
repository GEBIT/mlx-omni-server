#!/bin/bash

echo "Current Working Directory (inside script):"
echo "$(pwd)"
#source .venv/bin/activate || { echo "Failed to activate venv" >&2; exit 1; }
# uv run uvicorn mlx_omni_server.main:app --host 0.0.0.0 --port 10240
~/.local/bin/uv run gunicorn mlx_omni_server.main:app --bind 0.0.0.0:10240 --worker-class uvicorn.workers.UvicornWorker --workers 1 --log-level info
