#!/bin/bash

# --- Configuration ---
# Define the commands to run. We use 'sleep' as a placeholder for long-running tasks.
# Replace these with the actual commands you want to run.
# CMD1="uv run gunicorn mlx_omni_server.main:app --bind 0.0.0.0:10240 --worker-class uvicorn.workers.UvicornWorker --workers 2 --log-level info --env ALLOWED_MODEL=mlx-community/multilingual-e5-large"
# CMD2="uv run gunicorn mlx_omni_server.main:app --bind 0.0.0.0:10241 --worker-class uvicorn.workers.UvicornWorker --workers 1 --log-level info --env ALLOWED_MODEL=mlx-community/gpt-oss-120b-MXFP4-Q4"
# CMD3="uv run gunicorn mlx_omni_server.main:app --bind 0.0.0.0:10242 --worker-class uvicorn.workers.UvicornWorker --workers 1 --log-level info --env ALLOWED_MODEL=Qwen/Qwen3-30B-A3B-MLX-4bit"

MODEL1="mlx-community/multilingual-e5-large"
CMD1="uv run uvicorn mlx_omni_server.main:app --host 0.0.0.0 --port 10240 --log-level warning --workers 2"
# ALT: "./.venv/bin/python -m mlx_omni_server.main --host 0.0.0.0 --port 10240 --log-level warning --workers 2"
MODEL2="mlx-community/gpt-oss-120b-MXFP4-Q4"
CMD2="uv run uvicorn mlx_omni_server.main:app --host 0.0.0.0 --port 10241 --log-level warning --workers 1"
MODEL3="Qwen/Qwen3-30B-A3B-MLX-4bit"
CMD3="uv run uvicorn mlx_omni_server.main:app --host 0.0.0.0 --port 10242 --log-level warning --workers 1"

# Array to store the Process IDs (PIDs) of the background jobs
PIDS=()
TARGET_PORTS=(10240 10241 10242)

# --- Signal Handler: Cleanup Function ---
# This function is called when the script receives a SIGINT signal (Ctrl+C).
cleanup() {
  echo -e "\n\n[Runner] Caught Ctrl+C signal. Shutting down all parallel processes..."

  # 1. Polite Shutdown using Stored PIDs (SIGTERM)
  if [ ${#PIDS[@]} -gt 0 ]; then
    echo "[Runner] Sent graceful termination (SIGTERM) signal to PIDs: ${PIDS[*]}"
    kill "${PIDS[@]}" 2>/dev/null
  else
    echo "[Runner] No active background processes tracked by PID."
  fi

  # Give processes a moment to shut down gracefully
  sleep 1

  # 2. Force Kill Fallback by Port (SIGKILL)
  # This section ensures that anything still listening on the target ports is killed.
  echo "[Runner] Running force-kill cleanup check on ports ${TARGET_PORTS[*]}..."

  LEFT_OVER_PIDS=""
  for PORT in "${TARGET_PORTS[@]}"; do
    # lsof -t -i :PORT: lists PIDs that are using the port.
    lsof_result=$(lsof -t -i tcp:$PORT 2>/dev/null)
    if [ ! -z "$lsof_result" ]; then
        # Collect PIDs for final message
        LEFT_OVER_PIDS="$LEFT_OVER_PIDS $(echo $lsof_result | tr '\n' ' ')"
        # Use xargs to send SIGKILL (-9) to all discovered PIDs
        echo "$lsof_result" | xargs kill -9 2>/dev/null
    fi
  done

  if [ ! -z "$LEFT_OVER_PIDS" ]; then
    echo "[Runner] Forcefully killed remaining processes on target ports (PIDs: $LEFT_OVER_PIDS)"
  else
    echo "[Runner] All target ports were clean."
  fi

  # Exit the script cleanly
  exit 0
}

# Trap the SIGINT signal (triggered by Ctrl+C) and execute the cleanup function
trap cleanup INT

# --- Process Execution ---

echo "[Runner] Starting processes in parallel..."

# Process 1
echo "[Process 1] Running: $CMD1"
ALLOWED_MODEL="$MODEL1" MLX_OMNI_LOG_LEVEL=warning $CMD1 &
PIDS+=($!) # Add the PID of the last background command ($!) to the PIDS array

# Process 2
echo "[Process 2] Running: $CMD2"
ALLOWED_MODEL="$MODEL2" MLX_OMNI_LOG_LEVEL=warning $CMD2 &
PIDS+=($!) # Add PID to array

# Process 3
echo "[Process 3] Running: $CMD3"
ALLOWED_MODEL="$MODEL3" MLX_OMNI_LOG_LEVEL=warning $CMD3 &
PIDS+=($!) # Add PID to array

echo "[Runner] Successfully started processes."
echo "[Runner] Active PIDs: ${PIDS[*]}"
echo "[Runner] ------------------------------------------------------------------"
echo "[Runner] Monitoring. Press Ctrl+C at any time to kill all processes."
echo "[Runner] ------------------------------------------------------------------"

# 'wait' without arguments waits for all background jobs associated with the shell to finish.
# This keeps the main script running until either all jobs complete naturally (after 60, 70, 80 seconds)
# OR until the 'cleanup' function is triggered by Ctrl+C.
wait

echo "[Runner] All background processes finished naturally."

# Remove the trap just before a successful exit
trap - INT
