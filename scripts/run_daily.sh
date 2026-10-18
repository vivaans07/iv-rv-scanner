#!/usr/bin/env bash
# Nightly entry point for cron or launchd. Paths are relative to the repo so
# the script works wherever the project is cloned.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run volscan)
elif [ -x .venv/bin/volscan ]; then
  RUNNER=(.venv/bin/volscan)
else
  echo "volscan is not installed; run 'uv sync' first" >&2
  exit 1
fi

# run-daily exits quietly on weekends and NYSE holidays.
"${RUNNER[@]}" run-daily "$@" >>logs/cron.log 2>&1
