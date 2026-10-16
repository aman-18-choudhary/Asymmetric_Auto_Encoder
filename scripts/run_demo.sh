#!/usr/bin/env bash
# Launch the demo locally: bound to localhost only, usage statistics off. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-.venv/bin/python}" -m streamlit run demo/app.py \
  --server.address localhost --browser.gatherUsageStats false "$@"
