#!/bin/bash
# Launch this app from anywhere inside this repo.
# Usage: bash run_app.sh [port]   (default port: 8501)
#
# First time: pip install -r requirements.txt (or use a venv/uv for that).

PORT="${1:-8501}"
GIT_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

python -m streamlit run "$GIT_ROOT"/app.py --server.port "${PORT}" --theme.base light
