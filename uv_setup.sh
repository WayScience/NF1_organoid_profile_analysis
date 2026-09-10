#!/bin/bash

git_root=$(git rev-parse --show-toplevel)

ENV_PATH="$git_root/.venv"


if [[ -e "$ENV_PATH" ]]; then
    echo "Using existing virtual environment at $ENV_PATH"
else
    mkdir -p "$ENV_PATH"
    echo "Created virtual environment directory at $ENV_PATH"
fi

# run twice to ensure we are in a clean environment
# and not accidentally using an existing one
# try but will not fail if not in a conda environment
loops=2
for _ in $(seq 1 $loops); do
    if command -v conda >/dev/null 2>&1; then
        conda deactivate >/dev/null 2>&1 || true
    fi
    deactivate >/dev/null 2>&1 || true
done
unset VIRTUAL_ENV

rm -f uv.lock
rm -rf .venv

uv venv
uv sync

# shellcheck disable=SC1091
source .venv/bin/activate
# Use RELATIVE path - simple and reliable
uv pip install -e ./utils
uv pip install 'autogluon.tabular[mitra]'

echo "Virtual environment setup complete. To activate it, run:"
echo "source $ENV_PATH/bin/activate"
