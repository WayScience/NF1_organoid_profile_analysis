#!/bin/bash

set -eo pipefail

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

module_dir="$git_root/6.technical_platemap_analysis/"
venv_python="$git_root/.venv/bin/python"

if [[ ! -x "$venv_python" ]]; then
    echo "Error: uv-managed virtual environment not found at $git_root/.venv (run uv_setup.sh first)."
    exit 1
fi

if ! command -v uvr >/dev/null 2>&1; then
    echo "Error: uvr not found (needed to run the R plotting notebooks)."
    exit 1
fi

# regenerate scripts/ from notebooks/ so the mirrored .py/.r files never drift
uv run jupyter nbconvert --to=script \
    --FilesWriter.build_directory="$module_dir"/scripts/ \
    "$module_dir"/notebooks/*.ipynb

# deactivate any existing conda environment
conda deactivate >/dev/null 2>&1 || true
# deactivate any existing venv environment
deactivate >/dev/null 2>&1 || true

uvr run  "$module_dir/scripts/0.plot_PCs_per_well.r"
uvr run  "$module_dir/scripts/1.plot_PCs_against_z_dim.r"

