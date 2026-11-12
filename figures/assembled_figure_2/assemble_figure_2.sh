#!/bin/bash

set -eo pipefail

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

module_dir="$git_root/figures/assembled_figure_2"


# regenerate scripts/ from notebooks/ so the mirrored .py/.r files never drift
uv run jupyter nbconvert --to=script \
    --FilesWriter.build_directory="$module_dir"/scripts/ \
    "$module_dir"/notebooks/*.ipynb

# deactivate any existing conda environment
conda deactivate >/dev/null 2>&1 || true
# deactivate any existing venv environment
deactivate >/dev/null 2>&1 || true

uv run python "$module_dir/scripts/featurization_figure.py"
uvr run "$module_dir/scripts/assembled_figure_2.r"
