#!/bin/bash

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

uv run jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/figures/assembled_figures/scripts/ "$git_root"/figures/assembled_figures/notebooks/*.ipynb

# deactivate any existing conda environment
conda deactivate
# deactivate any existing venv environment
deactivate 2>/dev/null

uv run python "$git_root"/figures/assembled_figures/scripts/full_figure1.py

