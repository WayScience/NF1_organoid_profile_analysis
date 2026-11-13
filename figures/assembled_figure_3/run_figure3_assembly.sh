#!/bin/bash

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/figures/assembled_figure_3/scripts/ "$git_root"/figures/assembled_figure_3/notebooks/*.ipynb

# deactivate any existing conda environment
conda deactivate
# deactivate any existing venv environment
deactivate 2>/dev/null

uv run python "$git_root"/figures/assembled_figure_3/scripts/umap_image_frame.py
uvr run "$git_root"/figures/assembled_figure_3/scripts/assembled_figure_3.r

