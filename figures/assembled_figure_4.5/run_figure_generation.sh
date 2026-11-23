#!/bin/bash

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/figures/assembled_figure_4.5/scripts/ "$git_root"/figures/assembled_figure_4.5/notebooks/*.ipynb

# "try" block: deactivate any active environments, ignore failures
{
    # deactivate any existing conda environment
    conda deactivate 2>/dev/null
    # deactivate any existing venv environment
    deactivate 2>/dev/null
} || true
uvr run "$git_root"/figures/assembled_figure_4.5/scripts/assembled_figure.r

