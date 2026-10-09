#!/bin/bash

# set -eo pipefail

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

module_dir="$git_root/4.linear_modeling"
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
    --FilesWriter.build_directory="$module_dir/scripts/" \
    "$module_dir"/notebooks/*.ipynb

# deactivate any existing conda environment
conda deactivate >/dev/null 2>&1 || true
# deactivate any existing venv environment
deactivate >/dev/null 2>&1 || true


# Linear modeling steps
# pre-processing, data cleaning, and covariate construction
# linear modeling base model
# linear modeling with technical variables
uv run python "$module_dir/scripts/0.aggregating_non_fs_profiles.py"
uv run python "$module_dir/scripts/1.calculate_well_manhattan_distance.py"
uv run python "$module_dir/scripts/2.linear_modeling.py"
uv run python "$module_dir/scripts/3.linear_modeling_technical_vars.py"

# variance decomposition
uv run python "$module_dir/scripts/4.variance_decomposition.py"

# variate importance
uv run python "$module_dir/scripts/5.calculate_variate_importance.py"
uvr run "$module_dir/scripts/6.plot_variate_importance.r"  # panels C and D of 11 use this script's title-free multiresult_figure_subpanels PNGs

# variate class membership / clustermap
uv run python "$module_dir/scripts/7.calculate_variate_class_upsets_and_clustermap.py"
uvr run "$module_dir/scripts/8.plot_variate_class_upsets_and_clustermap.r"

# additional insights
uv run python "$module_dir/scripts/9.explore_linear_model_haystacks.py"
uvr run "$module_dir/scripts/10.plot_explore_linear_model_haystacks.r"

# 11: multi-panel summary figure (patchwork), built from the outputs of 6 and 9
uvr run "$module_dir/scripts/11.assembled_multiresult_figure.r"


echo "4.linear_modeling pipeline complete."
