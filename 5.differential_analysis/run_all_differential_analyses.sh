#!/bin/bash

set -eo pipefail

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

module_dir="$git_root/5.differential_analysis/"
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



uv run python "$module_dir/scripts/0.calculate_log2_fold_change.py"
uv run python "$module_dir/scripts/0.calculate_log2_fold_change.py"
uvr run "$module_dir/scripts/1.plot_log2_fold_change.r"
uv run "$module_dir/scripts/2.calculate_viability_by_tumor_type.py"
uvr run "$module_dir/scripts/3.plot_viability_by_tumor_type.r"
uv run "$module_dir/scripts/4.calculate_plate_position_effects.py"
uvr run "$module_dir/scripts/5.plot_plate_position_effects.r"
uv run "$module_dir/scripts/6.calculate_well_dmso_correlation.py"
uvr run "$module_dir/scripts/7.plot_well_dmso_correlation.r"
uv run "$module_dir/scripts/8.calculate_mek_signatures.py"
uvr run "$module_dir/scripts/9.plot_mek_signatures.r"

