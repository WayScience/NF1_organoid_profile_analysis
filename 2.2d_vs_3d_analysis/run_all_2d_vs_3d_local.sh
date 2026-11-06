#!/bin/bash

set -eo pipefail

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

uv run jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/2.2d_vs_3d_analysis/scripts/ "$git_root"/2.2d_vs_3d_analysis/notebooks/*.ipynb

# deactivate any existing conda environment
conda deactivate
# deactivate any existing venv environment
deactivate 2>/dev/null

uv run python "$git_root"/2.2d_vs_3d_analysis/scripts/1.generate_patient_correlation.py
uvr run "$git_root"/2.2d_vs_3d_analysis/scripts/2.plot_patient_correlation_heatmaps.r
uv run python "$git_root"/2.2d_vs_3d_analysis/scripts/3.calculate_inter_intra_patient_distances.py
uv run python "$git_root"/2.2d_vs_3d_analysis/scripts/4.calculate_mAP.py
uvr run "$git_root"/2.2d_vs_3d_analysis/scripts/5.plot_metrics_and_mAP.r
uv run python "$git_root"/2.2d_vs_3d_analysis/scripts/6.calculate_shannon_entropy.py
uvr run "$git_root"/2.2d_vs_3d_analysis/scripts/7.plot_shannon_entropy.r
uvr run "$git_root"/2.2d_vs_3d_analysis/scripts/8.kbet_analysis.r
uvr run "$git_root"/2.2d_vs_3d_analysis/scripts/9.sparse_cca_analysis.r
uvr run "$git_root"/2.2d_vs_3d_analysis/scripts/10.plot_sparse_cca_results.r
