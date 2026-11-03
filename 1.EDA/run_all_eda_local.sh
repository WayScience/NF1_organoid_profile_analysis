#!/bin/bash
set -eo pipefail

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

patient_array_file_path="$git_root/data/patient_IDs.txt"
# read the patient IDs from the file into an array
if [[ -f "$patient_array_file_path" ]]; then
    # shellcheck disable=SC2034
    readarray -t patient_array < "$patient_array_file_path"
else
    echo "Error: File $patient_array_file_path does not exist."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/1.EDA/scripts/ "$git_root"/1.EDA/notebooks/*.ipynb

# deactivate any existing conda environment
conda deactivate 2>/dev/null || true
# deactivate any existing venv environment
deactivate 2>/dev/null || true

uv run python "$git_root"/1.EDA/scripts/0.generate_umap.py
uvr run "$git_root"/1.EDA/scripts/1.plot_umap.r
uv run python "$git_root"/1.EDA/scripts/2.generate_pca.py
uvr run "$git_root"/1.EDA/scripts/3.plot_pca.r
uv run python "$git_root"/1.EDA/scripts/4.calculate_correlation_matrix.py
uvr run "$git_root"/1.EDA/scripts/5.plot_correlation_heatmaps.r
uv run python "$git_root"/1.EDA/scripts/5a.find_correlation_pairs_for_montages.py
uvr run "$git_root"/1.EDA/scripts/5b.plot_correlation_and_montages.r
uv run python "$git_root"/1.EDA/scripts/5c.plot_correlation_vs_viability_subsets.py
uv run python "$git_root"/1.EDA/scripts/7.generate_cell_counts.py
uvr run "$git_root"/1.EDA/scripts/8.plot_single_cell_counts.r
uvr run "$git_root"/1.EDA/scripts/9.plot_volume_area_distributions.r
uv run python "$git_root"/1.EDA/scripts/10.calculate_count_viability_join.py
uvr run "$git_root"/1.EDA/scripts/11.plot_count_viability_scatter.r
uvr run "$git_root"/1.EDA/scripts/12.plot_volume_area_vs_count.r
uv run python "$git_root"/1.EDA/scripts/13.calculate_neighbor_features.py
uvr run "$git_root"/1.EDA/scripts/14.plot_neighbor_features.r
uv run python "$git_root"/1.EDA/scripts/15.calculate_intensity_values.py
uvr run "$git_root"/1.EDA/scripts/16.plot_intensity_facets.r
uv run python "$git_root"/1.EDA/scripts/17.calculate_area_volume_by_patient_treatment.py
uvr run "$git_root"/1.EDA/scripts/18.plot_area_vs_volume.r
