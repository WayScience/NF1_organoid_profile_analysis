#!/bin/bash

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/1.EDA/scripts/ "$git_root"/1.EDA/notebooks/*.ipynb

# deactivate any existing conda environment
conda deactivate
# deactivate any existing venv environment
deactivate 2>/dev/null

conda run -n GFF_analysis python "$git_root"/1.EDA/scripts/0.generate_umap.py
conda run -n gff_figure_env Rscript "$git_root"/1.EDA/scripts/1.plot_umap.r
conda run -n GFF_analysis python "$git_root"/1.EDA/scripts/2.generate_pca.py
conda run -n gff_figure_env Rscript "$git_root"/1.EDA/scripts/3.plot_pca.r
conda run -n GFF_analysis python "$git_root"/1.EDA/scripts/4.calculate_correlation_matrix.py
conda run -n gff_figure_env Rscript "$git_root"/1.EDA/scripts/5.plot_correlation_heatmaps.r
conda run -n GFF_analysis python "$git_root"/1.EDA/scripts/7.generate_cell_counts.py
conda run -n gff_figure_env Rscript "$git_root"/1.EDA/scripts/8.plot_single_cell_counts.r
