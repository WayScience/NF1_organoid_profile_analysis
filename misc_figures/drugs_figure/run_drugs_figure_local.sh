#!/bin/bash

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/figures/drugs_figure/scripts/ "$git_root"/figures/drugs_figure/notebooks/*.ipynb

# deactivate any existing conda environment
conda deactivate
# deactivate any existing venv environment
deactivate 2>/dev/null

conda run -n GFF_analysis python "$git_root"/figures/drugs_figure/scripts/drugs_figure.py

# platemap_figure.r resolves its input relative to its own directory
cd "$git_root"/figures/drugs_figure/scripts || exit
conda run -n gff_figure_env Rscript platemap_figure.r
