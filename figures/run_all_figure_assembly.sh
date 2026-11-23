#!/bin/bash

# set -euo pipefail

git_root=$(git rev-parse --show-toplevel)

bash "$git_root/figures/assembled_figure_1/run_assembled_figure_generation.sh"
bash "$git_root/figures/assembled_figure_2/assemble_figure_2.sh"
bash "$git_root/figures/assembled_figure_3/run_figure3_assembly.sh"
bash "$git_root/figures/assembled_figure_4.5/run_figure_generation.sh"
bash "$git_root/figures/clinical_trial_data_plots/run_clinical_trial_figures_local.sh"
bash "$git_root/figures/drugs_figure/run_drugs_figure_local.sh"

echo "All figure assembly scripts have been run."
