alias help := default

default:
    @just --list

# Create/update the local uv virtual environment.
setup:
    ./uv_setup.sh

# Module 0: unpack the raw data archive.
# note that the data needs to be downloaded and unpacked before running any analysis.
download-data:
    ./0.download_data/run_setup_files_local.sh

# Module 1: exploratory data analysis (UMAP, PCA, correlations, cell/intensity/volume figures).
eda:
    ./1.EDA/run_all_eda_local.sh

# Module 2: 2D vs 3D organoid comparison (patient correlation, mAP, entropy, kBET, sparse CCA).
twod-vs-threed:
    ./2.2d_vs_3d_analysis/run_all_2d_vs_3d_local.sh

# Module 3: viability prediction model training.
viability:
    ./3.viability_prediction_models/run_training_locally.sh

# Module 4: linear modeling of profile features.
linear-modeling:
    ./4.linear_modeling/run_all_linear_modeling_local.sh

# Misc figures: drug FDA-approval status and platemap figures (repo-root figures/ dir).
drugs-figure:
    ./figures/drugs_figure/run_drugs_figure_local.sh

# Misc figures: clinical trial efficacy and response heterogeneity (repo-root figures/ dir).
clinical-trial-figures:
    ./figures/clinical_trial_data_plots/run_clinical_trial_figures_local.sh

# Run every module end to end, in pipeline order.
all: setup download-data eda twod-vs-threed viability linear-modeling drugs-figure clinical-trial-figures
