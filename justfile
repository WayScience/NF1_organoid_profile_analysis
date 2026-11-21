alias help := default

default:
    @just --list

# Create/update the local uv virtual environment.
setup:
    bash uv_setup.sh

# Remove generated results/ and figures/ dirs. Dry run (list only) unless delete=true.
clean delete="false":
    bash 0.download_data/zero_out_repo.sh {{delete}}

# Module 0: unpack the raw data archive.
# note that the data needs to be downloaded and unpacked before running any analysis.
download-data:
    bash 0.download_data/run_setup_files_local.sh

# Module 1: exploratory data analysis (UMAP, PCA, correlations, cell/intensity/volume figures).
eda:
    bash 1.EDA/run_all_eda_local.sh

# Module 2: 2D vs 3D organoid comparison (patient correlation, mAP, entropy, kBET, sparse CCA).
twod-vs-threed:
    bash 2.2d_vs_3d_analysis/run_all_2d_vs_3d_local.sh

# Module 3: viability prediction model training.
viability:
    bash 3.viability_prediction_models/run_training_locally.sh

# Module 4: linear modeling of profile features.
linear-modeling:
    bash 4.linear_modeling/run_all_linear_modeling_local.sh

# module 5: differential analysis
differential-analysis:
    bash 5.differential_analysis/run_all_differential_analyses.sh

# module 6: technical platemap analysis
platemap-analysis:
    bash 6.technical_platemap_analysis/run_all_technical_analyses.sh

# module 7: treatment prediction analysis
treatment-prediction:
    bash 7.treatment_predictive_model/run_training_locally.sh

# Misc figures: drug FDA-approval status and platemap figures (repo-root figures/ dir).
drugs-figure:
    bash misc_figures/drugs_figure/run_drugs_figure_local.sh

# update the data viewing app and associated data
update-app:
    (cd interactive_dashboard && just all_data)

# Misc figures: clinical trial efficacy and response heterogeneity (repo-root figures/ dir).
clinical-trial-figures:
    ./figures/clinical_trial_data_plots/run_clinical_trial_figures_local.sh

# Run every module end to end, in pipeline order. Logs full output to logs/all_<timestamp>.log.
all:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p logs
    logfile="logs/all_$(date +%Y%m%d_%H%M%S).log"
    just setup download-data eda twod-vs-threed viability linear-modeling differential-analysis platemap-analysis treatment-prediction drugs-figure clinical-trial-figures 2>&1 | tee "$logfile"

# Run every module end to end, in pipeline order.
all-clean:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p logs
    logfile="logs/all_clean_$(date +%Y%m%d_%H%M%S).log"
    { just setup; just clean true; just download-data eda twod-vs-threed viability linear-modeling differential-analysis platemap-analysis treatment-predictionw drugs-figure clinical-trial-figures; } 2>&1 | tee "$logfile"

