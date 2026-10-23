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

# Misc figures: drug FDA-approval status and platemap figures (repo-root figures/ dir).
drugs-figure:
    bash misc_figures/drugs_figure/run_drugs_figure_local.sh

# update the data viewing app and associated data
update-app:
    (cd interactive_dashboard && just all_data)

# Run every module end to end. Pass clean=true to wipe outputs first.
all clean="false": (clean clean) setup download-data eda viability linear-modeling drugs-figure update-app
