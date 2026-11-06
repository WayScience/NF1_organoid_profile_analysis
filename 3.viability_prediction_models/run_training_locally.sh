#!/bin/bash

git_root=$(git rev-parse --show-toplevel)

uv run jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/3.viability_prediction_models/scripts/ "$git_root"/3.viability_prediction_models/notebooks/*.ipynb

echo "Running pre-processing..."
uv run python "$git_root"/3.viability_prediction_models/scripts/0.pre-processing_profiles_for_viability_models.py

echo "Running viability prediction model training..."
uv run python "$git_root"/3.viability_prediction_models/scripts/1.viability_prediction.py

echo "Running model results visualization..."
uvr run "$git_root"/3.viability_prediction_models/scripts/2.visualize_model_results.r

echo "Visualizing single-cell data..."
uv run python "$git_root"/3.viability_prediction_models/scripts/3.single-cell_visualizations.py

echo "Running multi-panel summary figure..."
uvr run "$git_root"/3.viability_prediction_models/scripts/4.multi_panel_summary_figure.r
