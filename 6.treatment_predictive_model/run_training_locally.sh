#!/bin/bash

git_root=$(git rev-parse --show-toplevel)

# regenerate scripts/ from notebooks/ so the mirrored .py/.r files never drift
uv run jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/6.treatment_predictive_model/scripts/ "$git_root"/6.treatment_predictive_model/notebooks/*.ipynb

echo "Generating data splits..."
uv run python "$git_root"/6.treatment_predictive_model/scripts/0.generate_data_splits.py

echo "Training binary MLP control (DMSO vs. Staurosporine) classifiers..."
uv run python "$git_root"/6.treatment_predictive_model/scripts/1.train_binary_mlp_control.py

echo "Evaluating binary MLP control classifiers..."
uv run python "$git_root"/6.treatment_predictive_model/scripts/2.evaluate_binary_mlp_control.py

echo "Plotting binary MLP control performance..."
uvr run "$git_root"/6.treatment_predictive_model/scripts/3.plot_binary_mlp_control_performance.r

echo "Running binary MLP control inference on all treatments..."
uv run python "$git_root"/6.treatment_predictive_model/scripts/4.infer_binary_mlp_control.py

echo "Plotting binary MLP control inferences..."
uvr run "$git_root"/6.treatment_predictive_model/scripts/5.plot_binary_mlp_control_inference.r

echo "Training multiclass treatment+dose MLP classifier..."
uv run python "$git_root"/6.treatment_predictive_model/scripts/6.train_multiclass_mlp_treatment_dose.py

echo "Evaluating multiclass treatment+dose MLP classifier..."
uv run python "$git_root"/6.treatment_predictive_model/scripts/7.evaluate_multiclass_mlp_treatment_dose.py

echo "Plotting multiclass treatment+dose MLP performance..."
uvr run "$git_root"/6.treatment_predictive_model/scripts/8.plot_multiclass_mlp_treatment_dose.r

echo "Training multiclass target/MOA MLP classifier..."
uv run python "$git_root"/6.treatment_predictive_model/scripts/9.train_multiclass_mlp_target.py

echo "Evaluating multiclass target/MOA MLP classifier..."
uv run python "$git_root"/6.treatment_predictive_model/scripts/10.evaluate_multiclass_mlp_target.py

echo "Plotting multiclass target/MOA MLP performance..."
uvr run "$git_root"/6.treatment_predictive_model/scripts/11.plot_multiclass_mlp_target.r

echo "Assembling publication summary figure..."
uvr run "$git_root"/6.treatment_predictive_model/scripts/12.assembled_figure.r

echo "Assembling supplemental performance figure..."
uvr run "$git_root"/6.treatment_predictive_model/scripts/13.supplemental_performance_figure.r

echo
