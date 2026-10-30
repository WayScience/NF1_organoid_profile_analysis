#!/usr/bin/env python
# coding: utf-8

# This notebook applies each patient's trained DMSO-vs-Staurosporine MLP classifier (`1.train_binary_mlp_control.ipynb`) to every cell for that patient, across all treatments and doses - not just the DMSO/Staurosporine control wells used for training.
#
# For every cell this produces:
# - `Metadata_predicted_cytotoxic_probability` - the classifier's P(cytotoxic-like), i.e. how Staurosporine-like the cell's morphology looks.
# - `Metadata_distance_from_dmso_centroid` - the cell's Euclidean distance, in that patient's standardized feature space, from the centroid of their DMSO training cells.
#
# Cells that differ substantially from DMSO (high distance) without looking cytotoxic (low predicted probability) are candidates for a treatment-specific, non-toxic morphological response. This notebook only computes and saves these per-cell scores; plotting is done separately in `5.plot_binary_mlp_control_inference.ipynb` (R).

# In[1]:


import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from notebook_init_utils import init_notebook
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

root_dir, in_notebook = init_notebook()
progress_bar = tqdm_notebook if in_notebook else tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


# In[2]:


profile_file_path = pathlib.Path(
    f"{root_dir}/data/profiles_3D/all_patients/1.feature_selected_profiles/sc_norm_fs_profiles.parquet"
).resolve(strict=True)

split_idx_file_path = pathlib.Path(
    f"{root_dir}/6.treatment_predictive_model/data_splits/sc_norm_fs_profiles_split_idx.parquet"
).resolve(strict=True)

model_output_dir = pathlib.Path(
    f"{root_dir}/6.treatment_predictive_model/trained_models"
).resolve(strict=True)
model_results_dir = pathlib.Path(
    f"{root_dir}/6.treatment_predictive_model/model_results"
).resolve()
model_results_dir.mkdir(parents=True, exist_ok=True)


# ## Load data
# `df` is scored in full (every treatment/dose). `train_df` is rebuilt exactly as in `1.train_binary_mlp_control.ipynb` / `2.evaluate_binary_mlp_control.ipynb`, restricted to the DMSO/Staurosporine training rows, and is used only to refit each patient's `StandardScaler` and compute their DMSO centroid - never to select which rows get scored.

# In[3]:


PATIENT_COL = "Metadata_Biology_PatientTumor"
TREATMENT_COL = "Metadata_Experiment_Treatment"
LABEL_COL = "Metadata_control_label"
CONTROL_LABEL_MAP = {"DMSO": 0, "Staurosporine": 1}

indices_df = pd.read_parquet(split_idx_file_path)
df = pd.read_parquet(profile_file_path)

feature_cols = [col for col in df.columns if not col.startswith("Metadata_")]

# train_idx are positions into the UNFILTERED profile file (see
# 0.generate_data_splits.ipynb) - iloc on the still-unfiltered df before any
# row is dropped below, or the positions would no longer line up.
train_idx = indices_df["train_idx"].dropna().astype(int).tolist()
train_df = df.iloc[train_idx]
train_df = train_df[train_df[TREATMENT_COL].isin(CONTROL_LABEL_MAP)].copy()
train_df[LABEL_COL] = train_df[TREATMENT_COL].map(CONTROL_LABEL_MAP).astype(np.float32)

# Rows with no treatment/dose annotation can't be scored or compared to a
# treatment - exclude them from the set of cells to score. All feature
# columns are complete (no NaNs), so no rows are dropped on that account.
df = df.dropna(subset=feature_cols + [TREATMENT_COL, "Metadata_Experiment_Dose"])

print(f"Scoring {len(df)} cells total across {df[PATIENT_COL].nunique()} patients")
print(f"Treatments: {sorted(df[TREATMENT_COL].unique())}")


# ## Model architecture
# Must match `1.train_binary_mlp_control.ipynb` exactly so the saved `state_dict`s load correctly.

# In[ ]:


# hidden dims for the MLP classifier for the given
# middle input/output dimensions
HIDDEN_DIMS = (128, 64)


class MLPClassifier(nn.Module):
    """A feed-forward binary classifier over single-cell morphology features."""

    def __init__(self, input_dim: int, hidden_dims: tuple = HIDDEN_DIMS):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ## Score every cell for every patient with a trained model
# `NF0016_T1` has no `Staurosporine` wells, so it has no saved model and is skipped here too (consistent with `1.train_binary_mlp_control.ipynb` / `2.evaluate_binary_mlp_control.ipynb`).

# In[5]:


metadata_cols = [PATIENT_COL, TREATMENT_COL, "Metadata_Experiment_Dose"]
patients = sorted(df[PATIENT_COL].unique())
per_cell_records = []

for patient in progress_bar(patients, desc="Scoring patients"):
    model_path = model_output_dir / f"binary_mlp_control_{patient}.pt"
    if not model_path.exists():
        print(f"Skipping {patient}: no saved model (skipped during training).")
        continue

    patient_train_control = train_df[train_df[PATIENT_COL] == patient]
    patient_all = df[df[PATIENT_COL] == patient]

    # Refit on the training rows only - deterministic given the same input,
    # so this reproduces the exact transform used when the model was trained.
    scaler = StandardScaler()
    scaler.fit(patient_train_control[feature_cols].values)
    X_all = scaler.transform(patient_all[feature_cols].values)

    dmso_train_mask = patient_train_control[LABEL_COL] == 0
    dmso_centroid = scaler.transform(
        patient_train_control.loc[dmso_train_mask, feature_cols].values
    ).mean(axis=0)
    distance_from_dmso = np.linalg.norm(X_all - dmso_centroid, axis=1)

    model = MLPClassifier(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_all, dtype=torch.float32, device=device))
    probs = torch.sigmoid(logits).cpu().numpy()

    patient_records = patient_all[metadata_cols].copy()
    patient_records["Metadata_predicted_cytotoxic_probability"] = probs
    patient_records["Metadata_distance_from_dmso_centroid"] = distance_from_dmso
    per_cell_records.append(patient_records)

inference_df = pd.concat(per_cell_records, ignore_index=True)
inference_df.head()


# In[6]:


inference_output_path = (
    model_results_dir / "binary_mlp_control_all_treatments_inference.parquet"
)
inference_df.to_parquet(inference_output_path, index=False)


# ## Per-treatment summary
# Mean/median predicted cytotoxic probability and distance-from-DMSO, pooled across patients and also broken out per patient, for easy plotting in `5.plot_binary_mlp_control_inference.ipynb`.

# In[7]:


agg_spec = {
    "Metadata_predicted_cytotoxic_probability": ["mean", "median", "std"],
    "Metadata_distance_from_dmso_centroid": ["mean", "median", "std"],
}

per_treatment_summary = inference_df.groupby(TREATMENT_COL).agg(agg_spec)
per_treatment_summary.columns = ["_".join(col) for col in per_treatment_summary.columns]
per_treatment_summary["n_cells"] = inference_df.groupby(TREATMENT_COL).size()
per_treatment_summary = per_treatment_summary.reset_index()

per_patient_treatment_summary = inference_df.groupby([PATIENT_COL, TREATMENT_COL]).agg(
    agg_spec
)
per_patient_treatment_summary.columns = [
    "_".join(col) for col in per_patient_treatment_summary.columns
]
per_patient_treatment_summary["n_cells"] = inference_df.groupby(
    [PATIENT_COL, TREATMENT_COL]
).size()
per_patient_treatment_summary = per_patient_treatment_summary.reset_index()

per_treatment_summary.to_parquet(
    model_results_dir / "binary_mlp_control_per_treatment_summary.parquet", index=False
)
per_patient_treatment_summary.to_parquet(
    model_results_dir / "binary_mlp_control_per_patient_treatment_summary.parquet",
    index=False,
)
per_treatment_summary.sort_values(
    "Metadata_predicted_cytotoxic_probability_mean", ascending=False
)
