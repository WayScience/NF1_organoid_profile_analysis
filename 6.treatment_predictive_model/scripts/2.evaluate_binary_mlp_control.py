#!/usr/bin/env python
# coding: utf-8

# This notebook evaluates the per-patient DMSO-vs-Staurosporine MLP classifiers trained in `1.train_binary_mlp_control.ipynb`.
#
# For each patient, the classifier's held-out test-set performance is re-derived (same train/val/test split, same per-patient feature standardization, saved model weights): per-cell predictions, precision-recall curve points, and pooled/per-patient summary metrics are all computed and saved here as parquet. Plotting is done separately in `3.plot_binary_mlp_control_performance.ipynb` (R).

# In[1]:


import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from notebook_init_utils import init_notebook
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
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


# In[3]:


indices_df = pd.read_parquet(split_idx_file_path)
df = pd.read_parquet(profile_file_path)


# In[4]:


# get the train, validation, and test indices from the dataframe
train_idx = indices_df["train_idx"].dropna().astype(int).tolist()
val_idx = indices_df["val_idx"].dropna().astype(int).tolist()
test_idx = indices_df["test_idx"].dropna().astype(int).tolist()

train_df, val_df, test_df = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


# ## Rebuild the same control-only train/val/test subsets used in training
# This mirrors `1.train_binary_mlp_control.ipynb` exactly, so that refitting each patient's `StandardScaler` on `patient_train` reproduces the identical transform the saved model weights were trained under.

# In[5]:


PATIENT_COL = "Metadata_Biology_PatientTumor"
TREATMENT_COL = "Metadata_Experiment_Treatment"
LABEL_COL = "Metadata_control_label"
CONTROL_LABEL_MAP = {"DMSO": 0, "Staurosporine": 1}

feature_cols = [col for col in df.columns if not col.startswith("Metadata_")]

train_df = train_df[train_df[TREATMENT_COL].isin(CONTROL_LABEL_MAP)].copy()
val_df = val_df[val_df[TREATMENT_COL].isin(CONTROL_LABEL_MAP)].copy()
test_df = test_df[test_df[TREATMENT_COL].isin(CONTROL_LABEL_MAP)].copy()

for split_df in (train_df, val_df, test_df):
    split_df[LABEL_COL] = (
        split_df[TREATMENT_COL].map(CONTROL_LABEL_MAP).astype(np.float32)
    )

print(
    f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows"
)
print(f"Number of features: {len(feature_cols)}")


# ## Model architecture
# Must match `1.train_binary_mlp_control.ipynb` exactly so the saved `state_dict`s load correctly.

# In[6]:


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


# ## Score every held-out test cell with its patient's trained model
# Patients with a saved model (i.e. that had both classes present in every split during training) are scored; `NF0016_T1` has no `Staurosporine` wells and was skipped in training, so it is skipped here too.

# In[7]:


patients = sorted(train_df[PATIENT_COL].unique())
per_cell_records = []

for patient in progress_bar(patients, desc="Scoring per-patient test sets"):
    model_path = model_output_dir / f"binary_mlp_control_{patient}.pt"
    if not model_path.exists():
        print(f"Skipping {patient}: no saved model (skipped during training).")
        continue

    patient_train = train_df[train_df[PATIENT_COL] == patient]
    patient_test = test_df[test_df[PATIENT_COL] == patient]
    if patient_test.empty:
        print(f"Skipping {patient}: no test rows.")
        continue

    # Refit on the training rows only - deterministic given the same input,
    # so this reproduces the exact transform used when the model was trained.
    scaler = StandardScaler()
    scaler.fit(patient_train[feature_cols].values)
    X_test = scaler.transform(patient_test[feature_cols].values)
    y_test = patient_test[LABEL_COL].values

    model = MLPClassifier(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32, device=device))
    probs = torch.sigmoid(logits).cpu().numpy()

    per_cell_records.append(
        pd.DataFrame(
            {
                PATIENT_COL: patient,
                "Metadata_true_label": y_test,
                "Metadata_predicted_probability": probs,
            }
        )
    )

per_cell_df = pd.concat(per_cell_records, ignore_index=True)
per_cell_df["Metadata_predicted_label"] = (
    per_cell_df["Metadata_predicted_probability"] >= 0.5
).astype(int)
per_cell_df.head()


# In[8]:


per_cell_predictions_path = (
    model_results_dir / "binary_mlp_control_test_predictions.parquet"
)
per_cell_df.to_parquet(per_cell_predictions_path, index=False)


# ## Save precision-recall curve data
# More informative than ROC when the two classes are imbalanced (DMSO outnumbers Staurosporine for most patients).

# In[9]:


pr_records = []
for patient, group in per_cell_df.groupby(PATIENT_COL):
    precision, recall, _ = precision_recall_curve(
        group["Metadata_true_label"], group["Metadata_predicted_probability"]
    )
    ap = average_precision_score(
        group["Metadata_true_label"], group["Metadata_predicted_probability"]
    )
    pr_records.append(
        pd.DataFrame(
            {
                "series": patient,
                "recall": recall,
                "precision": precision,
                "average_precision": ap,
            }
        )
    )

pooled_precision, pooled_recall, _ = precision_recall_curve(
    per_cell_df["Metadata_true_label"], per_cell_df["Metadata_predicted_probability"]
)
pooled_ap = average_precision_score(
    per_cell_df["Metadata_true_label"], per_cell_df["Metadata_predicted_probability"]
)
pr_records.append(
    pd.DataFrame(
        {
            "series": "pooled",
            "recall": pooled_recall,
            "precision": pooled_precision,
            "average_precision": pooled_ap,
        }
    )
)

pr_curve_df = pd.concat(pr_records, ignore_index=True)
pr_curve_df.to_parquet(
    model_results_dir / "binary_mlp_control_pr_curve_points.parquet", index=False
)
pr_curve_df.head()


# ## Summary metrics table
# Per-patient and pooled accuracy, F1, ROC-AUC, average precision, and Brier score, computed directly from the re-derived per-cell test predictions above.

# In[10]:


def summarize(group: pd.DataFrame) -> pd.Series:
    y_true = group["Metadata_true_label"]
    y_prob = group["Metadata_predicted_probability"]
    y_pred = group["Metadata_predicted_label"]
    return pd.Series(
        {
            "n_test": len(group),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob)
            if y_true.nunique() > 1
            else np.nan,
            "average_precision": (
                average_precision_score(y_true, y_prob)
                if y_true.nunique() > 1
                else np.nan
            ),
            "brier_score": brier_score_loss(y_true, y_prob),
        }
    )


summary_df = per_cell_df.groupby(PATIENT_COL).apply(summarize, include_groups=False)
summary_df.loc["pooled"] = summarize(per_cell_df)
summary_df = summary_df.reset_index(names=PATIENT_COL)

summary_output_path = (
    model_results_dir / "binary_mlp_control_evaluation_summary.parquet"
)
summary_df.to_parquet(summary_output_path, index=False)
summary_df
