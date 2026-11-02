#!/usr/bin/env python
# coding: utf-8

# This notebook evaluates the per-patient DMSO-vs-Staurosporine MLP classifiers trained in `1.train_binary_mlp_control.ipynb`.
#
# For each patient, the classifier's held-out test-set performance is re-derived (same train/val/test split, saved `MLPClassifier` + `StandardScaler` loaded from the saved model file): per-cell predictions, precision-recall curve points, and pooled/per-patient summary metrics are all computed and saved here as parquet. Plotting is done separately in `3.plot_binary_mlp_control_performance.ipynb` (R).

# In[1]:


import pathlib

import joblib
import numpy as np
import pandas as pd
from notebook_init_utils import init_notebook
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

root_dir, in_notebook = init_notebook()
progress_bar = tqdm_notebook if in_notebook else tqdm


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


# In[ ]:


# get the train, validation, and test indices from the dataframe
train_idx = indices_df["train_idx"].dropna().astype(int).tolist()
val_idx = indices_df["val_idx"].dropna().astype(int).tolist()
test_idx = indices_df["test_idx"].dropna().astype(int).tolist()

train_df, val_df, test_df = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


# ## Rebuild the same control-only train/val/test subsets used in training
# This mirrors `1.train_binary_mlp_control.ipynb` exactly, so `test_df` matches the held-out rows used to report test metrics during training. Each patient's `StandardScaler` is loaded directly from the saved model file rather than refit.

# In[ ]:


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


# ## Score every held-out test cell with its patient's trained model
# Patients with a saved model (i.e. that had both classes present in every split during training) are scored; `NF0016_T1` has no `Staurosporine` wells and was skipped in training, so it is skipped here too.

# In[ ]:


patients = sorted(train_df[PATIENT_COL].unique())
per_cell_records = []

for patient in progress_bar(patients, desc="Scoring per-patient test sets"):
    model_path = model_output_dir / f"binary_mlp_control_{patient}.joblib"
    if not model_path.exists():
        print(f"Skipping {patient}: no saved model (skipped during training).")
        continue

    patient_test = test_df[test_df[PATIENT_COL] == patient]
    if patient_test.empty:
        print(f"Skipping {patient}: no test rows.")
        continue

    saved = joblib.load(model_path)
    model, scaler = saved["model"], saved["scaler"]

    X_test = scaler.transform(patient_test[feature_cols].values)
    y_test = patient_test[LABEL_COL].values
    probs = model.predict_proba(X_test)[:, 1]

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


# In[ ]:


per_cell_predictions_path = (
    model_results_dir / "binary_mlp_control_test_predictions.parquet"
)
per_cell_df.to_parquet(per_cell_predictions_path, index=False)


# ## Save precision-recall curve data
# More informative than ROC when the two classes are imbalanced (DMSO outnumbers Staurosporine for most patients).

# In[ ]:


def pr_curve_frame(group: pd.DataFrame) -> pd.DataFrame:
    precision, recall, _ = precision_recall_curve(
        group["Metadata_true_label"], group["Metadata_predicted_probability"]
    )
    ap = average_precision_score(
        group["Metadata_true_label"], group["Metadata_predicted_probability"]
    )
    return pd.DataFrame(
        {"recall": recall, "precision": precision, "average_precision": ap}
    )


pr_curve_df = (
    per_cell_df.groupby(PATIENT_COL, group_keys=True)
    .apply(pr_curve_frame, include_groups=False)
    .reset_index(level=0)
    .rename(columns={PATIENT_COL: "series"})
    .reset_index(drop=True)
)

pooled_pr_curve_df = pr_curve_frame(per_cell_df)
pooled_pr_curve_df.insert(0, "series", "pooled")

pr_curve_df = pd.concat([pr_curve_df, pooled_pr_curve_df], ignore_index=True)
pr_curve_df.to_parquet(
    model_results_dir / "binary_mlp_control_pr_curve_points.parquet", index=False
)
pr_curve_df.head()


# ## Summary metrics table
# Per-patient and pooled accuracy, F1, ROC-AUC, average precision, and Brier score, computed directly from the re-derived per-cell test predictions above.

# In[ ]:


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
