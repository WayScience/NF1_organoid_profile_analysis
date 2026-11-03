#!/usr/bin/env python
# coding: utf-8

# This notebook evaluates the global treatment+dose multiclass MLP trained in `6.train_multiclass_mlp_treatment_dose.ipynb`.
#
# It re-derives the held-out test-set predictions (same split, same deterministically-refit `StandardScaler`, saved model) and saves: per-cell predictions, overall and per-class accuracy/precision/recall/F1, and top-k accuracy (since a wrong top-1 guess that still lands on the correct mechanism-of-action neighbor, e.g. confusing two MEK inhibitors, is a very different kind of error than a wrong guess altogether). Plotting (including the confusion matrix) is done separately in `8.plot_multiclass_mlp_treatment_dose.ipynb` (R).

# In[1]:


import pathlib

import numpy as np
import pandas as pd
from joblib import load
from notebook_init_utils import init_notebook
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

root_dir, in_notebook = init_notebook()


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


# ## Rebuild the same data/label setup used in training
# This mirrors `6.train_multiclass_mlp_treatment_dose.ipynb` exactly, so refitting the `StandardScaler` and `LabelEncoder` on `train_df` reproduces what the saved model was trained under.

# In[3]:


TREATMENT_COL = "Metadata_Experiment_Treatment"
DOSE_COL = "Metadata_Experiment_Dose"
LABEL_COL = "Metadata_treatment_dose_label"

indices_df = pd.read_parquet(split_idx_file_path)
df = pd.read_parquet(profile_file_path)

feature_cols = [col for col in df.columns if not col.startswith("Metadata_")]
df[LABEL_COL] = df[TREATMENT_COL].astype(str) + "_" + df[DOSE_COL].astype(str)

train_idx = indices_df["train_idx"].dropna().astype(int).tolist()
test_idx = indices_df["test_idx"].dropna().astype(int).tolist()
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

label_encoder = LabelEncoder()
label_encoder.fit(train_df[LABEL_COL])
num_classes = len(label_encoder.classes_)

scaler = StandardScaler()
scaler.fit(train_df[feature_cols].values)
X_test = scaler.transform(test_df[feature_cols].values)
y_test = label_encoder.transform(test_df[LABEL_COL])

print(f"Test: {len(test_df)} rows, {num_classes} classes")


# ## Load the trained model

# In[4]:


model = load(model_output_dir / "multiclass_mlp_treatment_dose.joblib")


# ## Score the test set

# In[5]:


preds = model.predict(X_test)
probs = model.predict_proba(X_test)

test_predictions_df = pd.DataFrame(
    {
        "Metadata_true_label": label_encoder.inverse_transform(y_test),
        "Metadata_predicted_label": label_encoder.inverse_transform(preds),
    }
)
test_predictions_df.to_parquet(
    model_results_dir / "multiclass_mlp_treatment_dose_test_predictions.parquet",
    index=False,
)
test_predictions_df.head()


# ## Overall metrics and top-k accuracy

# In[6]:


top_k_values = [1, 2, 3, 5]
top_k_rows = [
    {
        "k": k,
        "top_k_accuracy": top_k_accuracy_score(
            y_test, probs, k=k, labels=np.arange(num_classes)
        ),
    }
    for k in top_k_values
]
top_k_df = pd.DataFrame(top_k_rows)
top_k_df.to_parquet(
    model_results_dir / "multiclass_mlp_treatment_dose_top_k_accuracy.parquet",
    index=False,
)

print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"Macro F1: {f1_score(y_test, preds, average='macro'):.4f}")
print(f"Weighted F1: {f1_score(y_test, preds, average='weighted'):.4f}")
top_k_df


# ## Per-class precision/recall/F1

# In[7]:


report_dict = classification_report(
    y_test,
    preds,
    labels=np.arange(num_classes),
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0,
)
report_df = (
    pd.DataFrame(report_dict)
    .T.reset_index(names="Metadata_class")
    .query("Metadata_class not in ['accuracy']")
)
report_df.to_parquet(
    model_results_dir / "multiclass_mlp_treatment_dose_classification_report.parquet",
    index=False,
)
report_df.sort_values("f1-score", ascending=False)
