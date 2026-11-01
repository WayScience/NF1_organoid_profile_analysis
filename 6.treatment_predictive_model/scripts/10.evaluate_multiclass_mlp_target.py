#!/usr/bin/env python
# coding: utf-8

# This notebook evaluates the global target/MOA multiclass MLP trained in `9.train_multiclass_mlp_target.ipynb`.
#
# It re-derives the train- and test-set predictions (same split, same deterministically-refit `StandardScaler`, saved model) and saves: per-cell predictions, overall and per-class accuracy/precision/recall/F1, and top-k accuracy - each tagged with its split (`train`/`test`) so performance can be compared across splits in `11.plot_multiclass_mlp_target.ipynb` (R).

# In[1]:


import pathlib

import numpy as np
import pandas as pd
from joblib import load
from notebook_init_utils import init_notebook
from sklearn.metrics import (
    balanced_accuracy_score,
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
# This mirrors `9.train_multiclass_mlp_target.ipynb` exactly, so refitting the `StandardScaler` and `LabelEncoder` on `train_df` reproduces what the saved model was trained under.

# In[3]:


LABEL_COL = "Metadata_Experiment_Target"

indices_df = pd.read_parquet(split_idx_file_path)
df = pd.read_parquet(profile_file_path)

feature_cols = [col for col in df.columns if not col.startswith("Metadata_")]

train_idx = indices_df["train_idx"].dropna().astype(int).tolist()
test_idx = indices_df["test_idx"].dropna().astype(int).tolist()
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

label_encoder = LabelEncoder()
label_encoder.fit(train_df[LABEL_COL])
num_classes = len(label_encoder.classes_)

scaler = StandardScaler()
scaler.fit(train_df[feature_cols].values)

print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows, {num_classes} classes")


# ## Load the trained model

# In[4]:


model = load(model_output_dir / "multiclass_mlp_target.joblib")


# ## Score each split
# Train and test are scored the same way so the plotting notebook can compare them directly (e.g. to spot overfitting or systematic over-prediction of particular MoA classes).

# In[5]:


TOP_K_VALUES = [1, 2, 3, 5]
splits = {"train": train_df, "test": test_df}

predictions_parts, top_k_parts, report_parts = [], [], []

for split_name, split_df in splits.items():
    X_split = scaler.transform(split_df[feature_cols].values)
    y_split = label_encoder.transform(split_df[LABEL_COL])
    preds = model.predict(X_split)
    probs = model.predict_proba(X_split)

    predictions_parts.append(
        pd.DataFrame(
            {
                "Metadata_split": split_name,
                "Metadata_true_label": label_encoder.inverse_transform(y_split),
                "Metadata_predicted_label": label_encoder.inverse_transform(preds),
            }
        )
    )

    top_k_parts.append(
        pd.DataFrame(
            [
                {
                    "Metadata_split": split_name,
                    "k": k,
                    "top_k_accuracy": top_k_accuracy_score(
                        y_split, probs, k=k, labels=np.arange(num_classes)
                    ),
                }
                for k in TOP_K_VALUES
            ]
        )
    )

    report_df = (
        pd.DataFrame(
            classification_report(
                y_split,
                preds,
                target_names=label_encoder.classes_,
                output_dict=True,
                zero_division=0,
            )
        )
        .T.reset_index(names="Metadata_class")
        .query("Metadata_class not in ['accuracy']")
    )
    report_df.insert(0, "Metadata_split", split_name)
    report_parts.append(report_df)

    print(
        f"[{split_name}] balanced accuracy: {balanced_accuracy_score(y_split, preds):.4f}"
    )
    print(f"[{split_name}] macro F1: {f1_score(y_split, preds, average='macro'):.4f}")


# ## Save per-cell predictions

# In[6]:


predictions_df = pd.concat(predictions_parts, ignore_index=True)
predictions_df.to_parquet(
    model_results_dir / "multiclass_mlp_target_predictions.parquet", index=False
)
predictions_df.head()


# ## Save top-k accuracy

# In[7]:


top_k_df = pd.concat(top_k_parts, ignore_index=True)
top_k_df.to_parquet(
    model_results_dir / "multiclass_mlp_target_top_k_accuracy.parquet", index=False
)
top_k_df


# ## Save per-class precision/recall/F1

# In[8]:


report_df = pd.concat(report_parts, ignore_index=True)
report_df.to_parquet(
    model_results_dir / "multiclass_mlp_target_classification_report.parquet",
    index=False,
)
report_df.sort_values(["Metadata_split", "f1-score"], ascending=[True, False])
