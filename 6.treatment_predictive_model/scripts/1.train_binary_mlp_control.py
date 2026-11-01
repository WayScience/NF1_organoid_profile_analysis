#!/usr/bin/env python
# coding: utf-8

# The goal of this notebook is to train a per-patient model that classifies cells as DMSO-like (negative control) or cytotoxic-like (positive control) based on their image-based morphological profiles.
# Applying this model to treated cells lets us measure how similar each cell is to the negative and positive controls.
# We are particularly interested in cells that differ from DMSO but do not resemble the cytotoxic phenotype, as these may reflect treatment-specific, non-toxic responses.
# To identify these cells, we [describe approach, e.g., combine the classifier probability with each cell's distance from the DMSO profile].

# In[1]:


import pathlib

import joblib
import numpy as np
import pandas as pd
from notebook_init_utils import init_notebook
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
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
).resolve()
model_results_dir = pathlib.Path(
    f"{root_dir}/6.treatment_predictive_model/model_results"
).resolve()
model_output_dir.mkdir(parents=True, exist_ok=True)
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


# ## Define the two classes
# `DMSO` is the negative control (vehicle only, "normal" morphology) and `Staurosporine` is the cytotoxic positive control. Every other treatment is excluded from training - the resulting classifier will later be applied to those treated cells to score how DMSO-like vs. cytotoxic-like they appear.
#
# A separate MLP is trained per patient, since patients are expected to differ in their baseline (DMSO) and cytotoxic-response morphology.

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


# ## MLP model and training/evaluation helpers
# Training uses `sklearn.neural_network.MLPClassifier` via `partial_fit`, one epoch at a time, so that early stopping can watch the same held-out `val_df` split used elsewhere in this pipeline (sklearn's own `early_stopping` option instead carves its validation fold out of the training data, which would not match that split). `MLPClassifier` has no dropout layer - `alpha` (L2 weight decay) is the closest packaged equivalent and is kept deliberately, since dropout's regularization effect is being traded for a maintained/packaged solver here. Class imbalance (previously handled with `BCEWithLogitsLoss(pos_weight=...)`) is instead handled by oversampling the minority class before each patient's training run, since `MLPClassifier.partial_fit` does not accept sample weights.

# In[6]:


BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIMS = (128, 64)


def balance_classes(
    X: np.ndarray, y: np.ndarray, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class so the training set is 50/50."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == n_neg:
        return X, y

    minority_label = 1 if n_pos < n_neg else 0
    majority_count = max(n_pos, n_neg)
    minority_mask = y == minority_label

    X_minority, y_minority = resample(
        X[minority_mask],
        y[minority_mask],
        replace=True,
        n_samples=majority_count,
        random_state=random_state,
    )
    X_balanced = np.concatenate([X[~minority_mask], X_minority])
    y_balanced = np.concatenate([y[~minority_mask], y_minority])
    return X_balanced, y_balanced


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[MLPClassifier, dict]:
    X_train, y_train = balance_classes(X_train, y_train)

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_DIMS,
        activation="relu",
        solver="adam",
        alpha=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        learning_rate_init=LEARNING_RATE,
        max_iter=1,
        warm_start=True,
        random_state=0,
    )

    best_val_loss = float("inf")
    best_params = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}
    rng = np.random.RandomState(0)

    for epoch in range(MAX_EPOCHS):
        perm = rng.permutation(len(X_train))
        if epoch == 0:
            model.partial_fit(X_train[perm], y_train[perm], classes=[0, 1])
        else:
            model.partial_fit(X_train[perm], y_train[perm])

        train_loss = log_loss(y_train, model.predict_proba(X_train)[:, 1])
        val_loss = log_loss(y_val, model.predict_proba(X_val)[:, 1])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = (
                [w.copy() for w in model.coefs_],
                [b.copy() for b in model.intercepts_],
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.coefs_, model.intercepts_ = best_params
    return model, history


def evaluate_mlp(model: MLPClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": balanced_accuracy_score(y, preds),
        "f1": f1_score(y, preds, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, probs) if len(np.unique(y)) > 1 else np.nan,
    }
    return metrics


# ## Train one MLP per patient
# Features are standardized per patient using only that patient's training rows. Patients without both classes present in every split (e.g. no `Staurosporine` wells) are skipped.

# In[7]:


patients = sorted(train_df[PATIENT_COL].unique())
per_patient_metrics = []
per_patient_histories = {}

for patient in progress_bar(patients, desc="Training per-patient MLPs"):
    patient_train = train_df[train_df[PATIENT_COL] == patient]
    patient_val = val_df[val_df[PATIENT_COL] == patient]
    patient_test = test_df[test_df[PATIENT_COL] == patient]

    if (
        patient_train[LABEL_COL].nunique() < 2
        or patient_val[LABEL_COL].nunique() < 2
        or patient_test[LABEL_COL].nunique() < 2
    ):
        print(f"Skipping {patient}: missing a class in train/val/test.")
        continue

    scaler = StandardScaler()
    X_train = scaler.fit_transform(patient_train[feature_cols].values)
    X_val = scaler.transform(patient_val[feature_cols].values)
    X_test = scaler.transform(patient_test[feature_cols].values)

    y_train = patient_train[LABEL_COL].values
    y_val = patient_val[LABEL_COL].values
    y_test = patient_test[LABEL_COL].values

    model, history = train_mlp(X_train, y_train, X_val, y_val)
    per_patient_histories[patient] = history

    test_metrics = evaluate_mlp(model, X_test, y_test)
    test_metrics[PATIENT_COL] = patient
    test_metrics["n_train"] = len(patient_train)
    test_metrics["n_val"] = len(patient_val)
    test_metrics["n_test"] = len(patient_test)
    test_metrics["n_epochs_trained"] = len(history["train_loss"])
    per_patient_metrics.append(test_metrics)

    model_output_path = model_output_dir / f"binary_mlp_control_{patient}.joblib"
    joblib.dump({"model": model, "scaler": scaler}, model_output_path)

metrics_df = pd.DataFrame(per_patient_metrics)
metrics_df


# In[8]:


metrics_output_path = model_results_dir / "binary_mlp_control_test_metrics.parquet"
metrics_df.to_parquet(metrics_output_path, index=False)


# ## Save per-patient training curves
# Plotting (per-patient test metrics, training curves) is done separately in `3.plot_binary_mlp_control_performance.ipynb` (R) - this notebook only computes and saves the underlying data.

# In[9]:


history_records = []
for patient, history in per_patient_histories.items():
    history_records.append(
        pd.DataFrame(
            {
                PATIENT_COL: patient,
                "epoch": range(1, len(history["train_loss"]) + 1),
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
            }
        )
    )

history_df = pd.concat(history_records, ignore_index=True)
history_output_path = model_results_dir / "binary_mlp_control_training_curves.parquet"
history_df.to_parquet(history_output_path, index=False)
history_df.head()
