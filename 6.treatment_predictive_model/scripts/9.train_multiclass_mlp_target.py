#!/usr/bin/env python
# coding: utf-8

# This notebook trains a single global multiclass MLP that predicts a cell's drug target / mechanism-of-action (MOA) class (e.g. `MEK1/2 inhibitor`, `mTOR inhibitor`, or `Control` for DMSO) directly from its image-based morphological profile, across all patients at once.
#
# This is a coarser-grained version of `6.train_multiclass_mlp_treatment_dose.ipynb`'s question - grouping individual drugs by shared mechanism should be an easier target to predict from morphology than the exact drug identity, since drugs sharing a MOA are expected to produce more similar phenotypes. It reuses the same train/val/test split from `0.generate_data_splits.ipynb`.

# In[ ]:


import pathlib

import numpy as np
import pandas as pd
from joblib import dump
from notebook_init_utils import init_notebook
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

root_dir, in_notebook = init_notebook()

RANDOM_STATE = 0


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


# ## Load data
# `Metadata_Experiment_Target` is already annotated per row (e.g. `DMSO` -> `Control`, `Trametinib` -> `MEK1/2 inhibitor`) - no extra label construction needed. Every row from the split is used (all targets, all patients).

# In[3]:


LABEL_COL = "Metadata_Experiment_Target"

indices_df = pd.read_parquet(split_idx_file_path)
df = pd.read_parquet(profile_file_path)

feature_cols = [col for col in df.columns if not col.startswith("Metadata_")]

train_idx = indices_df["train_idx"].dropna().astype(int).tolist()
val_idx = indices_df["val_idx"].dropna().astype(int).tolist()
test_idx = indices_df["test_idx"].dropna().astype(int).tolist()
train_df, val_df, test_df = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

# Encoder fit on train only - the stratified split guarantees every class
# appears in train, so val/test never see an unseen label.
label_encoder = LabelEncoder()
label_encoder.fit(train_df[LABEL_COL])
num_classes = len(label_encoder.classes_)

print(
    f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows"
)
print(f"Number of features: {len(feature_cols)}, number of classes: {num_classes}")
print(f"Classes: {list(label_encoder.classes_)}")


# ## MLP model and training helper
# A single global model (not per-patient): features are standardized once, fit on the full training split. Training uses `sklearn.neural_network.MLPClassifier` via `partial_fit`, one epoch at a time, mirroring `1.train_binary_mlp_control.ipynb`/`6.train_multiclass_mlp_treatment_dose.ipynb`'s pattern so that early stopping watches the same held-out `val_df` split used elsewhere in this pipeline, rather than sklearn's own `early_stopping` option, which would carve its validation fold out of the training data instead (and, since that option only exposes a validation *score* rather than a validation *loss*, couldn't be plotted on the same loss axis as the other training curves in `11.plot_multiclass_mlp_target.ipynb`).

# In[ ]:


MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization, passed as MLPClassifier's `alpha`
HIDDEN_DIMS = (128, 64)
BATCH_SIZE = 128


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    classes: np.ndarray,
) -> tuple[MLPClassifier, dict]:
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_DIMS,
        activation="relu",
        solver="adam",
        alpha=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        learning_rate_init=LEARNING_RATE,
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_STATE,
    )

    best_val_loss = float("inf")
    best_params = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}
    rng = np.random.RandomState(RANDOM_STATE)

    for epoch in range(MAX_EPOCHS):
        perm = rng.permutation(len(X_train))
        if epoch == 0:
            model.partial_fit(X_train[perm], y_train[perm], classes=classes)
        else:
            model.partial_fit(X_train[perm], y_train[perm])

        train_loss = log_loss(y_train, model.predict_proba(X_train), labels=classes)
        val_loss = log_loss(y_val, model.predict_proba(X_val), labels=classes)
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


# ## Train the model

# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].values)
X_val = scaler.transform(val_df[feature_cols].values)
X_test = scaler.transform(test_df[feature_cols].values)

y_train = label_encoder.transform(train_df[LABEL_COL])
y_val = label_encoder.transform(val_df[LABEL_COL])
y_test = label_encoder.transform(test_df[LABEL_COL])

model, history = train_mlp(X_train, y_train, X_val, y_val, np.arange(num_classes))

print(f"Trained for {len(history['train_loss'])} epochs")


# ## Save the model and training curve
# Plotting is done separately in `11.plot_multiclass_mlp_target.ipynb` (R).

# In[ ]:


model_output_path = model_output_dir / "multiclass_mlp_target.joblib"
dump(model, model_output_path)

history_df = pd.DataFrame(
    {
        "epoch": range(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
    }
)
history_df.to_parquet(
    model_results_dir / "multiclass_mlp_target_training_curve.parquet", index=False
)
history_df.head()


# ## Quick test-set check
# A full evaluation (confusion matrix, per-class precision/recall, top-k accuracy) is done separately in `10.evaluate_multiclass_mlp_target.ipynb`; this is just a sanity check that training worked.

# In[ ]:


y_pred = model.predict(X_test)
print(f"Test balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"Test macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
