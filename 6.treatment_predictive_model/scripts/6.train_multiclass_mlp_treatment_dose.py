#!/usr/bin/env python
# coding: utf-8

# This notebook trains a single global multiclass MLP that predicts a cell's treatment+dose identity (e.g. `Trametinib_10.0`) directly from its image-based morphological profile, across all patients at once.
#
# Unlike `1.train_binary_mlp_control.ipynb` (one DMSO-vs-Staurosporine model per patient), this asks a different question: how distinguishable are treatments from each other by morphology alone, pooling across patients? It reuses the same train/val/test split from `0.generate_data_splits.ipynb` (already stratified by patient + treatment + dose), restricted to no additional rows - every treatment, dose, and patient is included.

# In[1]:


import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from notebook_init_utils import init_notebook
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
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
).resolve()
model_results_dir = pathlib.Path(
    f"{root_dir}/6.treatment_predictive_model/model_results"
).resolve()
model_output_dir.mkdir(parents=True, exist_ok=True)
model_results_dir.mkdir(parents=True, exist_ok=True)


# ## Load data and define the treatment+dose label
# Every row from `0.generate_data_splits.ipynb`'s split is used (all treatments, all doses, all patients) - this is not restricted to controls.

# In[3]:


TREATMENT_COL = "Metadata_Experiment_Treatment"
DOSE_COL = "Metadata_Experiment_Dose"
LABEL_COL = "Metadata_treatment_dose_label"

indices_df = pd.read_parquet(split_idx_file_path)
df = pd.read_parquet(profile_file_path)

feature_cols = [col for col in df.columns if not col.startswith("Metadata_")]
df[LABEL_COL] = df[TREATMENT_COL].astype(str) + "_" + df[DOSE_COL].astype(str)

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


# ## MLP model and training/evaluation helpers
# A single global model (not per-patient): features are standardized once, fit on the full training split.

# In[4]:


BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIMS = (256, 128)


class MulticlassMLP(nn.Module):
    """A feed-forward multiclass classifier over single-cell morphology features."""

    def __init__(
        self, input_dim: int, num_classes: int, hidden_dims: tuple = HIDDEN_DIMS
    ):
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
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    # drop_last avoids handing BatchNorm1d a final batch of size 1 while
    # training (it requires >1 sample per channel to compute batch statistics).
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weight: torch.Tensor,
) -> tuple:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for _ in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        n_train_seen = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            n_train_seen += X_batch.size(0)
        train_loss = running_loss / n_train_seen

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_running_loss += criterion(
                    model(X_batch), y_batch
                ).item() * X_batch.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, history


def predict(model: nn.Module, data_loader: DataLoader) -> tuple:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            all_logits.append(model(X_batch.to(device)).cpu())
            all_labels.append(y_batch)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = logits.argmax(axis=1)
    return labels, preds, logits


# ## Train the model
# Class weights (inverse frequency, computed from the training split) counteract the imbalance across treatments/doses (e.g. `DMSO_1.0` has far more cells than `Vistusertib_1.0`).

# In[5]:


scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].values)
X_val = scaler.transform(val_df[feature_cols].values)
X_test = scaler.transform(test_df[feature_cols].values)

y_train = label_encoder.transform(train_df[LABEL_COL])
y_val = label_encoder.transform(val_df[LABEL_COL])
y_test = label_encoder.transform(test_df[LABEL_COL])

train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)
test_loader = make_loader(X_test, y_test, BATCH_SIZE, shuffle=False)

class_counts = np.bincount(y_train, minlength=num_classes)
class_weight = torch.tensor(
    len(y_train) / (num_classes * class_counts), dtype=torch.float32, device=device
)

model = MulticlassMLP(input_dim=len(feature_cols), num_classes=num_classes).to(device)
model, history = train_mlp(model, train_loader, val_loader, class_weight)

print(f"Trained for {len(history['train_loss'])} epochs")


# ## Save the model and training curve
# Plotting is done separately in `8.plot_multiclass_mlp_treatment_dose.ipynb` (R).

# In[6]:


model_output_path = model_output_dir / "multiclass_mlp_treatment_dose.pt"
torch.save(model.state_dict(), model_output_path)

history_df = pd.DataFrame(
    {
        "epoch": range(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
    }
)
history_df.to_parquet(
    model_results_dir / "multiclass_mlp_treatment_dose_training_curve.parquet",
    index=False,
)
history_df.head()


# ## Quick test-set check
# A full evaluation (confusion matrix, per-class precision/recall, top-k accuracy) is done separately in `7.evaluate_multiclass_mlp_treatment_dose.ipynb`; this is just a sanity check that training worked.

# In[7]:


from sklearn.metrics import accuracy_score, f1_score

y_true, y_pred, _ = predict(model, test_loader)
print(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Test macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
