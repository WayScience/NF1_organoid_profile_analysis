#!/usr/bin/env python
# coding: utf-8

# The goal of this notebook is to train a per-patient model that classifies cells as DMSO-like (negative control) or cytotoxic-like (positive control) based on their image-based morphological profiles.
# Applying this model to treated cells lets us measure how similar each cell is to the negative and positive controls.
# We are particularly interested in cells that differ from DMSO but do not resemble the cytotoxic phenotype, as these may reflect treatment-specific, non-toxic responses.
# To identify these cells, we [describe approach, e.g., combine the classifier probability with each cell's distance from the DMSO profile].

# In[1]:


import pathlib

import numpy as np
import pandas as pd
from notebook_init_utils import init_notebook
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
).resolve()
split_idx_file_path.parent.mkdir(parents=True, exist_ok=True)


# In[3]:


df = pd.read_parquet(profile_file_path)
df.head()


# In[4]:


df["Metadata_patient_treatment_dose"] = (
    df["Metadata_Biology_PatientTumor"].astype(str)
    + "_"
    + df["Metadata_Experiment_Treatment"].astype(str)
    + "_"
    + df["Metadata_Experiment_Dose"].astype(str)
)

# Drop rows with missing values in the morphology feature columns, or in the
# columns used to build the stratification key above. Several other
# Metadata_* columns (e.g. Metadata_Experiment_ViabilityPercentage, which is
# normalized relative to DMSO and therefore undefined for DMSO itself) are
# NaN by design for some treatments - a blanket dropna() would silently
# remove every DMSO (negative control) row.
feature_cols = [col for col in df.columns if not col.startswith("Metadata_")]
strata_dependency_cols = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Dose",
]
df.dropna(subset=feature_cols + strata_dependency_cols, inplace=True)
df.head()


# ## Data splitting
# 70 % training, 15 % validation, 15 % test split. The training and validation sets will be used to train the model, while the test set will be used to evaluate its performance on unseen data.

# In[5]:


# data splitting
train_frac = 0.7
val_frac = 0.15
test_frac = 0.15
strata_col = "Metadata_patient_treatment_dose"

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df, test_size=test_frac, stratify=df[strata_col], random_state=0, shuffle=True
)
train_df, val_df = train_test_split(
    train_df,
    test_size=val_frac,
    stratify=train_df[strata_col],
    random_state=0,
    shuffle=True,
)

# get the indexes of the train, validation, and test sets
train_idx = train_df.index
val_idx = val_df.index
test_idx = test_df.index

print(f"Train set size: {len(train_idx)}")
print(f"Validation set size: {len(val_idx)}")
print(f"Test set size: {len(test_idx)}")
if len(train_idx) + len(val_idx) + len(test_idx) != len(df):
    raise ValueError(
        "The sum of the lengths of the train, validation, and test sets does not equal the length of the original dataframe."
    )


# In[6]:


# save the indexes of the train, validation, and test sets to a parquet file

split_idx_df = pd.DataFrame(
    {
        "train_idx": pd.Series(train_idx),
        "val_idx": pd.Series(val_idx),
        "test_idx": pd.Series(test_idx),
    }
)
split_idx_df.to_parquet(split_idx_file_path, index=False)
